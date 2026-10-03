# -*- coding: utf-8 -*-
"""
Gaussian Process model implementation for Bayesian Optimization.
"""

from typing import Literal, List, Tuple
import logging

import gpytorch
import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior

# Fix relative import
from src.kernels import get_kernel

logger = logging.getLogger(__name__)


def build_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    kernel_name: Literal["matern", "rbf", "rq"] = "matern",
    kernel_params: dict = None,
) -> SingleTaskGP:
    """
    Create a single-output exact GP with specified kernel.

    Args:
        train_x: Training inputs [n_points, n_features]
        train_y: Training targets [n_points, 1]
        kernel_name: Kernel type ("matern", "rbf", or "rq")
        kernel_params: Additional kernel-specific parameters

    Returns:
        Configured SingleTaskGP model
    """
    # Set up likelihood with informative noise prior
    # Gamma(1.1, 0.05) favors noise around 0.01-0.1 (typical for normalized data)
    noise_prior = GammaPrior(1.1, 0.05)
    likelihood = GaussianLikelihood(noise_prior=noise_prior)

    # Get kernel with parameters
    kernel_kwargs = {"ard": True}  # Enable ARD for molecular descriptors
    if kernel_params:
        kernel_kwargs.update(kernel_params)

    covar_module = get_kernel(
        kernel_name=kernel_name,
        input_dim=train_x.shape[-1],
        **kernel_kwargs,
    )

    # Add informative priors to kernel hyperparameters
    # Length-scale prior: Gamma(3.0, 6.0) favors length-scales around 0.3-0.8
    # This prevents both very small (overfitting) and very large (underfitting) values
    covar_module.base_kernel.register_prior(
        "lengthscale_prior",
        GammaPrior(3.0, 6.0),
        lambda m: m.lengthscale,
        lambda m, v: m._set_lengthscale(v),
    )

    # Output scale prior: Gamma(2.0, 0.15) favors signal variance around 5-20
    # Prevents extreme signal amplitudes
    if hasattr(covar_module, "outputscale"):
        covar_module.register_prior(
            "outputscale_prior",
            GammaPrior(2.0, 0.15),
            lambda m: m.outputscale,
            lambda m, v: m._set_outputscale(v),
        )

    return SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        covar_module=covar_module,
        likelihood=likelihood,
    )


def fit_gp(
    model: SingleTaskGP,
    lr: float = 0.1,
    maxiter: int = 300,
    n_restarts: int = 3,
    patience: int = 20,
) -> dict:
    """
    Fit GP hyperparameters by maximizing marginal log-likelihood with multi-restart optimization.

    Uses multiple random restarts (perturbations of the initial hyperparameters) with
    adaptive learning rate scheduling to find better optima.
    Monitors convergence and tracks optimization progress.

    Args:
        model: GP model to fit
        lr: Initial learning rate for optimization
        maxiter: Maximum number of optimization steps per restart
        n_restarts: Number of random restarts (default: 3)
        patience: Patience for LR scheduler (default: 20)

    Returns:
        dict: Convergence information including:
            - converged: bool, whether optimization converged
            - final_loss: float, final negative log-likelihood
            - n_iterations: int, total number of iterations performed
            - loss_history: list, full loss trajectory from best restart
            - final_lml: float, final marginal log-likelihood value
            - best_restart: int, which restart achieved best result
    """
    model.train()
    model.likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    lr = float(lr) if not isinstance(lr, torch.Tensor) else float(lr.item())

    # Track best across restarts
    best_loss = float("inf")
    best_state = None
    best_loss_history = []
    best_restart_idx = 0
    total_iterations = 0

    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    for restart in range(n_restarts):
        # Random restarts perturb the *initial* hyperparameters (restart 0 is unperturbed);
        # starting from the previous restart's fitted values would not be a restart.
        if restart > 0:
            model.load_state_dict(initial_state)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "raw_" in name and param.requires_grad:
                        param.add_(0.5 * torch.randn_like(param))

        # Optimizer with LR scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience, min_lr=1e-5
        )

        # Track convergence for this restart
        loss_history = []
        converged = False
        convergence_threshold = 1e-4
        warmup_iterations = 20

        for i in range(maxiter):
            optimizer.zero_grad()
            output = model(model.train_inputs[0])
            loss = -mll(output, model.train_targets.squeeze(-1))
            loss.backward()

            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(loss)

            # Track loss
            loss_history.append(loss.item())
            total_iterations += 1

            # Check convergence after warmup period
            if i >= warmup_iterations and len(loss_history) >= 20:
                # Check if loss has plateaued over last 20 iterations
                recent_mean = np.mean(loss_history[-20:])
                older_mean = (
                    np.mean(loss_history[-40:-20])
                    if len(loss_history) >= 40
                    else loss_history[0]
                )
                relative_change = abs(recent_mean - older_mean) / (
                    abs(older_mean) + 1e-8
                )

                if relative_change < convergence_threshold:
                    converged = True
                    break

        # Check if this restart is better
        final_loss = loss_history[-1] if loss_history else float("inf")
        if final_loss < best_loss:
            best_loss = final_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_loss_history = loss_history.copy()
            best_restart_idx = restart

    # Restore best state across all restarts
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model.likelihood.eval()

    # Compute final log marginal likelihood with best hyperparameters
    with torch.no_grad():
        final_output = model(model.train_inputs[0])
        final_lml = mll(final_output, model.train_targets.squeeze(-1)).item()

    # Overall convergence: did we converge in at least one restart?
    overall_converged = converged or (
        len(best_loss_history) > warmup_iterations
        and abs(
            best_loss_history[-1]
            - best_loss_history[-min(20, len(best_loss_history) // 2)]
        )
        < convergence_threshold
        * abs(best_loss_history[-min(20, len(best_loss_history) // 2)])
    )

    return {
        "converged": overall_converged,
        "final_loss": best_loss,
        "n_iterations": total_iterations,
        "loss_history": best_loss_history,
        "final_lml": final_lml,
        "best_restart": best_restart_idx,
    }


def check_gp_stability(
    model: SingleTaskGP, iteration: int, logger_obj=None
) -> Tuple[bool, List[str]]:
    """
    Check for numerical instability in GP hyperparameters.

    Args:
        model: The fitted GP model
        iteration: Current BO iteration (for logging)
        logger_obj: Logger object for warnings

    Returns:
        Tuple of (is_stable, list_of_issues)
    """
    if logger_obj is None:
        logger_obj = logger

    issues = []

    # Extract hyperparameters
    try:
        # Get length-scales (can be vector with ARD)
        lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        if lengthscale.ndim > 1:
            lengthscale = lengthscale.flatten()

        # Get signal variance (outputscale)
        if hasattr(model.covar_module, "outputscale"):
            outputscale = model.covar_module.outputscale.detach().cpu().item()
        else:
            outputscale = 1.0  # No outputscale in some kernels

        # Get noise
        noise = model.likelihood.noise.detach().cpu().item()

        # Check for extreme values
        if np.any(lengthscale < 1e-6):
            issues.append(f"Very small length-scale: min={lengthscale.min():.2e}")
        if np.any(lengthscale > 1e3):
            issues.append(f"Very large length-scale: max={lengthscale.max():.2e}")
        if outputscale > 1e3:
            issues.append(f"Very large signal variance: {outputscale:.2e}")
        if noise < 1e-6:
            issues.append(f"Very small noise: {noise:.2e}")
        if noise > 10:
            issues.append(f"Very large noise: {noise:.2f}")

        # Check for NaN/Inf
        if np.any(np.isnan(lengthscale)) or np.any(np.isinf(lengthscale)):
            issues.append("NaN/Inf in length-scales")
        if np.isnan(outputscale) or np.isinf(outputscale):
            issues.append("NaN/Inf in signal variance")
        if np.isnan(noise) or np.isinf(noise):
            issues.append("NaN/Inf in noise")

        if issues:
            logger_obj.warning(
                f"[Iter {iteration}] GP stability issues: {'; '.join(issues)}"
            )
            return False, issues

        return True, []

    except Exception as e:
        logger_obj.error(f"[Iter {iteration}] Error checking GP stability: {e}")
        return False, [f"Exception: {str(e)}"]
