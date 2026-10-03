from __future__ import annotations

import os
import time
import logging
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import gpytorch
from botorch.acquisition import (
    qLogExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import psutil

# Fix relative imports to absolute imports
from src.acquisition import make_acquisition
from src.data_io import append_to_cache, load_descriptors, load_or_init_cache, pool_min, prune_columns
from src.feature_selection.base import FeatureSelector
from src.gp_model import build_gp, fit_gp, check_gp_stability
from src.utils import logger, seed_everything

POOL_CHUNK = 256


def score_pool(acq, pool_x: torch.Tensor, chunk: int = POOL_CHUNK) -> np.ndarray:
    """Evaluate a q=1 acquisition over the pool in chunks without autograd.

    A full-pool batch makes the posterior build an [N, n_train, d] tensor, so
    memory grows with pool size and training-set size; chunking bounds it.
    """
    x = pool_x.unsqueeze(1) if pool_x.dim() == 2 else pool_x
    with torch.no_grad():
        vals = [acq(x[i : i + chunk]).detach().cpu() for i in range(0, x.shape[0], chunk)]
    return torch.cat(vals).squeeze(-1).numpy()

# ──────────────────────────────────────────────────────────────────────────
# metrics
# ──────────────────────────────────────────────────────────────────────────
_rmse = lambda pred, true: float(np.sqrt(np.mean((pred - true) ** 2)))
_ae = lambda pred, true: float(np.mean(np.abs(pred - true)))


def _gp_hparams(model) -> Tuple[np.ndarray, float, float]:
    """Return (length‑scales, output‑scale, noise) of GP"""
    l = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel().copy()
    sigma2 = float(model.covar_module.outputscale.item())
    noise = float(model.likelihood.noise.item())
    return l, sigma2, noise


def log_memory_usage():
    """Log current memory usage in GB."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / 1024 / 1024 / 1024
        return memory_gb
    except:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────
# pipeline
# ──────────────────────────────────────────────────────────────────────────
class BOPipeline:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.logger: logging.Logger = logger

    def selector_factory(self) -> Optional[FeatureSelector]:
        """
        Factory for feature selectors. By default, no selection;
        subclasses override this when feature selection is needed.
        """
        return None

    def run(self, input_csv: str, cache_path: Path, simulator):
        # 0) lock down all RNGs
        seed_everything(self.cfg["random_state"])

        # 1) load full descriptor set
        df = load_descriptors(Path(input_csv))
        smiles_all = df["SMILES"].tolist()
        X_full_df = df.drop(columns="SMILES")

        # optional unsupervised column pruning on the whole X table (no targets)
        prune_threshold = self.cfg.get("data", {}).get("prune_correlated")
        kept_columns = None
        if prune_threshold is not None:
            kept_columns = prune_columns(X_full_df, prune_threshold)
            self.logger.info(
                f"pruned {X_full_df.shape[1]} -> {len(kept_columns)} columns "
                f"(threshold {prune_threshold})"
            )
            X_full_df = X_full_df[kept_columns]

        # 2) carve off test set
        X_tmp, X_test_df, smi_tmp, smi_test = train_test_split(
            X_full_df,
            smiles_all,
            test_size=self.cfg.get("data", {}).get("test_frac", 0.1),
            random_state=self.cfg["random_state"],
            shuffle=True,
        )

        # 3) carve off n_initial for starting design
        X_init_df, X_pool_df, smi_init, smi_pool = train_test_split(
            X_tmp,
            smi_tmp,
            train_size=self.cfg["data"]["n_initial"],
            random_state=self.cfg["random_state"],
            shuffle=True,
        )

        # 4) reset indices
        train_raw_df = X_init_df.reset_index(drop=True)
        pool_raw_df = X_pool_df.reset_index(drop=True)
        test_raw_df = X_test_df.reset_index(drop=True)
        smi_train = smi_init.copy()

        # 5) fetch / simulate y for initial
        cache = load_or_init_cache(cache_path)
        train_y_raw = []
        for k, smi in enumerate(smi_train, start=1):
            if smi in cache:
                y_val = cache[smi]
                self.logger.info(f"Cache hit for initial sample: {smi}")
            else:
                self.logger.info(f"Computing Binding Free Energy for: {smi}")
                y_val = simulator.compute(smi, index=k, total=len(smi_train))
                append_to_cache(smi, float(y_val), cache_path)
                cache[smi] = y_val
            train_y_raw.append(float(y_val))

        # helpers to scale X and normalise y
        # Pool-based BO knows every X up front (no target information is used), so the
        # MinMax scaler is fitted once on train + pool + test, i.e. on the whole table.
        x_scaler = MinMaxScaler().fit(X_full_df.to_numpy(dtype=float))

        def to_unit(*arrays):
            """MinMax-fit on all given arrays together -> torch tensors in [0, 1]."""
            s_ = MinMaxScaler().fit(np.vstack(arrays))
            return tuple(
                torch.tensor(s_.transform(a), dtype=torch.double) for a in arrays
            )

        def featurize(initial: bool):
            """Scaled raw X -> (optional) selector fitted on the current training set
            -> GP inputs. prepare() and update() always receive the same
            representation (scaled X) and both refit on the current train X, y."""
            arrays = [
                x_scaler.transform(d.to_numpy(dtype=float))
                for d in (train_raw_df, pool_raw_df, test_raw_df)
            ]
            if selector is None:
                return tuple(torch.tensor(a, dtype=torch.double) for a in arrays)
            fit = selector.prepare if initial else selector.update
            out = fit(arrays[0], np.asarray(train_y_raw, dtype=float), arrays[1], arrays[2])
            self._last_transformed_data = dict(
                zip(("train", "pool", "test"), (t.numpy() for t in out))
            )
            # selector outputs (PCA/PLS scores) have arbitrary range; bring them to
            # the unit scale the GP priors assume
            return to_unit(*(t.numpy() for t in out))

        def normalise_y(y_list: List[float]):
            arr = np.asarray(y_list, dtype=float)
            # Use robust scaling: median and interquartile range instead of mean/std
            # This is less sensitive to outliers and preserves relative differences better
            q75, q25 = np.percentile(arr, [75, 25])
            iqr = q75 - q25
            if iqr < 1e-8:  # If IQR is too small, fall back to std
                mu, sig = float(arr.mean()), float(arr.std() + 1e-8)
            else:
                mu, sig = float(np.median(arr)), float(
                    iqr / 1.35
                )  # 1.35 converts IQR to std equivalent

            return (
                torch.tensor(((arr - mu) / sig), dtype=torch.double).unsqueeze(-1),
                mu,
                sig,
            )

        # optional feature-selector
        selector = self.selector_factory()

        # 6) initial scaling / feature selection / normalization
        train_x, pool_x, test_x = featurize(initial=True)
        train_y, y_mu, y_sig = normalise_y(train_y_raw)

        print(f"Size of initial training set: {train_x.shape}")
        print(f"Size of pool set: {pool_x.shape}")
        print(f"Size of test set: {test_x.shape}")

        if selector is not None:
            try:
                support = getattr(selector, "get_support", lambda: None)()
                if support is not None:
                    support = len(support) if hasattr(support, "__len__") else support
            except (AttributeError, ValueError):
                support = None
            self.logger.info(f"Feature selection → kept {support} components/features")

        # 7) GP fit (+ hyper‑param log)
        kernel_params = self.cfg["optimization"].get("kernel_params", {})
        gp = build_gp(
            train_x,
            train_y,
            kernel_name=self.cfg["optimization"]["kernel"],
            kernel_params=kernel_params,
        )
        fit_info = fit_gp(
            gp,
            lr=self.cfg.get("training", {}).get("lr", 0.1),
            maxiter=self.cfg.get("training", {}).get("maxiter", 300),
            n_restarts=self.cfg.get("training", {}).get("n_restarts", 3),
            patience=self.cfg.get("training", {}).get("patience", 20),
        )
        l, sigma2, noise = _gp_hparams(gp)

        # Log convergence status
        convergence_msg = "converged" if fit_info["converged"] else "NOT CONVERGED"
        self.logger.info(
            f"GP θ → l={l}, σ²={sigma2:.3g}, noise={noise:.3g}, "
            f"LML={fit_info['final_lml']:.3f}, {convergence_msg} ({fit_info['n_iterations']} iters)"
        )

        # Check numerical stability
        is_stable, stability_issues = check_gp_stability(
            gp, iteration=0, logger_obj=self.logger
        )

        # Initial memory cleanup after GP fitting
        import gc

        gc.collect()

        # 8) baseline test metric
        y_test_true = np.array([cache[s] for s in smi_test], dtype=float)
        with torch.no_grad():
            preds_norm = gp.posterior(test_x).mean.detach().cpu().numpy()
        preds = preds_norm * y_sig + y_mu
        self.logger.info(
            f"Before BO → RMSE={_rmse(preds,y_test_true):.3f}, AE={_ae(preds,y_test_true):.3f}"
        )

        # track best values
        # incumbent starts at the best of the initial design (as in random search)
        best_bo = min(train_y_raw)

        # lowest energy in the initial candidate pool; fixed for the whole run
        best_pool_min = pool_min(smi_pool, cache)

        records = []

        # 9) BO loop
        batch_size = self.cfg["optimization"].get("batch_size", 1)

        for it in range(1, self.cfg["optimization"]["n_iter"] + 1):
            self.logger.info(
                f"=== BO iter {it}/{self.cfg['optimization']['n_iter']} ==="
            )

            # Memory management - periodic cleanup to prevent memory leaks
            # Only runs every 20 iterations to minimize performance impact
            if it % 20 == 0:
                current_memory = log_memory_usage()

                # Aggressive cleanup only if memory usage is high (>2GB)
                if current_memory > 2.0:
                    import gc

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.logger.info(
                        f"Memory cleanup (high usage: {current_memory:.2f}GB)"
                    )
                # Light CUDA cleanup for GPU systems (negligible cost)
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # a) acquisition
            best_f_norm = (best_bo - y_mu) / y_sig
            acq = make_acquisition(
                gp,
                best_f=best_f_norm,
                acq_name=self.cfg["optimization"]["acquisition"],
                beta=self.cfg["optimization"].get("beta", 1.0),
                batch_size=batch_size,
                maximize=False,  # Minimize binding energy (lower = better)
            )

            # Handle batch vs single point acquisition
            if batch_size > 1:
                # For batch acquisition, use BoTorch's optimize_acqf
                from botorch.optim import optimize_acqf

                bounds = torch.tensor(
                    [[0.0] * pool_x.shape[1], [1.0] * pool_x.shape[1]],
                    dtype=torch.double,
                )
                candidates, _ = optimize_acqf(
                    acq_function=acq,
                    bounds=bounds,
                    q=batch_size,
                    num_restarts=20,
                    raw_samples=512,
                )

                # Find closest points in pool
                indices_next = []
                for candidate in candidates:
                    distances = torch.sum((pool_x - candidate.unsqueeze(0)) ** 2, dim=1)
                    idx = torch.argmin(distances).item()
                    indices_next.append(idx)

            else:
                # Single point acquisition
                acq_vals = score_pool(acq, pool_x)
                indices_next = [int(np.argmax(acq_vals))]

            # b) query next point(s)
            new_y_values = []
            for idx_next in indices_next:
                smi_next = smi_pool.pop(idx_next)
                x_next = pool_raw_df.iloc[idx_next]

                if smi_next in cache:
                    y_next = cache[smi_next]
                    self.logger.info(f"Cache hit for BO sample: {smi_next}")
                else:
                    y_next = simulator.compute(
                        smi_next, index=it, total=self.cfg["optimization"]["n_iter"]
                    )
                    append_to_cache(smi_next, float(y_next), cache_path)
                    cache[smi_next] = y_next

                train_y_raw.append(float(y_next))
                new_y_values.append(float(y_next))
                best_bo = min(best_bo, y_next)

                # c) update raw dfs
                train_raw_df = pd.concat(
                    [train_raw_df, x_next.to_frame().T], ignore_index=True
                )
                pool_raw_df = pool_raw_df.drop(idx_next).reset_index(drop=True)

                # Update indices for remaining points in batch
                indices_next = [
                    idx - 1 if idx > idx_next else idx for idx in indices_next
                ]

            # d) re-scale, refit the selector on the current training set
            train_x, pool_x, test_x = featurize(initial=False)

            train_y, y_mu, y_sig = normalise_y(train_y_raw)

            # f) GP re‑fit
            kernel_params = self.cfg["optimization"].get("kernel_params", {})
            gp = build_gp(
                train_x,
                train_y,
                kernel_name=self.cfg["optimization"]["kernel"],
                kernel_params=kernel_params,
            )
            fit_info = fit_gp(
                gp,
                lr=self.cfg.get("training", {}).get("lr", 0.1),
                maxiter=self.cfg.get("training", {}).get("maxiter", 300),
                n_restarts=self.cfg.get("training", {}).get("n_restarts", 3),
                patience=self.cfg.get("training", {}).get("patience", 20),
            )
            l, sigma2, noise = _gp_hparams(gp)

            # Check numerical stability
            is_stable, stability_issues = check_gp_stability(
                gp, iteration=it, logger_obj=self.logger
            )

            convergence_msg = "✓" if fit_info["converged"] else "✗"
            self.logger.info(
                f"GP → l={l}, σ²={sigma2:.3g}, noise={noise:.3g} [{convergence_msg}]"
            )

            # g) evaluate on held‑out
            with torch.no_grad():
                preds_norm = gp.posterior(test_x).mean.detach().cpu().numpy()
            preds = preds_norm * y_sig + y_mu

            rmse = _rmse(preds, y_test_true)
            ae = _ae(preds, y_test_true)
            pearson_r, _ = pearsonr(preds.flatten(), y_test_true)
            r2 = r2_score(y_test_true, preds.flatten())

            # h) pool-min is now fixed and represents the true global minimum from the original pool

            # i) Log memory usage and aggressive cleanup
            memory_gb = log_memory_usage()

            # Aggressive memory cleanup every 10 iterations or if memory > 5GB
            if it % 10 == 0 or memory_gb > 5.0:
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                memory_gb = log_memory_usage()  # Re-check after cleanup

            self.logger.info(
                f"Iter {it} → RMSE={rmse:.3f}, AE={ae:.3f}, r={pearson_r:.3f}, R²={r2:.3f}, "
                f"best_bo={best_bo:.3g}, best_pool_min={best_pool_min:.3g}, Memory={memory_gb:.2f}GB"
            )

            # j) Add overfitting/underfitting warnings
            if r2 > 0.99:
                self.logger.warning(
                    f"[Iter {it}] ⚠️ Very high R² ({r2:.4f}) - possible data leakage or overfitting!"
                )
            elif r2 < -0.5:
                self.logger.warning(
                    f"[Iter {it}] ⚠️ Very negative R² ({r2:.4f}) - severe underfitting!"
                )
            elif r2 < 0.0:
                self.logger.warning(
                    f"[Iter {it}] ⚠️ Negative R² ({r2:.4f}) - model worse than mean prediction!"
                )

            records.append(
                {
                    "iter": it,
                    "rmse": rmse,
                    "ae": ae,
                    "pearson_r": pearson_r,
                    "r2": r2,
                    "best_bo": best_bo,
                    "best_pool_min": best_pool_min,
                    "l": l.tolist(),
                    "sigma2": sigma2,
                    "noise": noise,
                    "lml": fit_info["final_lml"],
                    "gp_converged": fit_info["converged"],
                    "gp_fit_iterations": fit_info["n_iterations"],
                    "gp_stable": is_stable,
                    "gp_stability_issues": (
                        "; ".join(stability_issues) if stability_issues else None
                    ),
                    "batch_size": batch_size,
                    "memory_gb": memory_gb,
                }
            )

        # 10) write out results
        dataset_name = Path(input_csv).stem
        method_name = self.__class__.__name__.replace("Pipeline", "").lower()
        beta_part = (
            f"_beta{self.cfg['optimization']['beta']}"
            if self.cfg["optimization"]["acquisition"].lower() == "ucb"
            else ""
        )
        batch_part = f"_batch{batch_size}" if batch_size > 1 else ""

        dir_name = f"{dataset_name}_{self.cfg['optimization']['kernel']}_{self.cfg['optimization']['acquisition']}{beta_part}_{method_name}{batch_part}"
        out_dir = (
            Path(self.cfg.get("results_dir", "results"))
            / dir_name
            / f"seed{self.cfg['random_state']}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(records).to_csv(out_dir / "bo_iteration_history.csv", index=False)
        if kept_columns is not None:
            (out_dir / "pruned_columns.txt").write_text("\n".join(kept_columns) + "\n")
        self.logger.info(f"Finished BO ✅  (results in {out_dir})")
