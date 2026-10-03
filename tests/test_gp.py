import torch

import src.gp_model as gm


def test_restarts_start_from_the_initial_hyperparameters(monkeypatch):
    """Each restart must be a perturbation of the *initial* state, not of the
    parameters the previous restart already fitted."""
    torch.manual_seed(0)
    x = torch.rand(12, 3, dtype=torch.double)
    y = (3 * x[:, :1] + 0.05 * torch.randn(12, 1, dtype=torch.double))
    gp = gm.build_gp(x, y, kernel_name="matern")
    initial = {k: v.clone() for k, v in gp.state_dict().items()}

    starts = []
    real_adam = torch.optim.Adam

    def spy_adam(params, **kw):
        starts.append({k: v.clone() for k, v in gp.state_dict().items()})
        return real_adam(params, **kw)

    monkeypatch.setattr(torch.optim, "Adam", spy_adam)
    # neutralise perturbations so any drift would come from reusing fitted state
    monkeypatch.setattr(torch, "rand_like", lambda t: torch.full_like(t, 0.5))
    monkeypatch.setattr(torch, "randn_like", lambda t: torch.zeros_like(t))
    gm.fit_gp(gp, lr=0.5, maxiter=30, n_restarts=3, patience=5)

    assert len(starts) == 3
    for s in starts[1:]:
        for k, v in initial.items():
            assert torch.allclose(s[k], v), f"restart did not start from initial {k}"
