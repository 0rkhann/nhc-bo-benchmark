import resource

import torch

from src.acquisition import make_acquisition
from src.gp_model import build_gp
from src.pipelines.base_pipeline import score_pool


def _problem(n_train=20, n_pool=700, d=40):
    torch.manual_seed(0)
    x = torch.rand(n_train, d, dtype=torch.double)
    y = x[:, :1] * 3 + 0.05 * torch.randn(n_train, 1, dtype=torch.double)
    pool = torch.rand(n_pool, d, dtype=torch.double)
    gp = build_gp(x, y, kernel_name="matern")
    acq = make_acquisition(gp, best_f=float(y.min()), acq_name="ei", maximize=False)
    return acq, pool


def test_chunked_scores_match_full_batch():
    acq, pool = _problem()
    full = acq(pool.unsqueeze(1)).detach().squeeze(-1).numpy()
    for chunk in (1, 64, 256, 10_000):
        chunked = score_pool(acq, pool, chunk=chunk)
        assert chunked.shape == full.shape
        assert (chunked == full).all() or abs(chunked - full).max() < 1e-12
        assert chunked.argmax() == full.argmax()


def test_scores_carry_no_autograd_graph():
    acq, pool = _problem(n_pool=50)
    # would raise if a graph-bearing tensor leaked into numpy conversion
    assert score_pool(acq, pool, chunk=16).shape == (50,)


def test_chunking_bounds_peak_memory():
    """Full batch builds an [N, n_train, d] tensor (~N*n*d*8 bytes); chunks must stay far below."""
    acq, pool = _problem(n_train=60, n_pool=3000, d=200)
    full_bytes = 3000 * 60 * 200 * 8  # ~288 MB for that one tensor alone
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    score_pool(acq, pool, chunk=256)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert after - before < full_bytes * 0.75
