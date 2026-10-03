import pandas as pd

from summarize_results import aggregate


def _history(path, best, pool_min):
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"iter": [1, 2], "best_bo": [best + 1, best], "best_pool_min": [pool_min] * 2}
    ).to_csv(path, index=False)
    return path


def test_pool_optimum_is_per_seed(tmp_path):
    """Bug 8: pool minimum of the first run was used for every seed."""
    f1 = _history(tmp_path / "seed1" / "h.csv", best=-10.0, pool_min=-10.0)
    f2 = _history(tmp_path / "seed2" / "h.csv", best=-5.0, pool_min=-5.0)  # optimum in test split
    row = aggregate("ds", "m", [f1, f2], "best_bo")
    assert row["n_hit_pool_min"] == 2
