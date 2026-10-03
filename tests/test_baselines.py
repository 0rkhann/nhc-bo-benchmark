import numpy as np
import pandas as pd

from baselines.random_search import RandomSearchPipeline
from tests.conftest import split_like_pipeline

N_ITER = 6


def _run(cls, synthetic, n_iter=N_ITER):
    cfg = {"n_initial": 10, "n_iter": n_iter, "random_state": 42,
           "results_dir": str(synthetic["results"]), "test_frac": 0.1}
    cls(cfg).run(str(synthetic["csv"]), synthetic["cache"], simulator=None)
    return pd.read_csv(next(synthetic["results"].rglob("*_history.csv")))


def test_random_search_best_pool_min_is_fixed_at_initial_pool_minimum(synthetic):
    """Bug 8: random search lowered best_pool_min as molecules left the pool."""
    _, pool, _ = split_like_pipeline(synthetic["df"])
    hist = _run(RandomSearchPipeline, synthetic, n_iter=len(pool) - 1)  # drains the pool
    np.testing.assert_allclose(hist["best_pool_min"], min(synthetic["energy"][s] for s in pool))


def test_outlier_search_visits_molecules_by_decreasing_max_abs_zscore(synthetic):
    from baselines.outlier_search import OutlierSearchPipeline

    hist = _run(OutlierSearchPipeline, synthetic)
    df = synthetic["df"]
    X = df.drop(columns="SMILES").to_numpy()
    sd = X.std(axis=0)
    z = np.abs((X - X.mean(axis=0)) / np.where(sd > 0, sd, 1)).max(axis=1)  # all X
    _, pool, _ = split_like_pipeline(df)
    score = dict(zip(df.SMILES, z))
    expected = sorted(pool, key=lambda s: -score[s])[:N_ITER]
    assert hist["smiles"].tolist() == expected
    best = min(synthetic["energy"][s] for s in expected)
    assert hist["best_outlier"].iloc[-1] <= best
