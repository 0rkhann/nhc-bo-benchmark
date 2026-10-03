import numpy as np
import pandas as pd
import torch

import src.pipelines.base_pipeline as bp
from src.feature_selection.base import FeatureSelector
from src.pipelines.base_pipeline import BOPipeline
from tests.conftest import bo_config, split_like_pipeline

N_ITER = 3


class SpySelector(FeatureSelector):
    """Identity selector that records every array it is given."""

    def __init__(self):
        self.calls = []

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X

    def _record(self, kind, tr, y, pool, test):
        self.calls.append((kind, tr.copy(), np.asarray(y), pool.copy(), test.copy()))
        return tuple(torch.tensor(a, dtype=torch.double) for a in (tr, pool, test))

    def prepare(self, tr, y, pool, test):
        return self._record("prepare", tr, y, pool, test)

    def update(self, tr, y, pool, test):
        return self._record("update", tr, y, pool, test)


class SpyPipeline(BOPipeline):
    spy = None

    def selector_factory(self):
        SpyPipeline.spy = SpySelector()
        return SpyPipeline.spy


def _run(synthetic, pipeline_cls=SpyPipeline, **fs):
    cfg = bo_config(synthetic["results"], n_iter=N_ITER, **fs)
    pipeline_cls(cfg).run(str(synthetic["csv"]), synthetic["cache"], simulator=None)
    out = next(synthetic["results"].rglob("bo_iteration_history.csv"))
    return pd.read_csv(out)


def test_selector_prepare_and_update_see_the_same_input_space(synthetic):
    """Bug 1: prepare() got MinMax-scaled arrays, update() got raw ones (TotalE ~ -2500)."""
    _run(synthetic)
    calls = SpyPipeline.spy.calls
    assert [c[0] for c in calls] == ["prepare"] + ["update"] * N_ITER
    _, tr0, _, pool0, test0 = calls[0]
    for kind, tr, _, pool, test in calls[1:]:
        # the initial design rows are numerically identical in every call
        np.testing.assert_allclose(tr[: len(tr0)], tr0)
        assert np.abs(tr).max() < 1e3 and np.abs(pool).max() < 1e3, "raw (unscaled) values"
        # a pool row that is still in the pool keeps its representation
        assert any(np.allclose(row, pool0[0]) for row in np.vstack([pool, tr[len(tr0):]]))
        np.testing.assert_allclose(test, test0)


def test_scaler_is_fitted_on_all_inputs_so_nothing_leaves_unit_range(synthetic):
    """Bug 5: scaler was fitted on the 10 initial points; pool values fell outside [0, 1]."""
    _run(synthetic)
    _, tr, _, pool, test = SpyPipeline.spy.calls[0]
    allx = np.vstack([tr, pool, test])
    assert allx.min() >= -1e-9 and allx.max() <= 1 + 1e-9
    # fitted on everything, so some column reaches both 0 and 1 over train+pool+test
    assert np.isclose(allx.min(axis=0), 0).all() and np.isclose(allx.max(axis=0), 1).all()


def test_selector_gets_growing_training_set_and_matching_targets(synthetic):
    _run(synthetic)
    calls = SpyPipeline.spy.calls
    sizes = [len(c[1]) for c in calls]
    assert sizes == [10 + i for i in range(N_ITER + 1)]
    for _, tr, y, _, _ in calls:
        assert len(y) == len(tr)


def test_first_bo_iteration_uses_initial_design_as_incumbent(synthetic):
    """Bug 4: best_bo started at inf, so iteration 1 ran EI with best_f=inf."""
    seen = []
    real = bp.make_acquisition

    def spy(gp, best_f, **kw):
        seen.append(best_f)
        return real(gp, best_f=best_f, **kw)

    bp.make_acquisition = spy
    try:
        hist = _run(synthetic)
    finally:
        bp.make_acquisition = real
    assert np.isfinite(seen[0])
    init, _, _ = split_like_pipeline(synthetic["df"])
    init_min = min(synthetic["energy"][s] for s in init)
    assert hist["best_bo"].iloc[0] <= init_min
    assert hist["best_bo"].is_monotonic_decreasing


def test_best_pool_min_is_the_initial_pool_minimum(synthetic):
    hist = _run(synthetic)
    _, pool, _ = split_like_pipeline(synthetic["df"])
    want = min(synthetic["energy"][s] for s in pool)
    np.testing.assert_allclose(hist["best_pool_min"], want)
