# tests/ml/test_adapters_foundation.py
import numpy as np
import pytest

from ml_models.benchmark import models, weights


def test_download_failure_is_reported(monkeypatch):            # Review Focus 4
    def offline(**kw):
        raise OSError("connection reset")
    monkeypatch.setattr(weights, "_download", offline)
    with pytest.raises(OSError, match="connection reset"):
        weights.fetch("Prior-Labs/tabpfn_3", "x.ckpt", "abc")


@pytest.mark.parametrize("name", ["tabpfn", "tabicl"])
def test_foundation_adapter_fit_predict(name):
    pytest.importorskip(name)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5)); y = X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=120)
    a = models.get(name)
    params = a.grid[0]
    pred = a.fit(X[:100], y[:100], params).predict(X[100:])
    assert pred.shape == (20,) and np.corrcoef(pred, y[100:])[0, 1] > 0.8
    assert "@" in a.checkpoint


def test_tabicl_processes_one_ensemble_member_at_a_time(monkeypatch):
    tabicl = pytest.importorskip("tabicl")
    seen = {}

    class Fake:
        def __init__(self, **kw): seen.update(kw)
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X))

    monkeypatch.setattr(tabicl, "TabICLRegressor", Fake)
    monkeypatch.setattr(weights, "fetch", lambda *a: "ckpt")
    models.get("tabicl").fit(np.ones((4, 2)), np.zeros(4), {"n_estimators": 8})
    assert seen["batch_size"] == 1
