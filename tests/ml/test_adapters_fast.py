import numpy as np
import optuna
import pytest

from ml_models.benchmark import models
from ml_models.benchmark.preprocess import InFold


def toy(n=120, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)); X[:, 5] = 3.0          # a constant column
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.1, n)
    return X, y


def test_infold_uses_training_statistics_only():
    X_tr = np.c_[np.ones(10), np.arange(10.0)]          # column 0 constant in train only
    X_te = np.c_[np.arange(5.0) + 100, np.arange(5.0)]
    pre = InFold(scale=True).fit(X_tr)
    out = pre.transform(X_te)
    assert out.shape == (5, 1)                          # column 0 dropped by the training rule
    assert np.allclose(out[:, 0], (np.arange(5.0) - 4.5) / np.arange(10.0).std())


@pytest.mark.parametrize("name", ["mean", "ridge", "rf", "xgb"])
def test_fast_adapters_fit_and_predict(name):
    X, y = toy()
    a = models.get(name)
    params = a.search_space(optuna.trial.FixedTrial(a.defaults())) if a.tunable else {}
    pred = a.fit(X[:100], y[:100], params).predict(X[100:])
    assert pred.shape == (20,) and np.isfinite(pred).all()
    if name != "mean":
        assert np.corrcoef(pred, y[100:])[0, 1] > 0.8


def test_mean_predictor_predicts_training_mean():
    X, y = toy()
    pred = models.get("mean").fit(X[:100], y[:100], {}).predict(X[100:])
    assert np.allclose(pred, y[:100].mean())
