# tests/ml/test_adapter_gp.py
import numpy as np
import pytest

pytest.importorskip("botorch")
from ml_models.benchmark import models


def test_gp_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(80, 3)); y = np.sin(4 * X[:, 0]) + 0.05 * rng.normal(size=80)
    pred = models.get("gp").fit(X[:60], y[:60], {}).predict(X[60:])
    assert pred.shape == (20,) and np.corrcoef(pred, y[60:])[0, 1] > 0.8


def test_gp_prunes_wide_inputs_on_training_rows_only():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 5))
    X = np.c_[base, base[:, [0]] * 2.0, rng.normal(size=(50, 700))]   # column 5 duplicates column 0
    gp = models.get("gp").fit(X[:40], base[:40, 0], {})
    assert 5 not in gp.kept_columns and len(gp.kept_columns) < X.shape[1]
