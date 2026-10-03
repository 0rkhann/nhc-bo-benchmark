import numpy as np
from ml_models.benchmark import models


def test_outlier_z_uses_training_statistics_only():
    X_tr = np.c_[np.zeros(10) + np.arange(10) * 0.1]
    X_te = np.array([[0.45], [100.0]])                 # the extreme test value must not shift z
    m = models.get("outlier").fit(X_tr, np.zeros(10), {})
    p = m.predict(X_te)
    assert p[1] < p[0]                                 # more extreme -> ranked as lower energy
    z_expected = abs(100.0 - X_tr.mean()) / X_tr.std()
    assert np.isclose(-p[1], z_expected)
