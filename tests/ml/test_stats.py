import numpy as np
from scipy import stats as st
from ml_models.benchmark import stats


def test_corrected_ttest_matches_hand_computation():
    a = np.array([0.30, 0.28, 0.31, 0.27, 0.29]); b = np.array([0.25, 0.26, 0.24, 0.27, 0.23])
    d = a - b
    t = d.mean() / np.sqrt((1 / 5 + 0.25) * d.var(ddof=1))
    expected_p = 2 * st.t.sf(abs(t), df=4)
    diff, p = stats.corrected_ttest(a, b)
    assert np.isclose(diff, d.mean()) and np.isclose(p, expected_p)


def test_holm():
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_ceiling_rule():
    assert stats.ceiling_moved(0.06, 0.01)
    assert not stats.ceiling_moved(0.04, 0.001)
    assert not stats.ceiling_moved(0.10, 0.2)
