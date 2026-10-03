"""Nadeau-Bengio corrected resampled t-test, Holm correction and the headline rule."""
import numpy as np
from scipy import stats as st

CEILING_DELTA_R2 = 0.05
ALPHA = 0.05


def corrected_ttest(a, b, test_train_ratio: float = 0.25) -> tuple[float, float]:
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float(d.mean()), (1.0 if d.mean() == 0 else 0.0)
    t = d.mean() / np.sqrt((1 / k + test_train_ratio) * var)
    return float(d.mean()), float(2 * st.t.sf(abs(t), df=k - 1))


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def ceiling_moved(delta_r2: float, p_holm: float) -> bool:
    return delta_r2 >= CEILING_DELTA_R2 and p_holm < ALPHA
