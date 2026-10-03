"""Per-fold metrics. Lower energy is better, so the 'top' molecules have the lowest values."""
import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def top1_recall(y_true, y_pred, seed: int = 0) -> float:
    """Share of the lowest-1% true energies that are among the lowest 5% predictions.
    Ties in the predictions are broken at random with a fixed seed."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(y_true)
    top = set(np.argsort(y_true, kind="stable")[: max(1, math.ceil(0.01 * n))])
    jitter = np.random.default_rng(seed).random(n)
    order = np.lexsort((jitter, y_pred))           # sort by prediction, then by jitter
    picked = set(order[: max(1, math.ceil(0.05 * n))])
    return len(top & picked) / len(top)


def fold_metrics(y_true, y_pred, ranking_only: bool = False) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {"spearman": float(spearmanr(y_true, y_pred)[0]), "top1_recall": top1_recall(y_true, y_pred)}
    if ranking_only:
        return out
    tail = np.argsort(y_true, kind="stable")[: max(1, math.ceil(0.10 * len(y_true)))]
    out.update(r2=float(r2_score(y_true, y_pred)),
               rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
               mae=float(mean_absolute_error(y_true, y_pred)),
               tail_rmse=float(np.sqrt(np.mean((y_true[tail] - y_pred[tail]) ** 2))))
    return out
