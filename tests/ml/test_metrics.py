import numpy as np
from ml_models.benchmark import metrics


def test_top1_recall_perfect_and_worst():
    y = np.arange(200, dtype=float)          # lowest 2 molecules are the top 1%
    assert metrics.top1_recall(y, y) == 1.0
    assert metrics.top1_recall(y, -y) == 0.0


def test_top1_recall_counts_at_least_one_molecule():
    y = np.arange(30, dtype=float)           # 1% of 30 rounds up to 1
    assert metrics.top1_recall(y, y) == 1.0


def test_constant_prediction_scores_near_chance():
    rng = np.random.default_rng(1)
    vals = [metrics.top1_recall(rng.normal(size=1400), np.zeros(1400), seed=s) for s in range(40)]
    assert 0.0 < np.mean(vals) < 0.15        # chance level is 5%


def test_tail_rmse_uses_lowest_decile():
    y = np.arange(100, dtype=float)
    p = y.copy(); p[:10] += 2.0              # error only in the lowest 10
    assert metrics.fold_metrics(y, p)["tail_rmse"] == 2.0


def test_ranking_only_keys():
    y = np.arange(100, dtype=float)
    assert set(metrics.fold_metrics(y, y, ranking_only=True)) == {"spearman", "top1_recall"}
