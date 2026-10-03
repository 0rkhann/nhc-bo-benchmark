"""
Outlier heuristic baseline: no model, no feature selection.

Pool molecules are evaluated in order of decreasing max |z-score| across features
(z computed on all X). Tests whether a method wins only by visiting extreme molecules.
"""

import numpy as np

from baselines.random_search import RandomSearchPipeline


class OutlierSearchPipeline(RandomSearchPipeline):
    name = "outlier_search"
    best_col = "best_outlier"

    def _pool_scores(self, X_full_df, pool_raw_df):
        X = X_full_df.to_numpy(dtype=float)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)  # constant columns carry no outlier signal
        return np.abs((pool_raw_df.to_numpy(dtype=float) - mu) / sd).max(axis=1)

    def _pick(self, scores) -> int:
        return int(np.argmax(scores))
