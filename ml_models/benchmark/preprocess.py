"""Preprocessing fitted on the training part of a split only."""
import numpy as np


class InFold:
    def __init__(self, scale: bool):
        self.scale = scale

    def fit(self, X_tr: np.ndarray) -> "InFold":
        sd = X_tr.std(axis=0)
        self.keep = sd > 0
        self.mu = X_tr[:, self.keep].mean(axis=0)
        self.sd = sd[self.keep]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X[:, self.keep]
        return (X - self.mu) / self.sd if self.scale else X
