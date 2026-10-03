"""The BO 'outlier' control as a ranker: no training, max |z| with training-part statistics."""
import numpy as np


class OutlierRank:
    name, tunable, grid = "outlier", False, None

    def defaults(self): return {}
    def search_space(self, trial): return {}

    def fit(self, X, y, params, smiles=None):
        sd = X.std(axis=0)
        self.keep = sd > 0
        self.mu, self.sd = X[:, self.keep].mean(axis=0), sd[self.keep]
        return self

    def predict(self, X, smiles=None):
        z = np.abs((X[:, self.keep] - self.mu) / self.sd).max(axis=1)
        return -z                                       # most extreme = predicted lowest energy
