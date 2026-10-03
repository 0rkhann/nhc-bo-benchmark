import numpy as np


class Mean:
    name, tunable, grid = "mean", False, None

    def defaults(self): return {}
    def search_space(self, trial): return {}

    def fit(self, X, y, params, smiles=None):
        self.mu = float(np.mean(y)); return self

    def predict(self, X, smiles=None):
        return np.full(len(X), self.mu)
