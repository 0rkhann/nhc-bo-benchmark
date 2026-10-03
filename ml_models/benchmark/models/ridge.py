from sklearn.linear_model import Ridge as _Ridge

from ml_models.benchmark.preprocess import InFold


class Ridge:
    name, tunable, grid = "ridge", True, None

    def defaults(self): return {"alpha": 1.0}

    def search_space(self, trial):
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}

    def fit(self, X, y, params, smiles=None):
        self.pre = InFold(scale=True).fit(X)
        self.m = _Ridge(alpha=params["alpha"]).fit(self.pre.transform(X), y)
        return self

    def predict(self, X, smiles=None):
        return self.m.predict(self.pre.transform(X))
