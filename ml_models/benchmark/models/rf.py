import os

from sklearn.ensemble import RandomForestRegressor

from ml_models.benchmark.preprocess import InFold


class RF:
    name, tunable, grid = "rf", True, None
    refit_estimate_s = 1200   # largest trees on Mordred at 5,480 rows

    def defaults(self):
        return {"n_estimators": 300, "max_features": 0.5, "min_samples_leaf": 1, "max_depth": 0}

    def search_space(self, trial):
        return {"n_estimators": trial.suggest_int("n_estimators", 300, 1000),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_depth": trial.suggest_categorical("max_depth", [0, 10, 20, 30, 40])}  # 0 = None

    def fit(self, X, y, params, smiles=None):
        p = dict(params); p["max_depth"] = p["max_depth"] or None
        self.pre = InFold(scale=False).fit(X)
        self.m = RandomForestRegressor(**p, n_jobs=os.cpu_count(), random_state=0)
        self.m.fit(self.pre.transform(X), y)
        return self

    def predict(self, X, smiles=None):
        return self.m.predict(self.pre.transform(X))
