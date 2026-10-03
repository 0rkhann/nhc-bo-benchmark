import os

from xgboost import XGBRegressor

from ml_models.benchmark.preprocess import InFold


class XGB:
    name, tunable, grid = "xgb", True, None
    refit_estimate_s = 600   # largest trees on Mordred at 5,480 rows

    def defaults(self):
        return {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1,
                "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 1e-3}

    def search_space(self, trial):
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True)}

    def fit(self, X, y, params, smiles=None):
        self.pre = InFold(scale=False).fit(X)
        self.m = XGBRegressor(**params, tree_method="hist", n_jobs=os.cpu_count(), random_state=0)
        self.m.fit(self.pre.transform(X), y)
        return self

    def predict(self, X, smiles=None):
        return self.m.predict(self.pre.transform(X))
