"""TabPFN-3 (weights under a non-commercial licence; research use only)."""
import os

import torch

from ml_models.benchmark import weights
from ml_models.benchmark.preprocess import InFold

REPO, FILE, REV = "Prior-Labs/tabpfn_3", "tabpfn-v3-regressor-v3_default.ckpt", "24a16a89d245878b846555110985634aa2e656d7"
GRID = [{"n_estimators": n, "softmax_temperature": t} for n in (8, 16) for t in (0.75, 0.9)]


class TabPFN:
    name, tunable = "tabpfn", True
    checkpoint = f"{REPO}/{FILE}@{REV}"
    refit_estimate_s = 1500

    def __init__(self):
        self.grid = list(GRID)

    def defaults(self): return dict(self.grid[0])

    def search_space(self, trial):
        return {"n_estimators": trial.suggest_categorical("n_estimators", sorted({g["n_estimators"] for g in self.grid})),
                "softmax_temperature": trial.suggest_categorical(
                    "softmax_temperature", sorted({g["softmax_temperature"] for g in self.grid}))}

    def fit(self, X, y, params, smiles=None):
        from tabpfn import TabPFNRegressor
        torch.set_num_threads(os.cpu_count())
        self.pre = InFold(scale=False).fit(X)
        self.m = TabPFNRegressor(model_path=str(weights.fetch(REPO, FILE, REV)), device="cpu",
                                 ignore_pretraining_limits=True, random_state=0, **params)
        self.m.fit(self.pre.transform(X), y)
        return self

    def predict(self, X, smiles=None):
        return self.m.predict(self.pre.transform(X))
