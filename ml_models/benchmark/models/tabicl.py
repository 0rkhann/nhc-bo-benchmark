"""TabICLv2 (BSD-3-Clause)."""
import os

import torch

from ml_models.benchmark import weights
from ml_models.benchmark.preprocess import InFold

REPO, FILE, REV = "jingang/TabICL", "tabicl-regressor-v2-20260212.ckpt", "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"


class TabICL:
    name, tunable = "tabicl", True
    checkpoint = f"{REPO}/{FILE}@{REV}"
    refit_estimate_s = 2100

    def __init__(self):
        self.grid = [{"n_estimators": n} for n in (8, 16, 32)]

    def defaults(self): return dict(self.grid[0])

    def search_space(self, trial):
        return {"n_estimators": trial.suggest_categorical("n_estimators", [g["n_estimators"] for g in self.grid])}

    def fit(self, X, y, params, smiles=None):
        from tabicl import TabICLRegressor
        torch.set_num_threads(os.cpu_count())
        self.pre = InFold(scale=False).fit(X)
        # batch_size=1 runs one ensemble member at a time; the default (8) builds a
        # (8, rows, columns, embed_dim) tensor that exceeds a 16 GB runner on 384+ columns.
        self.m = TabICLRegressor(model_path=str(weights.fetch(REPO, FILE, REV)), allow_auto_download=False,
                                 device="cpu", batch_size=1, **params)
        self.m.fit(self.pre.transform(X), y)
        return self

    def predict(self, X, smiles=None):
        return self.m.predict(self.pre.transform(X))
