"""Exact GP with the BO loop's own kernel, priors and fitting code."""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from src.data_io import prune_columns
from src.gp_model import build_gp, fit_gp

WIDE = 600          # vanilla BO prunes Mordred; the 384-dim ChemBERTa-2 table is left unpruned there
PRUNE_AT = 0.95


class GP:
    name, tunable, grid = "gp", False, None
    refit_estimate_s = 2700

    def defaults(self): return {}
    def search_space(self, trial): return {}

    def fit(self, X, y, params, smiles=None):
        self.kept_columns = list(range(X.shape[1]))
        if X.shape[1] > WIDE:
            df = pd.DataFrame(X, columns=range(X.shape[1]))
            self.kept_columns = prune_columns(df, PRUNE_AT)
        X = X[:, self.kept_columns]
        # The BO loop feeds MinMax-scaled inputs in [0, 1], which the lengthscale prior assumes.
        self.pre = MinMaxScaler().fit(X)
        Xs = torch.tensor(self.pre.transform(X), dtype=torch.double)
        self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y))
        ys = torch.tensor((y - self.y_mu) / self.y_sd, dtype=torch.double).unsqueeze(-1)
        self.model = build_gp(Xs, ys, kernel_name="matern")
        fit_gp(self.model)
        return self

    def predict(self, X, smiles=None):
        Xs = torch.tensor(self.pre.transform(X[:, self.kept_columns]), dtype=torch.double)
        self.model.eval()
        with torch.no_grad():
            mean = torch.cat([self.model.posterior(c).mean.squeeze(-1) for c in Xs.split(512)])
        return mean.numpy() * self.y_sd + self.y_mu
