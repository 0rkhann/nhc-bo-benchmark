import numpy as np
import torch
from typing import Union
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from .base import FeatureSelector

from pyopls.opls import OPLS


class OPLSFeatureSelector(FeatureSelector):
    def __init__(
        self,
        cv: int = 5,
        max_predictive: int = 15,
        max_orthogonal: int = 5,
        min_predictive: int = 2,
        min_orthogonal: int = 1,
        random_state=None,
        scale: bool = True,
        logger=None,
        **kwargs,
    ):
        """
        cv              : number of folds for cross-validation
        max_predictive  : maximum number of predictive (PLS) components to try
        max_orthogonal  : maximum number of orthogonal (OPLS filter) components to try
        min_predictive  : minimum number of predictive components (prevents too few)
        min_orthogonal  : minimum number of orthogonal components; >= 1 keeps this OPLS, not plain PLS
        random_state    : for reproducibility
        scale           : whether to standardize X before fitting
        logger          : optional logger
        **kwargs        : catch legacy args
        """
        super().__init__()
        self.cv = cv
        # Ensure max_predictive, max_orthogonal, and min_predictive are integers
        self.max_predictive = max_predictive if max_predictive is not None else 15
        self.max_orthogonal = max_orthogonal if max_orthogonal is not None else 5
        self.min_predictive = min_predictive if min_predictive is not None else 2
        self.min_orthogonal = min_orthogonal if min_orthogonal is not None else 0
        self.random_state = random_state
        self.scale = scale
        self.logger = logger

        # to be set in fit()
        self.scaler = None
        self.ortho_filter = None
        self.pls = None
        self.n_predictive = None
        self.n_orthogonal = None

    def fit(self, X: np.ndarray, y) -> "OPLSFeatureSelector":
        """
        Fit OPLS feature selector.
        Grid-search CV over (predictive, orthogonal) component pairs and fit final models.
        Scaling is done here only; the OPLS filter itself is fitted without scaling in
        the same (standardised) space it is applied to in transform().
        """
        y_arr = np.asarray(y, dtype=float)

        # optional scaling
        if self.scale:
            self.scaler = StandardScaler()
            X_proc = self.scaler.fit_transform(X)
        else:
            X_proc = X.copy()

        n_samples, n_feats = X_proc.shape
        max_pred = min(self.max_predictive, n_feats - 1)
        min_orth = min(self.min_orthogonal, n_feats - 1)
        max_orth = max(min(self.max_orthogonal, n_feats - 1), min_orth)

        best_score = -np.inf
        best_pred, best_orth = 1, min_orth
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        # nested CV search
        for n_pred in range(1, max_pred + 1):
            for n_orth in range(min_orth, max_orth + 1):
                scores = []
                for train_idx, val_idx in kf.split(X_proc):
                    Xtr = X_proc[train_idx]
                    Xv = X_proc[val_idx]
                    ytr = y_arr[train_idx]
                    yv = y_arr[val_idx]

                    # orthogonal filtering if requested
                    if n_orth > 0:
                        n_orth_safe = min(n_orth, len(Xtr) - 1, Xtr.shape[1] - 1)
                        if n_orth_safe < 1:
                            scores.append(-np.inf)
                            continue
                        opls = OPLS(n_components=n_orth_safe, scale=False)
                        Xtr_filt = opls.fit_transform(Xtr, ytr)
                        Xv_filt = opls.transform(Xv)
                    else:
                        Xtr_filt, Xv_filt = Xtr, Xv

                    # predictive PLS - ensure n_pred doesn't exceed training samples
                    n_pred_safe = min(n_pred, len(Xtr) - 1, Xtr_filt.shape[1] - 1)
                    if n_pred_safe < 1:
                        scores.append(-np.inf)
                        continue

                    try:
                        pls = PLSRegression(n_components=n_pred_safe)
                        pls.fit(Xtr_filt, ytr)
                        yv_pred = pls.predict(Xv_filt).ravel()

                        # compute R^2
                        ss_res = np.sum((yv - yv_pred) ** 2)
                        ss_tot = np.sum((yv - yv.mean()) ** 2)
                        if ss_tot > 0:
                            scores.append(1 - ss_res / ss_tot)
                        else:
                            scores.append(0.0)
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"PLS failed in OPLS CV fold: {e}")
                        scores.append(-np.inf)

                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_pred, best_orth = n_pred, n_orth

        # final fit on full data (min_predictive / min_orthogonal enforced)
        self.n_orthogonal = max(min_orth, best_orth)
        self.n_predictive = max(self.min_predictive, best_pred)
        self.n_predictive = min(self.n_predictive, n_samples - 1, n_feats)

        if self.n_orthogonal > 0:
            self.ortho_filter = OPLS(n_components=self.n_orthogonal, scale=False)
            X_filt_all = self.ortho_filter.fit_transform(X_proc, y_arr)
        else:
            self.ortho_filter = None
            X_filt_all = X_proc

        self.pls = PLSRegression(n_components=self.n_predictive)
        self.pls.fit(X_filt_all, y_arr)

        if self.logger:
            self.logger.info(
                f"OPLSFeatureSelector fit → predictive={self.n_predictive}, orthogonal={self.n_orthogonal}, CV_score={best_score:.4f}"
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data to predictive score space."""
        if self.scale and self.scaler is not None:
            X_proc = self.scaler.transform(X)
        else:
            X_proc = X

        if (
            self.n_orthogonal is not None
            and self.n_orthogonal > 0
            and self.ortho_filter is not None
        ):
            X_filt = self.ortho_filter.transform(X_proc)
        else:
            X_filt = X_proc

        T = self.pls.transform(X_filt)
        return T

    def get_support(self) -> int:
        """Return total number of components used (predictive + orthogonal)."""
        if self.n_predictive is None or self.n_orthogonal is None:
            raise RuntimeError(
                "OPLSFeatureSelector: fit must be called before get_support."
            )
        return self.n_predictive + self.n_orthogonal

    def fit_transform(self, X: np.ndarray, y) -> np.ndarray:
        """Fit on data then transform."""
        return self.fit(X, y).transform(X)

    def prepare(
        self,
        train_X: np.ndarray,
        train_y: Union[list, np.ndarray],
        pool_X: np.ndarray,
        test_X: np.ndarray,
    ):
        """Fit on train set and transform train/pool/test splits."""
        self.fit(train_X, train_y)
        return (
            torch.tensor(self.transform(train_X), dtype=torch.double),
            torch.tensor(self.transform(pool_X), dtype=torch.double),
            torch.tensor(self.transform(test_X), dtype=torch.double),
        )
