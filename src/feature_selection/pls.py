import numpy as np
import warnings
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
import torch
from .base import FeatureSelector


class PLSFeatureSelector(FeatureSelector):
    def __init__(self, cv=5, max_components=20, min_components=3, random_state=None, logger=None):
        """
        cv             : number of folds for cross-validation
        max_components : maximum number of PLS components to try
        min_components : minimum number of PLS components (prevents too few)
        random_state   : for reproducibility
        logger         : optional logger for progress/info
        """
        super().__init__()
        self.cv = cv
        # Ensure max_components and min_components are integers
        self.max_components = max_components if max_components is not None else 20
        self.min_components = min_components if min_components is not None else 3
        self.random_state = random_state
        self.logger = logger
        self.n_components: int | None = None
        self.best_score: float | None = None
        self.model: PLSRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSFeatureSelector":
        n_samples, n_feats = X.shape
        max_comp = max(1, min(self.max_components, n_samples - 1, n_feats))

        if self.logger:
            self.logger.info(
                f"Fitting PLS up to {max_comp} comps (samples={n_samples}, feats={n_feats})"
            )

        best_score = -np.inf
        best_n = 1
        best_model = None

        for n in range(1, max_comp + 1):
            try:
                # Ensure n doesn't exceed what CV folds can handle
                cv_folds = min(self.cv, n_samples)
                train_size = (
                    n_samples * (cv_folds - 1) // cv_folds
                )  # Approximate training size per fold
                n_safe = min(n, train_size - 1, n_feats - 1)

                if n_safe < 1:
                    continue

                pls = PLSRegression(n_components=n_safe)

                # Suppress sklearn warnings for numerical edge cases
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=".*R\\^2 score is not well-defined.*"
                    )
                    warnings.filterwarnings(
                        "ignore", message=".*y residual is constant.*"
                    )
                    warnings.filterwarnings("ignore", message=".*fits failed.*")
                    warnings.filterwarnings("ignore", message=".*n_components.*")
                    score = np.mean(
                        cross_val_score(pls, X, y, cv=cv_folds, scoring="r2")
                    )
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"PLS n={n} failed: {e}")
                continue
            if score > best_score:
                best_score = score
                best_model = pls
                best_n = n_safe

        if best_model is None:
            raise ValueError("No valid PLS model could be fitted.")

        # Enforce minimum components: refit if CV picked fewer, so that the model
        # we keep and the number we report are the same thing.
        best_n = best_model.n_components
        min_n = min(self.min_components, n_samples - 1, n_feats)
        if best_n < min_n:
            best_n = min_n
            best_model = PLSRegression(n_components=best_n)

        self.model = best_model.fit(X, y)
        self.n_components = best_n
        self.best_score = best_score
        if self.logger:
            self.logger.info(
                f"Selected PLS n_components={best_n} with CV R²={best_score:.3f}"
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "PLSFeatureSelector: fit must be called before transform."
            )
        return self.model.transform(X)

    def prepare(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        pool_X: np.ndarray,
        test_X: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.fit(train_X, train_y)
        Xtr = self.transform(train_X)
        Xp = self.transform(pool_X)
        Xt = self.transform(test_X)
        return (
            torch.tensor(Xtr, dtype=torch.double),
            torch.tensor(Xp, dtype=torch.double),
            torch.tensor(Xt, dtype=torch.double),
        )

    def get_support(self) -> int:
        if self.n_components is None:
            raise RuntimeError(
                "PLSFeatureSelector: fit must be called before get_support."
            )
        return self.n_components
