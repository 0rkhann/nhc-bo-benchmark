from typing import Optional
from .base import FeatureSelector
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import torch


class SpearmanFABOSelector(FeatureSelector):
    def __init__(
        self,
        *,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        max_features: Optional[int] = None,
        min_features: int = 5,  # NEW: Minimum features to prevent too aggressive selection
        min_correlation: float = 0.1,
        cv_folds: int = 5,
        logger=None,
    ):
        """
        k: number of features to select (if None, auto-select)
        threshold: minimum correlation threshold (if None, auto-select)
        max_features: maximum number of features to consider
        min_features: minimum number of features to select (prevents too aggressive reduction)
        min_correlation: minimum correlation to consider a feature
        cv_folds: number of cross-validation folds for performance-based selection
        logger: optional logger for info/debug messages
        """
        if k is None and threshold is None:
            # Auto-select mode
            self.k = None
            self.threshold = None
        else:
            self.k = k
            self.threshold = threshold

        self.max_features = max_features
        self.min_features = min_features  # NEW
        self.min_correlation = min_correlation
        self.cv_folds = cv_folds
        self.logger = logger
        self.selected_indices_: Optional[np.ndarray] = None
        self.selected_k: Optional[int] = None

    def _auto_select_features(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[int, np.ndarray]:
        """
        Automatically select optimal number of features using multiple criteria:
        1. Correlation strength analysis
        2. Cross-validation performance
        3. Information content preservation
        """
        n_samples, n_features = X.shape

        # Ensure we have enough samples for meaningful feature selection
        if n_samples < 2:
            if self.logger:
                self.logger.warning(
                    f"Only {n_samples} samples available, using 1 feature"
                )
            return 1, np.array([0])  # Return first feature

        # Calculate maximum features safely
        max_feat = min(
            self.max_features or n_features,
            max(1, n_samples - 1),  # Ensure at least 1 feature
            n_features,
        )

        # For very small datasets, limit features
        if n_samples <= 5:
            max_feat = min(max_feat, n_samples - 1)
            if self.logger:
                self.logger.info(
                    f"Small dataset ({n_samples} samples), limiting to {max_feat} features"
                )

        if self.logger:
            self.logger.info(
                f"Auto-selecting FABO features from {n_features} total features"
            )

        # Method 1: Calculate correlations for all features
        corrs = np.array([abs(spearmanr(X[:, j], y)[0]) for j in range(n_features)])

        # Filter by minimum correlation
        valid_features = corrs >= self.min_correlation
        if not np.any(valid_features):
            if self.logger:
                self.logger.warning(
                    f"No features meet minimum correlation {self.min_correlation}, lowering to 0.05"
                )
            self.min_correlation = 0.05
            valid_features = corrs >= self.min_correlation

        # Sort features by correlation strength
        sorted_indices = np.argsort(corrs)[::-1]
        valid_sorted_indices = sorted_indices[valid_features[sorted_indices]]

        if self.logger:
            self.logger.info(
                f"Found {len(valid_sorted_indices)} features with correlation >= {self.min_correlation}"
            )

        # Method 2: Cross-validation performance for different k values
        max_k_to_try = min(
            max_feat, len(valid_sorted_indices), 50
        )  # Limit for efficiency

        # For very small datasets, limit k range
        if n_samples <= 5:
            max_k_to_try = min(max_k_to_try, n_samples - 1)

        k_range = range(1, max_k_to_try + 1)

        cv_scores = []
        for k in k_range:
            try:
                # Select top k features
                selected_idx = valid_sorted_indices[:k]
                X_selected = X[:, selected_idx]

                # Use linear regression for CV
                lr = LinearRegression()
                scores = cross_val_score(
                    lr, X_selected, y, cv=min(self.cv_folds, n_samples), scoring="r2"
                )
                cv_scores.append(np.mean(scores))
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"CV failed for k={k}: {e}")
                cv_scores.append(-np.inf)

        # Find best CV performance
        if cv_scores and any(score > -np.inf for score in cv_scores):
            best_cv_idx = np.argmax(cv_scores)
            best_k = k_range[best_cv_idx]
            best_cv_score = cv_scores[best_cv_idx]
        else:
            # Fallback if CV fails
            best_k = 1
            best_cv_score = -np.inf

        if self.logger:
            self.logger.info(
                f"Cross-validation best: k={best_k} features (R² = {best_cv_score:.3f})"
            )

        # Method 3: Information content preservation
        # Find where adding more features gives diminishing returns
        if len(cv_scores) > 1:
            score_improvements = np.diff(cv_scores)
            # Find where improvement drops below threshold
            improvement_threshold = 0.01  # 1% improvement threshold
            diminishing_returns_k = (
                np.argmax(score_improvements < improvement_threshold) + 1
            )

            if self.logger:
                self.logger.info(
                    f"Diminishing returns analysis: k={diminishing_returns_k} features"
                )

            # Use the minimum of CV best and diminishing returns
            final_k = min(best_k, diminishing_returns_k)
        else:
            final_k = best_k

        # Ensure reasonable bounds
        final_k = max(1, min(final_k, max_feat))
        
        # NEW: Enforce minimum features to prevent too aggressive selection
        if final_k < self.min_features:
            original_k = final_k
            final_k = min(self.min_features, max_feat)  # Can't exceed available features
            if self.logger:
                self.logger.info(
                    f"⚠️  Enforcing minimum {self.min_features} features (was {original_k})"
                )

        # Select final features
        final_indices = valid_sorted_indices[:final_k]

        if self.logger:
            self.logger.info(f"Final selection: k={final_k} features")
            self.logger.info(
                f"Correlation range: {corrs[final_indices].min():.3f} to {corrs[final_indices].max():.3f}"
            )

        return final_k, final_indices

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpearmanFABOSelector":
        if self.k is None and self.threshold is None:
            # Auto-select mode
            self.selected_k, self.selected_indices_ = self._auto_select_features(X, y)
        else:
            # Manual mode (original logic)
            corrs = np.array([abs(spearmanr(X[:, j], y)[0]) for j in range(X.shape[1])])

            if self.threshold is not None:
                # Start from your threshold, drop by .05 until you find something
                t = self.threshold
                idx = np.where(corrs >= t)[0]
                while len(idx) == 0 and t > 0:
                    old_t = t
                    t = max(0.0, t - 0.05)
                    if self.logger:
                        self.logger.warning(
                            f"No features ≥ {old_t:.3f}; lowering threshold to {t:.3f}"
                        )
                    idx = np.where(corrs >= t)[0]
                if len(idx) == 0:
                    # still nothing → fall back below
                    if self.k is not None:
                        idx = np.argsort(corrs)[::-1][: self.k]
                    else:
                        idx = np.arange(X.shape[1])
            else:
                # pure‐k mode never changes
                idx = np.argsort(corrs)[::-1][: self.k]

            self.selected_indices_ = np.sort(idx)
            self.selected_k = len(self.selected_indices_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("FABO must be fitted before transform")
        return X[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def prepare(
        self,
        train_X: "np.ndarray",
        train_y: "np.ndarray",
        pool_X: "np.ndarray",
        test_X: "np.ndarray",
    ):
        """
        Fit FABO on training data and transform all datasets.
        This actually reduces the number of features.
        """
        if self.logger:
            if self.k is None and self.threshold is None:
                self.logger.info(
                    f"Applying FABO feature selection: {train_X.shape[1]} -> auto-selected features"
                )
            else:
                self.logger.info(
                    f"Applying FABO feature selection: {train_X.shape[1]} -> {self.k if self.k else 'threshold-based'} features"
                )

        # Fit FABO on training data
        self.fit(train_X, train_y)

        # Transform all datasets
        train_X_transformed = self.transform(train_X)
        pool_X_transformed = self.transform(pool_X)
        test_X_transformed = self.transform(test_X)

        if self.logger:
            self.logger.info(
                f"FABO selection complete. Selected {self.selected_k} features. Shapes: train={train_X_transformed.shape}, pool={pool_X_transformed.shape}, test={test_X_transformed.shape}"
            )

        return (
            torch.tensor(train_X_transformed, dtype=torch.double),
            torch.tensor(pool_X_transformed, dtype=torch.double),
            torch.tensor(test_X_transformed, dtype=torch.double),
        )

    def get_support(self) -> int:
        if self.selected_indices_ is None:
            raise ValueError("Must call fit() before get_support()")
        return self.selected_k
