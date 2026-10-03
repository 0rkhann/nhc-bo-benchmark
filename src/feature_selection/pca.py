import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from .base import FeatureSelector
import logging
from typing import Optional
import torch


class PCAFeatureSelector(FeatureSelector):
    def __init__(
        self,
        n_components: Optional[int] = None,
        max_components: Optional[int] = None,
        explained_variance_threshold: float = 0.95,
        cv_folds: int = 5,
        logger: Optional[logging.Logger] = None,
        **kwargs,  # Accept and ignore extra parameters like random_state
    ):
        """
        n_components: number of principal components to keep (if None, auto-select)
        max_components: maximum number of components to try (if None, use all features)
        explained_variance_threshold: minimum explained variance to maintain (default: 0.95)
        cv_folds: number of cross-validation folds for performance-based selection
        logger: optional logger for info/debug messages
        """
        self.n_components = n_components
        self.max_components = max_components
        self.explained_variance_threshold = explained_variance_threshold
        self.cv_folds = cv_folds
        self.logger = logger
        self.model: Optional[PCA] = None
        self.selected_components: Optional[int] = None

    def _auto_select_components(self, X: np.ndarray, y: np.ndarray) -> int:
        """
        Automatically select optimal number of components using multiple criteria:
        1. Explained variance threshold
        2. Cross-validation performance
        3. Elbow method on explained variance
        """
        n_samples, n_features = X.shape

        # Ensure we have enough samples for meaningful PCA
        if n_samples < 2:
            if self.logger:
                self.logger.warning(
                    f"Only {n_samples} samples available, using 1 component"
                )
            return 1

        # Calculate maximum components safely
        max_comp = min(
            self.max_components or n_features,
            max(1, n_samples - 1),  # Ensure at least 1 component
            n_features,
        )

        # For very small datasets, limit components
        if n_samples <= 5:
            max_comp = min(max_comp, n_samples - 1)
            if self.logger:
                self.logger.info(
                    f"Small dataset ({n_samples} samples), limiting to {max_comp} components"
                )

        if self.logger:
            self.logger.info(f"Auto-selecting PCA components from 1 to {max_comp}")

        # Method 1: Explained variance threshold
        pca_full = PCA(n_components=max_comp)
        pca_full.fit(X)
        explained_var_ratio = pca_full.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var_ratio)

        # Find components needed for threshold
        threshold_components = (
            np.argmax(cumulative_var >= self.explained_variance_threshold) + 1
        )

        # Ensure threshold_components doesn't exceed max_comp
        threshold_components = min(threshold_components, max_comp)

        if self.logger:
            self.logger.info(
                f"Explained variance threshold ({self.explained_variance_threshold}): {threshold_components} components"
            )

        # Method 2: Cross-validation performance
        cv_scores = []
        component_range = range(
            1, min(max_comp + 1, threshold_components + 10)
        )  # Limit range for efficiency

        # For very small datasets, limit CV range
        if n_samples <= 5:
            component_range = range(1, min(max_comp + 1, n_samples))

        for n_comp in component_range:
            try:
                # Transform data
                X_transformed = pca_full.transform(X)[:, :n_comp]

                # Use linear regression for CV (simple but effective)
                lr = LinearRegression()
                scores = cross_val_score(
                    lr, X_transformed, y, cv=min(self.cv_folds, n_samples), scoring="r2"
                )
                cv_scores.append(np.mean(scores))
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"CV failed for {n_comp} components: {e}")
                cv_scores.append(-np.inf)

        # Find best CV performance
        if cv_scores and any(score > -np.inf for score in cv_scores):
            best_cv_idx = np.argmax(cv_scores)
            best_cv_components = component_range[best_cv_idx]
            best_cv_score = cv_scores[best_cv_idx]
        else:
            # Fallback if CV fails
            best_cv_components = 1
            best_cv_score = -np.inf

        if self.logger:
            self.logger.info(
                f"Cross-validation best: {best_cv_components} components (R² = {best_cv_score:.3f})"
            )

        # Method 3: Elbow method on explained variance
        # Find where the rate of explained variance increase slows down
        if len(explained_var_ratio) > 2:
            var_diffs = np.diff(explained_var_ratio)
            var_diff_ratios = var_diffs[1:] / (
                var_diffs[:-1] + 1e-10
            )  # Avoid division by zero

            # Find elbow point (where ratio drops significantly)
            elbow_threshold = 0.5  # Adjustable threshold
            elbow_components = (
                np.argmax(var_diff_ratios < elbow_threshold) + 2
            )  # +2 because of diff and indexing

            # Ensure elbow_components doesn't exceed max_comp
            elbow_components = min(elbow_components, max_comp)
        else:
            elbow_components = 1

        if self.logger:
            self.logger.info(f"Elbow method: {elbow_components} components")

        # Combine all methods: take the minimum of threshold and CV, but not less than elbow
        final_components = max(
            min(threshold_components, best_cv_components), elbow_components
        )

        # Ensure reasonable bounds
        final_components = max(1, min(final_components, max_comp))

        if self.logger:
            self.logger.info(f"Final selection: {final_components} components")
            self.logger.info(
                f"Explained variance: {cumulative_var[final_components-1]:.3f}"
            )

        return final_components

    def fit(self, X: np.ndarray, y=None):
        # Auto-select components if not specified
        if self.n_components is None:
            self.selected_components = self._auto_select_components(X, y)
        else:
            self.selected_components = self.n_components

        if self.logger:
            self.logger.info(f"Fitting PCA with {self.selected_components} components")

        # Fit PCA with selected components
        self.model = PCA(n_components=self.selected_components).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA must be fitted before transform")
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def prepare(
        self,
        train_X: "np.ndarray",
        train_y: "np.ndarray",
        pool_X: "np.ndarray",
        test_X: "np.ndarray",
    ):
        """
        Fit PCA on training data and transform all datasets.
        This actually reduces the number of features.
        """
        if self.logger:
            self.logger.info(
                f"Applying PCA transformation: {train_X.shape[1]} -> auto-selected components"
            )

        # Fit PCA on training data
        self.fit(train_X, train_y)

        # Transform all datasets
        train_X_transformed = self.transform(train_X)
        pool_X_transformed = self.transform(pool_X)
        test_X_transformed = self.transform(test_X)

        if self.logger:
            self.logger.info(
                f"PCA transformation complete. Selected {self.selected_components} components. Shapes: train={train_X_transformed.shape}, pool={pool_X_transformed.shape}, test={test_X_transformed.shape}"
            )

        return (
            torch.tensor(train_X_transformed, dtype=torch.double),
            torch.tensor(pool_X_transformed, dtype=torch.double),
            torch.tensor(test_X_transformed, dtype=torch.double),
        )

    def get_support(self):
        # Return the actual number of components selected
        if self.selected_components is None:
            raise RuntimeError("PCA must be fitted before get_support")
        return self.selected_components
