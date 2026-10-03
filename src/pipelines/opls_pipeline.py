"""
OPLS BO pipeline with OPLS feature selection.
"""

from src.pipelines.base_pipeline import BOPipeline
from src.feature_selection.opls import OPLSFeatureSelector


class OPLSPipeline(BOPipeline):
    """BO pipeline with OPLS feature selection."""

    def selector_factory(self):
        """Create OPLS feature selector."""
        fs = self.cfg.get("feature_selection", {})
        n_components = fs.get("opls_components")
        n_orthogonal = fs.get("opls_orthogonal")

        # None = adaptive: up to 10 predictive and 1-3 orthogonal components.
        max_predictive = 10 if n_components is None else n_components
        if n_orthogonal is None:
            min_orthogonal, max_orthogonal = 1, 3  # at least one, or it is plain PLS
        else:
            min_orthogonal = max_orthogonal = n_orthogonal

        return OPLSFeatureSelector(
            max_predictive=max_predictive,
            min_predictive=min(2, max_predictive),
            max_orthogonal=max_orthogonal,
            min_orthogonal=min_orthogonal,
            cv=fs.get("opls_cv_folds", 5),
            random_state=self.cfg.get("random_state", 42),
            scale=fs.get("opls_scale", True),
            logger=self.logger,
        )
