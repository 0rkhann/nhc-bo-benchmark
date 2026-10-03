import numpy as np
import pytest

from src.feature_selection.fabo import SpearmanFABOSelector
from src.feature_selection.opls import OPLSFeatureSelector
from src.feature_selection.pca import PCAFeatureSelector
from src.feature_selection.pls import PLSFeatureSelector

FACTORIES = {
    "fabo": lambda: SpearmanFABOSelector(),
    "pca": lambda: PCAFeatureSelector(),
    "pls": lambda: PLSFeatureSelector(),
    "opls": lambda: OPLSFeatureSelector(random_state=0, max_predictive=4, max_orthogonal=2),
}


def _data(n, seed=0, informative=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 12))
    y = 5 * X[:, informative] + 0.05 * rng.normal(size=n)
    return X, y


@pytest.mark.parametrize("name", FACTORIES)
def test_update_refits_on_current_training_set(name):
    """update() must equal a fresh fit on the new train set (bug 2: it reused the
    model fitted on the 10 initial points)."""
    X0, y0 = _data(10, seed=1, informative=0)
    # later training set: different design, and the target now depends on another column
    X1, y1 = _data(25, seed=2, informative=7)
    pool, test = _data(30, seed=3)[0], _data(8, seed=4)[0]

    sel = FACTORIES[name]()
    sel.prepare(X0, y0, pool, test)
    updated = sel.update(X1, y1, pool, test)

    fresh = FACTORIES[name]().prepare(X1, y1, pool, test)
    for got, want in zip(updated, fresh):
        assert got.shape == want.shape
        np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-8)


def test_pls_reported_components_match_fitted_model():
    """Bug (minor): n_components was raised to min_components but the model kept fewer."""
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(30, 12))
    y = 4 * X[:, 0] + 0.01 * rng.normal(size=30)
    sel = PLSFeatureSelector(max_components=2, min_components=3).fit(X, y)
    assert sel.n_components == 3
    assert sel.transform(X).shape[1] == sel.n_components


def test_opls_filter_is_fitted_in_the_space_it_is_applied_in():
    """Bug 6: ortho filter was fitted on raw X but applied to standardised X."""
    rng = np.random.default_rng(0)
    X = 1000 + 50 * rng.normal(size=(30, 8))  # raw columns far from zero mean / unit variance
    y = 0.02 * X[:, 0] + rng.normal(0, 0.1, 30)
    sel = OPLSFeatureSelector(random_state=0, max_predictive=3, max_orthogonal=2, min_orthogonal=1)
    sel.fit(X, y)
    assert sel.ortho_filter is not None
    np.testing.assert_allclose(sel.ortho_filter.x_mean_, 0, atol=1e-6)


def test_opls_uses_the_enforced_number_of_predictive_components():
    """Bug 6: final PLS used best_pred instead of the enforced n_predictive."""
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(30, 8))
    y = X[:, 0] + 0.01 * rng.normal(size=30)
    sel = OPLSFeatureSelector(
        random_state=0, max_predictive=2, min_predictive=3, max_orthogonal=1
    ).fit(X, y)
    assert sel.n_predictive == 3
    assert sel.pls.n_components == sel.n_predictive


def test_opls_enforces_minimum_orthogonal_components():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(30, 8))
    y = X[:, 0] + 0.01 * rng.normal(size=30)
    sel = OPLSFeatureSelector(
        random_state=0, max_predictive=3, max_orthogonal=1, min_orthogonal=2
    ).fit(X, y)
    assert sel.n_orthogonal == 2
