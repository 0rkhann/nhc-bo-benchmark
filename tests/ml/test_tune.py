import time
import numpy as np
import pytest

from ml_models.benchmark import models, tune


def toy(n=90):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4)); y = X[:, 0] + rng.normal(0, 0.1, n)
    return X, y, [f"S{i}" for i in range(n)]


def test_tune_respects_trial_cap():
    X, y, s = toy()
    best, log = tune.tune(models.get("ridge"), X, y, s, inner="cv3", n_trials=5, time_cap_s=60)
    assert len(log) == 5 and "alpha" in best


def test_tune_stops_at_time_cap():
    X, y, s = toy()
    t0 = time.time()
    _, log = tune.tune(models.get("rf"), X, y, s, inner="cv3", n_trials=10_000, time_cap_s=3)
    assert time.time() - t0 < 30 and 1 <= len(log) < 10_000


class Exploding:
    name, tunable, grid = "boom", True, None
    def defaults(self): return {"a": 1.0}
    def search_space(self, trial): return {"a": trial.suggest_float("a", 0, 1)}
    def fit(self, X, y, params, smiles=None): raise FloatingPointError("loss is NaN")
    def predict(self, X, smiles=None): raise AssertionError


def test_all_trials_failing_raises_with_last_error():           # Review Focus 1
    X, y, s = toy()
    with pytest.raises(tune.AllTrialsFailed, match="loss is NaN"):
        tune.tune(Exploding(), X, y, s, inner="cv3", n_trials=4, time_cap_s=60)


def test_grid_mode_tries_every_config():
    X, y, s = toy()
    a = models.get("ridge"); a.grid = [{"alpha": 0.1}, {"alpha": 10.0}]
    best, log = tune.tune(a, X, y, s, inner="cv3", n_trials=99, time_cap_s=60)
    assert len(log) == 2 and best["alpha"] in (0.1, 10.0)


class Slow:
    name, tunable, grid = "slow", True, None
    def defaults(self): return {"a": 0.5}
    def search_space(self, trial): return {"a": trial.suggest_float("a", 0, 1)}
    def fit(self, X, y, params, smiles=None):
        time.sleep(0.4); self.mu = float(np.mean(y)); return self
    def predict(self, X, smiles=None): return np.full(len(X), self.mu)


def test_tune_does_not_start_a_trial_that_would_overrun_the_cap():
    X, y, s = toy()                                    # each trial = 3 inner fits = about 1.2 s
    t0 = time.time()
    _, log = tune.tune(Slow(), X, y, s, inner="cv3", n_trials=100, time_cap_s=3.0)
    assert time.time() - t0 < 3.0 + 0.5               # never runs a full trial past the cap
    assert 1 <= len(log) <= 2
