"""Optuna tuning on inner splits of the outer training part."""
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)


class AllTrialsFailed(RuntimeError):
    pass


def _inner_splits(n: int, inner: str):
    idx = np.arange(n)
    if inner == "cv3":
        return list(KFold(3, shuffle=True, random_state=0).split(idx))
    if inner == "holdout10":
        tr, va = train_test_split(idx, test_size=0.1, random_state=0)
        return [(tr, va)]
    raise ValueError(f"unknown inner scheme {inner!r}")


def tune(adapter, X, y, smiles, inner: str, n_trials: int, time_cap_s: float):
    splits = _inner_splits(len(y), inner)
    smiles = np.asarray(smiles, dtype=object)
    last_error: list[str] = []

    def objective(trial):
        params = adapter.search_space(trial)
        rmses = []
        try:
            for tr, va in splits:
                m = type(adapter)()
                if adapter.grid is not None:
                    m.grid = adapter.grid
                pred = m.fit(X[tr], y[tr], params, smiles=list(smiles[tr])).predict(X[va], smiles=list(smiles[va]))
                rmses.append(float(np.sqrt(np.mean((y[va] - pred) ** 2))))
        except Exception as exc:                      # a failed trial must not stop the study
            last_error[:] = [f"{type(exc).__name__}: {exc}"]
            raise optuna.TrialPruned(last_error[0])
        return float(np.mean(rmses))

    if adapter.grid is not None:
        keys = sorted({k for g in adapter.grid for k in g})
        sampler = optuna.samplers.GridSampler({k: sorted({g[k] for g in adapter.grid}) for k in keys}, seed=0)
        n_trials = len(adapter.grid)
    else:
        sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    t_start, slowest = time.time(), [0.0]

    def stop_before_overrun(study, trial):
        # Optuna only checks its timeout between trials, so stop once the slowest trial so far
        # would no longer fit in the remaining time.
        if trial.datetime_complete and trial.datetime_start:
            slowest[0] = max(slowest[0], (trial.datetime_complete - trial.datetime_start).total_seconds())
        if time.time() - t_start + slowest[0] > time_cap_s:
            study.stop()

    study.optimize(objective, n_trials=n_trials, timeout=time_cap_s, catch=(), callbacks=[stop_before_overrun])
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    log = study.trials_dataframe()
    if not done:
        raise AllTrialsFailed(last_error[0] if last_error else "no trial completed within the time cap")
    return dict(study.best_trial.params), log
