"""Run one benchmark cell: tune on the outer training part, refit, score the outer test fold."""
import argparse
import importlib.metadata as md
import json
import os
import resource
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ml_models.benchmark import data, metrics, models, tune

ROOT = data.ROOT
FOLDS = ROOT / "ml_results" / "benchmark" / "folds.json"
OUT = ROOT / "ml_results" / "benchmark"
BUDGET = ROOT / "ml_models" / "benchmark" / "budget.json"   # cells reduced by the timing spike
TRIALS = {"ridge": 30, "rf": 50, "xgb": 50, "chemprop": 20}
INNER = {"chemprop": "holdout10"}
LC_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
PACKAGES = ["numpy", "scikit-learn", "xgboost", "torch", "botorch", "gpytorch", "tabpfn", "tabicl",
            "chemprop", "optuna"]


def _versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        return ""


def _features(rep: str, smiles: list[str]) -> np.ndarray:
    if rep == "smiles":
        return np.zeros((len(smiles), 0))
    return data.load_rep(rep, smiles)


def _fit_and_score(adapter, params, X, y, smiles, tr, te):
    s = list(np.asarray(smiles, dtype=object)[tr]), list(np.asarray(smiles, dtype=object)[te])
    pred = adapter.fit(X[tr], y[tr], params, smiles=s[0]).predict(X[te], smiles=s[1])
    return pred


def run_cell(model, rep, fold, out_root=OUT, n_trials=None, job_budget_s=300 * 60,
             learning_curve=False, folds_path=FOLDS):
    t0 = time.time()
    cell = Path(out_root) / model / rep / f"fold{fold}"
    cell.mkdir(parents=True, exist_ok=True)
    saved = data.load_folds(folds_path)
    result = {"model": model, "rep": rep, "fold": fold, "status": "ok", "error": None,
              "versions": _versions(), "git_sha": _git_sha(), "data_hashes": saved["data_hashes"],
              "run_id": os.environ.get("GITHUB_RUN_ID", ""), "checkpoint": None, "n_trials_done": 0}
    try:
        y_all = data.load_target()
        smiles = list(y_all.index)
        y = y_all.to_numpy()
        X = _features(rep, smiles)
        tr, te = data.split(saved["folds"], fold, smiles)
        adapter = models.get(model)
        result["checkpoint"] = getattr(adapter, "checkpoint", None)
        params = {}
        reduced = json.loads(BUDGET.read_text()).get("reduced", {}).get(f"{model}/{rep}") if BUDGET.exists() else None
        if reduced:
            params = dict(reduced["params"])
            result["reduced"] = reduced["reason"]
            pd.DataFrame().to_csv(cell / "trials.csv", index=False)
        elif adapter.tunable:
            refit_reserve = getattr(adapter, "refit_estimate_s", 60) * 2
            cap = max(60.0, job_budget_s - (time.time() - t0) - refit_reserve)
            params, log = tune.tune(adapter, X[tr], y[tr], list(np.asarray(smiles, dtype=object)[tr]),
                                    INNER.get(model, "cv3"), n_trials or TRIALS.get(model, 50), cap)
            log.to_csv(cell / "trials.csv", index=False)
            result["n_trials_done"] = int((log["state"] == "COMPLETE").sum())
        else:
            pd.DataFrame().to_csv(cell / "trials.csv", index=False)
        (cell / "best_params.json").write_text(json.dumps(params, indent=1))
        pred = _fit_and_score(adapter, params, X, y, smiles, tr, te)
        pd.DataFrame({"SMILES": np.asarray(smiles, dtype=object)[te], "true": y[te], "pred": pred}).to_csv(
            cell / "predictions.csv", index=False)
        result["metrics"] = metrics.fold_metrics(y[te], pred, ranking_only=(model == "outlier"))
        if learning_curve:
            rng = np.random.default_rng(0)
            order = rng.permutation(tr)                 # nested subsets: prefixes of one permutation
            result["learning_curve"] = []
            for f in LC_FRACTIONS:
                sub = order[: max(2, int(round(f * len(tr))))]
                p = _fit_and_score(models.get(model), params, X, y, smiles, sub, te)
                result["learning_curve"].append({"fraction": f, "n_train": int(len(sub)),
                                                 **metrics.fold_metrics(y[te], p)})
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}")
    result["wall_s"] = round(time.time() - t0, 1)
    result["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    (cell / "metrics.json").write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(models.REGISTRY))
    ap.add_argument("--rep", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--job-budget-min", type=float, default=300)  # 50 min under the 350-min job timeout
    ap.add_argument("--learning-curve", action="store_true")
    a = ap.parse_args()
    if not FOLDS.exists():
        y = data.load_target()
        data.save_folds(data.make_folds(list(y.index)), FOLDS)
    r = run_cell(a.model, a.rep, a.fold, n_trials=a.n_trials, job_budget_s=a.job_budget_min * 60,
                 learning_curve=a.learning_curve)
    print(json.dumps({k: r[k] for k in ("model", "rep", "fold", "status", "wall_s")}))
    raise SystemExit(0 if r["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
