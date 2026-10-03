"""Combine benchmark cells into summary tables, with provenance checks."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ml_models.benchmark import stats

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "ml_results" / "benchmark"
METRICS = ["r2", "rmse", "mae", "spearman", "top1_recall", "tail_rmse"]
TABPFN_NOTICE = ("Predictions in this folder were produced with TabPFN-3 weights, released under the TabPFN-3\n"
                 "non-commercial licence (https://huggingface.co/Prior-Labs/tabpfn_3). They may only be used for\n"
                 "non-commercial purposes.\n")


def collect(root: Path = BENCH) -> pd.DataFrame:
    rows = []
    for cell in sorted(Path(root).glob("*/*/fold*")):
        model, rep, fold = cell.parts[-3], cell.parts[-2], int(cell.name[4:])
        f = cell / "metrics.json"
        if not f.exists():
            rows.append({"model": model, "rep": rep, "fold": fold, "status": "missing"})
            continue
        m = json.loads(f.read_text())
        rows.append({"model": model, "rep": rep, "fold": fold, "status": m["status"],
                     "wall_s": m.get("wall_s", np.nan), "data_hashes": json.dumps(m.get("data_hashes"), sort_keys=True),
                     "versions": json.dumps(m.get("versions"), sort_keys=True), **(m.get("metrics") or {})})
    return pd.DataFrame(rows)


def check_provenance(cells: pd.DataFrame) -> None:
    ok = cells[cells.status == "ok"]
    for (model, rep), g in ok.groupby(["model", "rep"]):
        if g.data_hashes.nunique() > 1:
            raise ValueError(f"{model}/{rep}: folds were computed on different data files")
        if g.versions.nunique() > 1:
            raise ValueError(f"{model}/{rep}: folds were computed with different package versions")


def summary(cells: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, rep), g in cells.groupby(["model", "rep"]):
        ok = g[g.status == "ok"]
        row = {"model": model, "rep": rep, "folds_ok": len(ok), "folds_total": len(g),
               "cpu_minutes": round(ok.wall_s.sum() / 60, 1) if "wall_s" in ok else np.nan}
        for m in METRICS:
            if m in ok and ok[m].notna().any():
                row[f"{m}_mean"], row[f"{m}_sd"] = ok[m].mean(), ok[m].std(ddof=1)
        out.append(row)
    return pd.DataFrame(out)


def _paired(cells, model_a, rep_a, model_b, rep_b, metric):
    a = cells[(cells.model == model_a) & (cells.rep == rep_a) & (cells.status == "ok")].set_index("fold")[metric]
    b = cells[(cells.model == model_b) & (cells.rep == rep_b) & (cells.status == "ok")].set_index("fold")[metric]
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return None
    return stats.corrected_ttest(a[common].to_numpy(), b[common].to_numpy())


def comparisons(cells: pd.DataFrame) -> pd.DataFrame:
    s = summary(cells)
    full = s[(s.folds_ok == 5) & s.get("r2_mean", pd.Series(dtype=float)).notna()]
    rows = []
    for rep, g in full.groupby("rep"):
        for model in g.model:
            if model == "rf":
                continue
            for metric in ("r2", "top1_recall"):
                r = _paired(cells, model, rep, "rf", rep, metric)
                if r:
                    rows.append({"family": "vs_rf", "rep": rep, "model": model, "ref": "rf", "metric": metric,
                                 f"delta_{metric}": r[0], "delta": r[0], "p": r[1]})
        best = g.sort_values("r2_mean").iloc[-1].model
        for model in g.model:
            if model != best:
                r = _paired(cells, best, rep, model, rep, "r2")
                if r:
                    rows.append({"family": "best_vs_others", "rep": rep, "model": best, "ref": model,
                                 "metric": "r2", "delta": r[0], "p": r[1]})
    for model, g in full.groupby("model"):
        if len(g) > 1:
            best_rep = g.sort_values("r2_mean").iloc[-1].rep
            for rep in g.rep:
                if rep != best_rep:
                    r = _paired(cells, model, best_rep, model, rep, "r2")
                    if r:
                        rows.append({"family": "across_reps", "rep": best_rep, "model": model, "ref": rep,
                                     "metric": "r2", "delta": r[0], "p": r[1]})
    c = pd.DataFrame(rows)
    if c.empty:
        return c
    c["p_holm"] = np.nan
    for fam, idx in c.groupby(["family", "metric"]).groups.items():
        c.loc[idx, "p_holm"] = stats.holm(c.loc[idx, "p"].tolist())
    vr = c[(c.family == "vs_rf") & (c.metric == "r2")]
    moved = bool(((vr.delta >= stats.CEILING_DELTA_R2) & (vr.p_holm < stats.ALPHA)).any())
    c["delta_r2"] = np.where(c.metric == "r2", c.delta, np.nan)
    return pd.concat([c, pd.DataFrame([{"family": "ceiling", "ceiling_moved": moved}])], ignore_index=True)


def oof(root: Path, model: str, rep: str) -> pd.DataFrame:
    parts = [pd.read_csv(f) for f in sorted(Path(root, model, rep).glob("fold*/predictions.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def story_table(cells: pd.DataFrame, bo_summary: Path = ROOT / "analysis" / "summary.csv") -> pd.DataFrame:
    s = summary(cells)
    bo = pd.read_csv(bo_summary)
    bo = bo[~bo.method.isin(["random", "outlier"])].groupby("rep").auc_median.min()
    rows = []
    for rep, g in s[s.rep.isin(bo.index)].groupby("rep"):
        trained = g[~g.model.isin(["outlier", "mean"])].dropna(subset=["top1_recall_mean"])
        best = trained.sort_values("top1_recall_mean").iloc[-1] if len(trained) else None
        gp = g[g.model == "gp"]; out = g[g.model == "outlier"]
        rows.append({"rep": rep, "best_model": None if best is None else best.model,
                     "best_top1_recall": None if best is None else best.top1_recall_mean,
                     "gp_top1_recall": gp.top1_recall_mean.iloc[0] if len(gp) else np.nan,
                     "outlier_top1_recall": out.top1_recall_mean.iloc[0] if len(out) else np.nan,
                     "bo_best_regret_auc": bo[rep]})
    return pd.DataFrame(rows)


def merge(new_root: Path, existing_root: Path) -> None:
    for cell in Path(new_root).glob("*/*/fold*"):
        dest = Path(existing_root) / cell.relative_to(new_root)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(cell, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=BENCH)
    a = ap.parse_args()
    cells = collect(a.root)
    check_provenance(cells)
    summary(cells).to_csv(a.root / "summary.csv", index=False)
    comparisons(cells).to_csv(a.root / "comparisons.csv", index=False)
    story_table(cells).to_csv(a.root / "story_links.csv", index=False)
    if (a.root / "tabpfn").is_dir():
        (a.root / "tabpfn" / "NOTICE").write_text(TABPFN_NOTICE)
    (a.root / "oof_predictions").mkdir(exist_ok=True)
    for (model, rep), _ in cells[cells.status == "ok"].groupby(["model", "rep"]):
        oof(a.root, model, rep).to_csv(a.root / "oof_predictions" / f"{model}_{rep}.csv", index=False)
    print(summary(cells).to_string(index=False))


if __name__ == "__main__":
    main()
