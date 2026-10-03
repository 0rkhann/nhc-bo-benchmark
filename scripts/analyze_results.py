#!/usr/bin/env python3
"""Per-run metrics, summary tables and paired tests for the BO experiments.

Reads results/**/*history.csv and ml_results/benchmark/summary.csv. The only other
input is the data the pipelines themselves read (data/<rep>.csv, data/dft_G.json),
used to rebuild each seed's candidate pool with the same train_test_split calls as
baselines/random_search.py. The rebuilt pool optimum is asserted equal to
`best_pool_min` in every history file, so the split is verified, not assumed.

Writes analysis/per_run_metrics.csv, analysis/summary.csv, analysis/paired_tests.csv
and prints Markdown tables.

Metrics (per run, 100 iterations, `best` = running best energy, lower is better):
  final_best      best energy after 100 iterations
  it_optimum      first iteration with best <= pool optimum; 100 if never (censored)
  hit_optimum     whether it was reached
  it_top1         first iteration with best <= 1st percentile of the pool energies; 100 if never
  regret_auc      mean over iterations of (best - pool_opt) / (init_best - pool_opt),
                  clipped at 0 above; init_best = best of the 10-molecule initial design
                  (identical for every method of a seed)

Usage: python scripts/analyze_results.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RESULTS, ML_RESULTS, OUT = ROOT / "results", ROOT / "ml_results", ROOT / "analysis"
REPS = {"dft_descriptors": "DFT descriptors", "dft_chemberta2": "ChemBERTa-2", "dft_mordred": "Mordred"}
BO = ["fabo", "pca", "pls", "opls", "vanilla"]
METHODS = BO + ["random", "outlier"]
SEEDS = range(42, 62)
N_ITER, N_INITIAL, TEST_FRAC = 100, 10, 0.1


def history_path(rep, method, seed):
    if method in BO:
        d = RESULTS / rep / f"{method}_Matern_EI" / f"seed{seed}"
        return next(d.glob("**/bo_iteration_history.csv")), "best_bo"
    tag = {"random": "random_search", "outlier": "outlier_search"}[method]
    return RESULTS / f"{rep}_{tag}" / f"seed{seed}" / f"{tag}_history.csv", f"best_{method}"


def split_pool(rep, seed):
    """Initial design and candidate pool SMILES, as in baselines/random_search.py."""
    smiles = pd.read_csv(ROOT / "data" / f"{rep}.csv", usecols=["SMILES"])["SMILES"].tolist()
    tmp, _ = train_test_split(smiles, test_size=TEST_FRAC, random_state=seed, shuffle=True)
    init, pool = train_test_split(tmp, train_size=N_INITIAL, random_state=seed, shuffle=True)
    return init, pool


def first_hit(best, thr):
    hit = np.flatnonzero(best <= thr + 1e-6)
    return (int(hit[0]) + 1, True) if len(hit) else (N_ITER, False)


def per_run():
    energy = {d["SMILES"]: d["energy"] for d in json.loads((ROOT / "data" / "dft_G.json").read_text())}
    rows = []
    for rep in REPS:
        for seed in SEEDS:
            init, pool = split_pool(rep, seed)
            pe = np.array([energy[s] for s in pool])
            opt, top1 = pe.min(), np.quantile(pe, 0.01)
            init_best = min(energy[s] for s in init)
            for m in METHODS:
                path, col = history_path(rep, m, seed)
                h = pd.read_csv(path)
                assert len(h) == N_ITER, path
                assert abs(h["best_pool_min"].iloc[-1] - opt) < 1e-6, f"pool mismatch {path}"
                best = np.minimum.accumulate(h[col].to_numpy())
                it_opt, hit = first_hit(best, opt)
                it_top1, _ = first_hit(best, top1)
                denom = max(init_best - opt, 1e-9)
                regret = np.clip((best - opt) / denom, 0, 1)
                rows.append(dict(rep=rep, method=m, seed=seed, final_best=best[-1], pool_opt=opt,
                                 pool_top1=top1, pool_size=len(pool), init_best=init_best,
                                 it_optimum=it_opt, hit_optimum=hit, it_top1=it_top1,
                                 regret_auc=regret.mean()))
    return pd.DataFrame(rows)


def summarize(runs):
    rows = []
    for (rep, m), g in runs.groupby(["rep", "method"], sort=False):
        rows.append(dict(
            rep=rep, method=m, n=len(g),
            best_mean=g.final_best.mean(), best_sd=g.final_best.std(ddof=1),
            hits=int(g.hit_optimum.sum()), it_opt_median=g.it_optimum.median(),
            top1_hits=int(g.it_top1.lt(N_ITER).sum()),
            it_top1_median=g.it_top1.median(),
            it_top1_iqr_lo=g.it_top1.quantile(0.25), it_top1_iqr_hi=g.it_top1.quantile(0.75),
            auc_median=g.regret_auc.median(), auc_mean=g.regret_auc.mean(),
        ))
    return pd.DataFrame(rows)


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation of paired differences d = a - b.
    Negative: a is lower (better, for both metrics) than b."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_tests(runs):
    rows = []
    for rep in REPS:
        g = runs[runs.rep == rep]
        for metric in ["it_top1", "regret_auc"]:
            wide = g.pivot(index="seed", columns="method", values=metric)
            for ref in ["random", "outlier"]:
                for m in BO:
                    d = (wide[m] - wide[ref]).to_numpy()
                    p = 1.0 if np.all(d == 0) else wilcoxon(d, zero_method="wilcox").pvalue
                    rows.append(dict(rep=rep, metric=metric, method=m, ref=ref,
                                     median_diff=float(np.median(d)), rank_biserial=rank_biserial(d),
                                     wins=int((d < 0).sum()), losses=int((d > 0).sum()),
                                     ties=int((d == 0).sum()), p=p))
    t = pd.DataFrame(rows)
    # Holm correction within each metric x reference family (3 representations x 5 methods)
    t["p_holm"] = np.nan
    for _, idx in t.groupby(["metric", "ref"]).groups.items():
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "p"].to_numpy())
    return t


def facts(runs):
    out = []
    for m in ["random"]:
        a = [pd.read_csv(history_path("dft_chemberta2", m, s)[0]) for s in SEEDS]
        b = [pd.read_csv(history_path("dft_mordred", m, s)[0]) for s in SEEDS]
        same = sum(x["smiles"].equals(y["smiles"]) and x["energy"].equals(y["energy"]) for x, y in zip(a, b))
        out.append(f"Random-search histories identical (SMILES and energies) between ChemBERTa-2 and Mordred: {same}/{len(a)} seeds")
        c = [pd.read_csv(history_path("dft_descriptors", m, s)[0]) for s in SEEDS]
        same = sum(x["smiles"].equals(y["smiles"]) for x, y in zip(a, c))
        out.append(f"Random-search histories identical between ChemBERTa-2 and descriptors: {same}/{len(a)} seeds")
    # why the outlier heuristic works on descriptors
    desc = pd.read_csv(ROOT / "data" / "dft_descriptors.csv")
    X = desc.drop(columns="SMILES").to_numpy(dtype=float)
    mu, sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
    z = np.abs((X - mu) / sd)
    score = z.max(1)
    energy = {d["SMILES"]: d["energy"] for d in json.loads((ROOT / "data" / "dft_G.json").read_text())}
    e = desc["SMILES"].map(energy).to_numpy()
    cols = desc.columns[1:]
    g = runs[(runs.rep == "dft_descriptors") & (runs.method == "outlier")]
    out.append(f"Outlier heuristic on descriptors: pool size {int(g.pool_size.iloc[0])}; "
               f"median iteration reaching the pool optimum {g.it_optimum.median():.0f} "
               f"(= top {100 * g.it_optimum.median() / g.pool_size.iloc[0]:.2f}% of the pool by outlier score)")
    out.append(f"Global energy optimum {e.min():.4f}: rank {int((score > score[e.argmin()]).sum()) + 1} of {len(e)} by max|z| "
               f"({score[e.argmin()]:.1f} sigma on '{cols[z[e.argmin()].argmax()]}')")
    top = np.argsort(e)[:10]
    out.append("Ten lowest-energy molecules: outlier-score rank (of %d) = %s" %
               (len(e), sorted(int((score > score[i]).sum()) + 1 for i in top)))
    out.append(f"Spearman correlation between max|z| and energy over all molecules: "
               f"{pd.Series(score).corr(pd.Series(e), method='spearman'):.3f}")
    return out


def md_summary(s):
    for rep, label in REPS.items():
        print(f"\n#### {label}\n")
        print("| Method | Final best (mean ± sd) | Hit pool optimum | Iter. to optimum (median) "
              "| Reached top 1% | Iter. to top 1% (median [IQR]) | Regret AUC (median) |")
        print("|---|---|---|---|---|---|---|")
        for _, r in s[s.rep == rep].iterrows():
            print(f"| {r.method} | {r.best_mean:.2f} ± {r.best_sd:.2f} | {r.hits}/{r.n} | {r.it_opt_median:.0f} "
                  f"| {r.top1_hits}/{r.n} | {r.it_top1_median:.1f} [{r.it_top1_iqr_lo:.0f}–{r.it_top1_iqr_hi:.0f}] "
                  f"| {r.auc_median:.3f} |")


def md_tests(t):
    for metric, label in [("it_top1", "iterations to top 1%"), ("regret_auc", "regret AUC")]:
        for ref in ["random", "outlier"]:
            print(f"\n#### {label}: BO method vs `{ref}` (paired by seed; negative = method better)\n")
            print("| Representation | Method | Median diff | Rank-biserial r | W / L / T | p | p (Holm) |")
            print("|---|---|---|---|---|---|---|")
            for _, r in t[(t.metric == metric) & (t.ref == ref)].iterrows():
                print(f"| {REPS[r.rep]} | {r.method} | {r.median_diff:+.3g} | {r.rank_biserial:+.2f} "
                      f"| {r.wins}/{r.losses}/{r.ties} | {r.p:.3g} | {r.p_holm:.3g} |")


def md_ml():
    f = ML_RESULTS / "benchmark" / "summary.csv"
    print("\n#### ML benchmark (5-fold nested CV, mean ± sd)\n")
    if not f.exists():
        print("ml_results/benchmark/summary.csv not found; run the ML benchmark workflow first.")
        return
    s = pd.read_csv(f)
    print("| Representation | Model | Folds | R² | RMSE | Spearman | Top-1% recall |")
    print("|---|---|---|---|---|---|---|")
    fmt = lambda r, m, d: f"{r[m + '_mean']:.{d}f} ± {r[m + '_sd']:.{d}f}" if pd.notna(r.get(m + "_mean")) else "—"
    for _, r in s.sort_values(["rep", "model"]).iterrows():
        print(f"| {r.rep} | {r.model} | {r.folds_ok}/{r.folds_total} | {fmt(r, 'r2', 3)} | {fmt(r, 'rmse', 2)} "
              f"| {fmt(r, 'spearman', 3)} | {fmt(r, 'top1_recall', 2)} |")


def main():
    OUT.mkdir(exist_ok=True)
    runs = per_run()
    runs.to_csv(OUT / "per_run_metrics.csv", index=False)
    s = summarize(runs)
    s.to_csv(OUT / "summary.csv", index=False)
    t = paired_tests(runs)
    t.to_csv(OUT / "paired_tests.csv", index=False)
    print(f"{len(runs)} runs; pool optimum per seed, by representation:")
    for rep, g in runs.groupby("rep"):
        print(f"  {rep}: {g.pool_opt.min():.2f} to {g.pool_opt.max():.2f}, "
              f"top-1% threshold {g.pool_top1.min():.2f} to {g.pool_top1.max():.2f}, pool size {sorted(set(g.pool_size))}")
    md_summary(s)
    md_tests(t)
    md_ml()
    print("\n#### Facts\n")
    for line in facts(runs):
        print(f"- {line}")


if __name__ == "__main__":
    main()
