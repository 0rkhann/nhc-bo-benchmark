#!/usr/bin/env python3
"""Figures for the README: convergence, iterations to top 1%, regret AUC + effect sizes,
the descriptor-space outlier plot (plots/), and the ML benchmark and parity plots (ml_plots/).

Every number comes from the CSVs in results/, ml_results/benchmark/ and analysis/*.csv
(written by scripts/analyze_results.py); figure 4 also reads data/dft_descriptors.csv
and data/dft_G.json. Each figure is written as SVG (README) and PDF (papers, slides).

Usage: python scripts/analyze_results.py && python plotting/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_results as ar  # noqa: E402

# Okabe-Ito, one fixed colour per method in every figure; the two controls are dashed.
COLOR = {"fabo": "#E69F00", "pca": "#56B4E9", "pls": "#009E73", "opls": "#D55E00",
         "vanilla": "#CC79A7", "random": "#9AA0A6", "outlier": "#0072B2"}
STYLE = {"random": "--", "outlier": "-."}
LABEL = {"fabo": "FABO", "pca": "PCA", "pls": "PLS", "opls": "OPLS", "vanilla": "Vanilla GP",
         "random": "Random", "outlier": "Outlier (no model)"}
# Each figure is drawn twice, with text colours matched to GitHub's light and dark themes; the README
# picks one with <picture> and prefers-color-scheme. Font sizes assume the figure is shown about
# 900 px wide, so the smallest text still renders at roughly 12 px.
THEMES = {"light": ("#1f2328", "#d0d7de"), "dark": ("#e6edf3", "#30363d")}
INK, GRID = THEMES["light"]
THEME = "light"


def apply_theme(theme):
    global INK, GRID, THEME
    THEME = theme
    INK, GRID = THEMES[theme]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 13,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "figure.facecolor": "none", "axes.facecolor": "none",
        "savefig.transparent": True,
    })

PLOTS, ML_PLOTS = ROOT / "plots", ROOT / "ml_plots"
REPS = ar.REPS


def save(fig, out_dir, name):
    """Light theme: <name>.svg (README default) and <name>.pdf (papers, slides). Dark: <name>-dark.svg."""
    out_dir.mkdir(exist_ok=True)
    exts, stem = (("svg", "pdf"), name) if THEME == "light" else (("svg",), f"{name}-dark")
    for ext in exts:
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)
    print(f"wrote {out_dir.name}/{stem}." + " and .".join(exts))


def method_legend(fig, y=1.06, ncol=4):
    handles = [Line2D([], [], color=COLOR[m], lw=2.2, ls=STYLE.get(m, "-"), label=LABEL[m]) for m in ar.METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, columnspacing=1.4,
               handlelength=2.6)


def running_best(rep, method, seed):
    path, col = ar.history_path(rep, method, seed)
    return np.minimum.accumulate(pd.read_csv(path)[col].to_numpy())


# ---- Figure 1: convergence ----------------------------------------------------------------------
def fig_convergence(runs):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    x = np.arange(1, ar.N_ITER + 1)
    for ax, (rep, label) in zip(axes, REPS.items()):
        for m in ar.METHODS:
            curves = np.stack([running_best(rep, m, s) for s in ar.SEEDS])
            lo, med, hi = np.percentile(curves, [25, 50, 75], axis=0)
            ax.fill_between(x, lo, hi, color=COLOR[m], alpha=0.13, lw=0)
            ax.plot(x, med, color=COLOR[m], ls=STYLE.get(m, "-"), lw=2.0 if m not in ("random", "outlier") else 1.8)
        opt = runs[runs.rep == rep].pool_opt.median()
        ax.axhline(opt, color=INK, lw=0.9, ls=":")
        ax.text(2, opt - 0.8, f"median pool optimum {opt:.1f}", ha="left", va="top", fontsize=12)
        ax.set_ylim(-46, None)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xlabel("BO iteration")
        ax.set_xlim(1, ar.N_ITER)
    axes[0].set_ylabel("Best energy found (lower is better)")
    method_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, PLOTS, "fig1_convergence")


# ---- Figure 2: iterations to top 1% --------------------------------------------------------------
def fig_top1(runs, tests):
    t = tests[(tests.metric == "it_top1") & (tests.ref == "outlier")].set_index(["rep", "method"])
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (rep, label) in zip(axes, REPS.items()):
        for i, m in enumerate(ar.METHODS):
            v = runs[(runs.rep == rep) & (runs.method == m)].it_top1.to_numpy()
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=14, color=COLOR[m], alpha=0.75, lw=0, zorder=3)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.vlines(i, q1, q3, color=COLOR[m], lw=5, alpha=0.35, zorder=2)
            ax.hlines(med, i - 0.3, i + 0.3, color=COLOR[m], lw=2.6, zorder=4)
            if (rep, m) in t.index and t.loc[(rep, m)].p_holm < 0.05:
                faster = t.loc[(rep, m)].rank_biserial < 0
                ax.text(i, 108, "*" if faster else "†", ha="center", va="center", fontsize=17, color=INK)
        ax.axhline(ar.N_ITER, color=INK, lw=1.0, ls=":")
        ax.text(len(ar.METHODS) - 0.5, ar.N_ITER + 1, "not reached", ha="right", va="bottom", fontsize=12)
        ax.set_xticks(range(len(ar.METHODS)))
        ax.set_xticklabels([LABEL[m].replace(" (no model)", "") for m in ar.METHODS], rotation=45, ha="right")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_ylim(-3, 114)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Iterations to reach the top 1% of the pool\n(lower is faster)")
    fig.text(0.5, -0.1, "Dots: seeds (20). Bar: IQR. Tick: median.\n* significantly faster than the outlier "
             "heuristic, † significantly slower (paired Wilcoxon, Holm p < 0.05)", ha="center", fontsize=12.5)
    fig.tight_layout()
    save(fig, PLOTS, "fig2_iterations_to_top1")


# ---- Figure 3: regret AUC + effect sizes ---------------------------------------------------------
def fig_regret(runs, tests):
    """Top: median regret AUC per method and representation. Bottom: paired effect sizes as a heatmap."""
    fig = plt.figure(figsize=(13, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.55, wspace=0.08)
    ys = np.arange(len(ar.METHODS))[::-1]
    first = None
    for k, (rep, label) in enumerate(REPS.items()):
        ax = fig.add_subplot(gs[0, k], sharey=first)
        first = first or ax
        for y, m in zip(ys, ar.METHODS):
            v = runs[(runs.rep == rep) & (runs.method == m)].regret_auc.to_numpy()
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.hlines(y, q1, q3, color=COLOR[m], lw=4, alpha=0.5)
            ax.plot(med, y, "o", color=COLOR[m], ms=9)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "", "0.5", "", "1"])
        ax.set_ylim(-0.5, len(ar.METHODS) - 0.5)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.grid(axis="y", visible=False)
        if k == 0:
            ax.set_yticks(ys)
            ax.set_yticklabels([LABEL[m].replace(" (no model)", "") for m in ar.METHODS])
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if k == 1:
            ax.set_xlabel("Regret AUC: median and IQR over 20 seeds (lower is better)")

    ax = fig.add_subplot(gs[1, :])
    cols = [(rep, ref) for ref in ("outlier", "random") for rep in REPS]
    t = tests[tests.metric == "regret_auc"].set_index(["rep", "ref", "method"])
    cmap = plt.get_cmap("PuOr")  # negative r (method better) -> orange, positive -> purple
    hy = np.arange(len(ar.BO))[::-1]
    for j, (rep, ref) in enumerate(cols):
        for y, m in zip(hy, ar.BO):
            r = t.loc[(rep, ref, m)]
            ax.add_patch(plt.Rectangle((j - 0.48, y - 0.46), 0.96, 0.92, color=cmap(0.5 + 0.35 * r.rank_biserial), lw=0))
            sig = r.p_holm < 0.05
            ax.text(j, y, f"{r.rank_biserial:+.2f}" + ("*" if sig else ""), ha="center", va="center", fontsize=13,
                    fontweight="bold" if sig else "normal", color="black")
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(-0.5, len(ar.BO) + 0.3)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([REPS[rep] for rep, _ in cols], fontsize=12.5)
    ax.set_yticks(hy)
    ax.set_yticklabels([LABEL[m] for m in ar.BO])
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(2.5, color=INK, lw=1.2)
    ax.text(1, len(ar.BO) - 0.05, "BO method vs outlier heuristic", ha="center", fontweight="bold")
    ax.text(4, len(ar.BO) - 0.05, "BO method vs random search", ha="center", fontweight="bold")
    ax.set_xlabel("Paired rank-biserial r on regret AUC (negative = BO method better;  * Holm p < 0.05)")
    save(fig, PLOTS, "fig3_regret_auc")


# ---- Figure 4: descriptor space ------------------------------------------------------------------
def fig_descriptor_space():
    desc = pd.read_csv(ROOT / "data" / "dft_descriptors.csv")
    X = desc.drop(columns="SMILES").to_numpy(dtype=float)
    sd = np.where(X.std(0) > 0, X.std(0), 1.0)
    z = np.abs((X - X.mean(0)) / sd)
    score = z.max(1)
    energy = {d["SMILES"]: d["energy"] for d in json.loads((ROOT / "data" / "dft_G.json").read_text())}
    e = desc["SMILES"].map(energy).to_numpy()
    cols = desc.columns[1:]
    top = np.argsort(e)[:10]
    best = top[0]
    hi = COLOR["opls"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    a.hist(score, bins=70, color=INK, alpha=0.45, lw=0)
    a.set_yscale("log")
    for i in top:
        a.axvline(score[i], color=hi, lw=1.0, alpha=0.8)
    a.legend(handles=[Line2D([], [], color=hi, lw=1.5, label="10 lowest-energy molecules")], loc="upper right")
    a.set_xlabel("Outlier score: max |z| over the 29 descriptors")
    a.set_ylabel("Molecules (log scale)")
    a.set_title("Outlier score distribution", loc="left", fontweight="bold")

    b.scatter(score, e, s=7, color=INK, alpha=0.35, lw=0)
    b.scatter(score[top], e[top], s=18, color=hi, lw=0)
    b.scatter(score[best], e[best], s=130, facecolor="none", edgecolor=hi, lw=2)
    b.annotate(f"optimum {e[best]:.2f}\n{score[best]:.1f}σ on '{cols[z[best].argmax()]}'",
               (score[best], e[best]), xytext=(14, 8), textcoords="offset points", fontsize=12.5, color=INK)
    b.set_xlabel("Outlier score: max |z|")
    b.set_ylabel("xTB binding free energy")
    b.set_title("Energy vs outlier score", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, PLOTS, "fig4_descriptor_outliers")


# ---- Parity plot (limitations) -------------------------------------------------------------------
ML_BENCH = ROOT / "ml_results" / "benchmark"
ML_ORDER = ["mean", "ridge", "rf", "xgb", "gp", "tabpfn", "tabicl", "chemprop", "outlier"]
ML_LABEL = {"mean": "Mean predictor", "ridge": "Ridge", "rf": "Random Forest", "xgb": "XGBoost",
            "gp": "GP (BO surrogate)", "tabpfn": "TabPFN-3", "tabicl": "TabICLv2", "chemprop": "Chemprop",
            "outlier": "Outlier score (no model)"}
ML_REP = {"dft_descriptors": ("DFT descriptors", "#E69F00", "o"), "dft_chemberta2": ("ChemBERTa-2", "#56B4E9", "s"),
          "dft_mordred": ("Mordred", "#009E73", "D"), "smiles": ("SMILES graph", "#999999", "^")}


def _ml_summary():
    f = ML_BENCH / "summary.csv"
    if not f.exists():
        print("skip ML figures: ml_results/benchmark/summary.csv not found")
        return None
    return pd.read_csv(f)


def fig_ml_benchmark():
    s = _ml_summary()
    if s is None:
        return
    models = [m for m in ML_ORDER if m in set(s.model)]
    panels = [("r2", "R² (5-fold, mean ± sd)", 0.0), ("spearman", "Spearman ρ", None),
              ("top1_recall", "Top-1% recall at 5%", 0.05)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.62 * len(models) + 2.6), sharey=True)
    ys = {m: len(models) - 1 - i for i, m in enumerate(models)}
    offsets = dict(zip(ML_REP, np.linspace(-0.28, 0.28, len(ML_REP))))
    for ax, (metric, title, ref) in zip(axes, panels):
        for _, r in s.iterrows():
            if r.model not in ys or pd.isna(r.get(f"{metric}_mean")):
                continue
            label, col, marker = ML_REP[r.rep]
            y = ys[r.model] + offsets[r.rep]
            ax.errorbar(r[f"{metric}_mean"], y, xerr=r.get(f"{metric}_sd", 0) or 0, fmt=marker, color=col,
                        ms=8, elinewidth=2, capsize=0, label=label)
        if ref is not None:
            ax.axvline(ref, color=INK, lw=1, ls=":")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(list(ys.values()))
    axes[0].set_yticklabels([ML_LABEL[m] for m in models])
    handles = {h.get_label(): h for ax in axes for h in ax.get_legend_handles_labels()[0]}
    fig.legend(handles.values(), handles.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=5,
               handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, ML_PLOTS, "ml_benchmark")


def fig_parity():
    """Out-of-fold predictions of the best model (highest mean R²) for each representation."""
    s = _ml_summary()
    if s is None:
        return
    trained = s[~s.model.isin(["mean", "outlier"])].dropna(subset=["r2_mean"])
    groups = [("dft_descriptors",), ("dft_chemberta2",), ("dft_mordred",), ("smiles",)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
    for ax, reps in zip(axes.flat, groups):
        g = trained[trained.rep.isin(reps)]
        if g.empty:
            ax.set_visible(False)
            continue
        best = g.sort_values("r2_mean").iloc[-1]
        df = pd.read_csv(ML_BENCH / "oof_predictions" / f"{best.model}_{best.rep}.csv")
        y, p = df["true"].to_numpy(), df["pred"].to_numpy()
        label, col, _ = ML_REP[best.rep]
        ax.scatter(y, p, s=6, color=col, alpha=0.35, lw=0)
        lim = [min(y.min(), p.min()) - 1, max(y.max(), p.max()) + 1]
        ax.plot(lim, lim, color=INK, lw=1, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.text(0.04, 0.95, f"{ML_LABEL[best.model]}\nR² = {best.r2_mean:.3f} ± {best.r2_sd:.3f}\nn = {len(y):,} ({best.folds_ok} folds)",
                transform=ax.transAxes, va="top", fontsize=13)
        ax.set_title(label, loc="left", fontweight="bold")
    for ax in axes[1]:
        ax.set_xlabel("True energy, kJ/mol (out-of-fold)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted energy, kJ/mol")
    fig.tight_layout()
    save(fig, ML_PLOTS, "ml_parity")


def main():
    runs = pd.read_csv(ROOT / "analysis" / "per_run_metrics.csv")
    tests = pd.read_csv(ROOT / "analysis" / "paired_tests.csv")
    for theme in THEMES:
        apply_theme(theme)
        fig_convergence(runs)
        fig_top1(runs, tests)
        fig_regret(runs, tests)
        fig_descriptor_space()
        fig_ml_benchmark()
        fig_parity()


if __name__ == "__main__":
    main()
