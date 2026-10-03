import sys

import numpy as np
import pandas as pd

from src import cli
from src.data_io import prune_columns


def test_drops_constant_and_correlated_columns_in_original_order():
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    X = pd.DataFrame({
        "const": np.ones(50),
        "a": a,
        "a_copy": 2 * a + 1,        # |r| = 1 with a, visited later -> dropped
        "neg_a": -a,                # |r| = 1 -> dropped
        "b": rng.normal(size=50),
    })
    assert prune_columns(X, 0.95) == ["a", "b"]


def test_keeps_column_whose_correlation_equals_threshold():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    X = pd.DataFrame({"x": x, "y": x})
    assert prune_columns(X, 1.0) == ["x", "y"]  # |r| > threshold is required to drop


def test_flag_defaults_to_off():
    args = cli.build_parser().parse_args(
        ["--mode", "vanilla", "--input", "x", "--cache", "c",
         "--n-initial", "10", "--n-iter", "1", "--seed", "1"])
    assert args.prune_correlated is None
    assert cli.build_config(args, "Matern", "EI", 1.0, 1)["data"]["prune_correlated"] is None


def _run(synthetic, monkeypatch, *extra):
    argv = ["cli", "--mode", "vanilla", "--input", str(synthetic["csv"]),
            "--cache", str(synthetic["cache"]), "--output-dir", str(synthetic["results"]),
            "--n-initial", "10", "--n-iter", "2", "--seed", "42",
            "--kernels", "Matern", "--acquisitions", "EI", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    return next(synthetic["results"].rglob("bo_iteration_history.csv"))


def test_cli_prunes_and_writes_column_list(synthetic, monkeypatch):
    df = synthetic["df"].copy()
    df["dup"] = df["f0"] * 3          # correlated with f0
    df["flat"] = 7.0                  # constant
    df.to_csv(synthetic["csv"], index=False)
    hist = _run(synthetic, monkeypatch, "--prune-correlated", "0.95")
    kept = (hist.parent / "pruned_columns.txt").read_text().split()
    assert "dup" not in kept and "flat" not in kept and "f0" in kept
    assert len(kept) == df.shape[1] - 1 - 2
    assert len(pd.read_csv(hist)["l"].map(eval)[0]) == len(kept)


def test_no_pruning_without_flag(synthetic, monkeypatch):
    hist = _run(synthetic, monkeypatch)
    assert not (hist.parent / "pruned_columns.txt").exists()
