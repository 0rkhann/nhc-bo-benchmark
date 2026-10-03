import sys

import pandas as pd
import pytest

from src import cli
from src.pipelines.fabo_pipeline import FABOPipeline
from src.pipelines.opls_pipeline import OPLSPipeline


def _config(*extra, mode="opls"):
    args = cli.build_parser().parse_args(
        ["--mode", mode, "--input", "x.csv", "--cache", "c.json",
         "--n-initial", "10", "--n-iter", "1", "--seed", "42", *extra]
    )
    return cli.build_config(args, kernel="Matern", acq="EI", beta=1.0, seed=42)


def test_default_opls_has_orthogonal_components():
    """Bug 3: default --opls-n-components=1 gave max_orthogonal = min(3, 1 // 2) = 0."""
    sel = OPLSPipeline(_config()).selector_factory()
    assert sel.max_orthogonal >= 1
    assert sel.min_orthogonal >= 1
    assert sel.max_predictive >= 2


def test_explicit_single_opls_component_still_gets_orthogonal_component():
    sel = OPLSPipeline(_config("--opls-n-components", "1")).selector_factory()
    assert sel.max_orthogonal >= 1


def test_fabo_cli_flags_reach_the_selector():
    """Bug 7: fabo_k / fabo_threshold were put at the top level of the config."""
    sel = FABOPipeline(_config("--fabo-k", "7", mode="fabo")).selector_factory()
    assert sel.k == 7
    sel = FABOPipeline(_config("--fabo-threshold", "0.4", mode="fabo")).selector_factory()
    assert sel.threshold == 0.4


def test_fabo_default_is_auto_selection():
    sel = FABOPipeline(_config(mode="fabo")).selector_factory()
    assert sel.k is None and sel.threshold is None


def test_opls_scale_flag_is_honoured():
    assert OPLSPipeline(_config()).selector_factory().scale is True
    assert OPLSPipeline(_config("--no-opls-scale")).selector_factory().scale is False


@pytest.mark.parametrize("mode", ["vanilla", "random", "outlier"])
def test_control_methods_run_from_the_cli(synthetic, monkeypatch, mode):
    argv = ["cli", "--mode", mode, "--input", str(synthetic["csv"]),
            "--cache", str(synthetic["cache"]), "--output-dir", str(synthetic["results"]),
            "--n-initial", "10", "--n-iter", "2", "--seed", "42",
            "--kernels", "Matern", "--acquisitions", "EI"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    files = list(synthetic["results"].rglob("*history.csv"))
    assert len(files) == 1
    assert len(pd.read_csv(files[0])) == 2
