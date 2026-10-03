import json
from pathlib import Path

import numpy as np
import pytest

from ml_models.benchmark import summarize


def write_cell(root, model, rep, fold, r2, hashes=None, version="1.0", status="ok"):
    d = Path(root) / model / rep / f"fold{fold}"; d.mkdir(parents=True)
    m = {"r2": r2, "rmse": 1 - r2, "mae": 1 - r2, "spearman": r2, "top1_recall": r2, "tail_rmse": 1 - r2}
    (d / "metrics.json").write_text(json.dumps({
        "model": model, "rep": rep, "fold": fold, "status": status, "metrics": m, "wall_s": 60,
        "data_hashes": hashes or {"dft_G.json": "a"}, "versions": {"scikit-learn": version}}))
    return d


def test_missing_metrics_file_counts_as_missing_fold(tmp_path):      # Review Focus 2
    for k in range(4):
        write_cell(tmp_path, "rf", "dft_descriptors", k, 0.3)
    (tmp_path / "rf" / "dft_descriptors" / "fold4").mkdir()            # killed job, no metrics.json
    cells = summarize.collect(tmp_path)
    s = summarize.summary(cells)
    row = s[(s.model == "rf") & (s.rep == "dft_descriptors")].iloc[0]
    assert row.folds_ok == 4 and (cells.status == "missing").sum() == 1


def test_mixed_data_hashes_are_refused(tmp_path):
    write_cell(tmp_path, "rf", "dft_descriptors", 0, 0.3, hashes={"dft_G.json": "a"})
    write_cell(tmp_path, "rf", "dft_descriptors", 1, 0.3, hashes={"dft_G.json": "b"})
    with pytest.raises(ValueError, match="data"):
        summarize.check_provenance(summarize.collect(tmp_path))


def test_mixed_versions_are_refused(tmp_path):
    write_cell(tmp_path, "rf", "dft_descriptors", 0, 0.3, version="1.0")
    write_cell(tmp_path, "rf", "dft_descriptors", 1, 0.3, version="2.0")
    with pytest.raises(ValueError, match="version"):
        summarize.check_provenance(summarize.collect(tmp_path))


def test_comparisons_and_ceiling_rule(tmp_path):
    for k in range(5):
        write_cell(tmp_path, "rf", "dft_descriptors", k, 0.28 + 0.001 * k)
        write_cell(tmp_path, "tabpfn", "dft_descriptors", k, 0.40 + 0.002 * k)
    c = summarize.comparisons(summarize.collect(tmp_path))
    vs_rf = c[(c.family == "vs_rf") & (c.model == "tabpfn")].iloc[0]
    assert vs_rf.delta_r2 > 0.1
    assert bool(c[c.family == "ceiling"].iloc[0].ceiling_moved)


def test_merge_replaces_only_rerun_cells(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    write_cell(old, "rf", "dft_descriptors", 0, 0.1); write_cell(old, "xgb", "dft_descriptors", 0, 0.2)
    write_cell(new, "rf", "dft_descriptors", 0, 0.3)
    summarize.merge(new, old)
    cells = summarize.collect(old).set_index("model")
    assert cells.loc["rf", "r2"] == 0.3 and cells.loc["xgb", "r2"] == 0.2
