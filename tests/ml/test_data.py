import json
import numpy as np
import pandas as pd
import pytest

from ml_models.benchmark import data


def test_folds_are_deterministic_disjoint_and_complete():
    smi = [f"S{i}" for i in range(103)]
    a, b = data.make_folds(smi), data.make_folds(smi)
    assert a == b
    flat = [s for f in a for s in f]
    assert sorted(flat) == sorted(smi) and len(flat) == len(set(flat))
    assert len(a) == 5


def test_split_maps_by_smiles_not_row_order():
    smi = [f"S{i}" for i in range(50)]
    folds = data.make_folds(smi)
    shuffled = list(reversed(smi))
    tr, te = data.split(folds, 0, shuffled)
    assert {shuffled[i] for i in te} == set(folds[0])
    assert set(tr).isdisjoint(te)


def test_load_rep_fails_on_missing_smiles(tmp_path, monkeypatch):
    pd.DataFrame({"SMILES": ["A", "B"], "f": [1.0, 2.0]}).to_csv(tmp_path / "x.csv", index=False)
    monkeypatch.setattr(data, "DATA", tmp_path)
    monkeypatch.setitem(data.REPS, "toy", "x.csv")
    with pytest.raises(ValueError, match="1 SMILES"):
        data.load_rep("toy", ["A", "B", "C"])


def test_real_data_facts():
    y = data.load_target()
    assert len(y) == 6850 and y.index.is_unique and not y.isna().any()
    for rep in data.REPS:
        X = data.load_rep(rep, list(y.index))
        assert X.shape[0] == 6850 and np.isfinite(X).all()


def test_all_smiles_parse_in_rdkit():
    Chem = pytest.importorskip("rdkit.Chem")
    bad = [s for s in data.load_target().index if Chem.MolFromSmiles(s) is None]
    assert bad == []


def test_folds_file_records_data_hashes(tmp_path):
    folds = data.make_folds([f"S{i}" for i in range(20)])
    data.save_folds(folds, tmp_path / "folds.json")
    saved = data.load_folds(tmp_path / "folds.json")
    assert saved["folds"] == folds
    assert set(saved["data_hashes"]) == {"dft_G.json", *data.REPS.values()}
