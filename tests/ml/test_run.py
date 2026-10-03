import json
import numpy as np
import pandas as pd

from ml_models.benchmark import data, run


def fake_data(tmp_path, monkeypatch, n=150):
    rng = np.random.default_rng(0)
    smi = [f"C{'C' * (i % 7)}O{i}" for i in range(n)]
    X = rng.normal(size=(n, 5)); y = 3 * X[:, 0] + rng.normal(0, 0.2, n)
    (tmp_path / "dft_G.json").write_text(json.dumps([{"SMILES": s, "energy": float(v)} for s, v in zip(smi, y)]))
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(5)]); df.insert(0, "SMILES", smi)
    for f in data.REPS.values():
        df.to_csv(tmp_path / f, index=False)
    monkeypatch.setattr(data, "DATA", tmp_path)
    folds = tmp_path / "folds.json"
    data.save_folds(data.make_folds(smi), folds)
    return folds


def test_run_cell_writes_outputs(tmp_path, monkeypatch):
    folds = fake_data(tmp_path, monkeypatch)
    out = tmp_path / "out"
    m = run.run_cell("rf", "dft_descriptors", 0, out, n_trials=3, job_budget_s=120, folds_path=folds)
    cell = out / "rf" / "dft_descriptors" / "fold0"
    assert m["status"] == "ok" and m["metrics"]["r2"] > 0.5
    for f in ["metrics.json", "predictions.csv", "best_params.json", "trials.csv"]:
        assert (cell / f).exists()
    assert set(m["data_hashes"]) == set(data.load_folds(folds)["data_hashes"])


def test_run_cell_records_failure(tmp_path, monkeypatch):
    folds = fake_data(tmp_path, monkeypatch)
    monkeypatch.setattr(run, "_fit_and_score", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    m = run.run_cell("rf", "dft_descriptors", 0, tmp_path / "out", n_trials=2, job_budget_s=60, folds_path=folds)
    saved = json.loads((tmp_path / "out/rf/dft_descriptors/fold0/metrics.json").read_text())
    assert m["status"] == saved["status"] == "failed" and "boom" in saved["error"]


def test_learning_curve_uses_nested_training_subsets(tmp_path, monkeypatch):
    folds = fake_data(tmp_path, monkeypatch)
    m = run.run_cell("ridge", "dft_descriptors", 1, tmp_path / "out", n_trials=2, job_budget_s=60,
                     learning_curve=True, folds_path=folds)
    lc = m["learning_curve"]
    assert [p["fraction"] for p in lc] == [0.1, 0.25, 0.5, 1.0]
    assert lc[-1]["n_train"] == 120 and all(p["n_train"] < 120 for p in lc[:-1])


def test_budget_reduction_skips_tuning_and_is_recorded(tmp_path, monkeypatch):
    folds = fake_data(tmp_path, monkeypatch)
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"reduced": {"ridge/dft_descriptors": {
        "params": {"alpha": 3.0}, "reason": "grid exceeds the CPU budget"}}}))
    monkeypatch.setattr(run, "BUDGET", budget)
    m = run.run_cell("ridge", "dft_descriptors", 0, tmp_path / "out", n_trials=5, job_budget_s=60, folds_path=folds)
    saved = json.loads((tmp_path / "out/ridge/dft_descriptors/fold0/best_params.json").read_text())
    assert m["status"] == "ok" and m["n_trials_done"] == 0 and saved == {"alpha": 3.0}
    assert m["reduced"] == "grid exceeds the CPU budget"
