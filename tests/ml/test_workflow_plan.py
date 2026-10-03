# tests/ml/test_workflow_plan.py
import pytest

from ml_models.benchmark import plan_matrix


@pytest.fixture(autouse=True)
def no_budget(tmp_path, monkeypatch):
    """Planner tests use an empty budget; the real budget.json is checked separately."""
    monkeypatch.setattr(plan_matrix, "BUDGET", tmp_path / "missing.json")


def test_real_budget_skips_exactly_its_not_run_cells(monkeypatch):
    import json
    from pathlib import Path
    real = Path(plan_matrix.__file__).with_name("budget.json")
    monkeypatch.setattr(plan_matrix, "BUDGET", real)
    skipped = json.loads(real.read_text())["not_run"]
    jobs = plan_matrix.plan("benchmark", "mean_ridge rf xgb gp tabpfn tabicl outlier chemprop",
                            "dft_descriptors dft_chemberta2 dft_mordred", "0 1 2 3 4")
    assert len(jobs) == 110 - 5 * len(skipped)


def test_default_benchmark_matrix_has_110_jobs():
    jobs = plan_matrix.plan("benchmark", "mean_ridge rf xgb gp tabpfn tabicl outlier chemprop",
                            "dft_descriptors dft_chemberta2 dft_mordred", "0 1 2 3 4")
    assert len(jobs) == (7 * 3 + 1) * 5      # 7 tabular job types (incl. the outlier ranker) x 3 reps + Chemprop on SMILES
    assert {"model": "chemprop", "rep": "smiles", "fold": 0} in jobs


def test_timing_mode_runs_one_job_per_model_and_rep():
    jobs = plan_matrix.plan("timing", "tabpfn tabicl", "dft_mordred", "0 1 2 3 4")
    assert jobs == [{"model": "tabpfn", "rep": "dft_mordred", "fold": -1},
                    {"model": "tabicl", "rep": "dft_mordred", "fold": -1}]


def test_bad_names_are_rejected():
    import pytest
    with pytest.raises(SystemExit):
        plan_matrix.plan("benchmark", "rf; rm -rf /", "dft_descriptors", "0")


def test_not_run_cells_are_left_out(tmp_path, monkeypatch):
    import json
    b = tmp_path / "budget.json"
    b.write_text(json.dumps({"not_run": {"tabicl/dft_mordred": "out of memory"}}))
    monkeypatch.setattr(plan_matrix, "BUDGET", b)
    jobs = plan_matrix.plan("benchmark", "tabicl rf", "dft_descriptors dft_mordred", "0 1")
    assert {"model": "tabicl", "rep": "dft_mordred", "fold": 0} not in jobs
    assert {"model": "rf", "rep": "dft_mordred", "fold": 0} in jobs and len(jobs) == 6
