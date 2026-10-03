from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_old_single_split_ml_code_is_gone():
    for p in ["ml_models/train_models.py", "ml_models/random_forest.py", "ml_models/xgboost_model.py",
              "ml_models/base_model.py", "ml_models/plot_model_results.py", "scripts/recompute_holdout_metrics.py",
              "train_all_ml_models.sh", "ml_plots/ml_model_summary.csv", "ml_results/dft"]:
        assert not (ROOT / p).exists(), p


def test_analysis_scripts_read_the_new_summary():
    for p in ["scripts/analyze_results.py", "scripts/summarize_results.py", "plotting/make_figures.py"]:
        src = (ROOT / p).read_text()
        assert "ml_results/benchmark" in src or '"benchmark"' in src, p
        assert "model_comparison.csv" not in src and "recompute_holdout" not in src, p
