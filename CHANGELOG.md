# Changelog

## v1.0.2

### Fixed
- The README showed TabPFN-3's gains over Random Forest as +0.050 on both Mordred and DFT descriptors, so the ceiling rule looked inconsistent. They are now shown at four decimals: +0.0504 and +0.0498.

## v1.0.1

### Added
- Expected result for the unfinished Chemprop SMILES+DFT-descriptor fold 2 ([`scripts/project_chemprop_fold2.py`](scripts/project_chemprop_fold2.py)). The README reports it as a projection: a 5-fold R² of 0.388 (range 0.384–0.391) and a top-1% recall of 0.60.

## v1.0.0

Every BO and ML result changed in this release. The numbers in the internship report correspond to [`v0.1.0-internship`](https://github.com/0rkhann/nhc-bo-benchmark/tree/v0.1.0-internship).

### Fixed (BO pipeline)
- Feature selectors were fitted on scaled inputs but applied to raw inputs from the first iteration on.
- FABO, PCA, PLS and OPLS were fitted once on the 10 initial molecules and never refitted.
- "OPLS" ran as one-component PLS with no orthogonal component.
- The first BO pick was wasted (`best_f = inf`), and the initial design was ignored when tracking the best value.
- The MinMax scaler was fitted on the 10 initial molecules only.
- FABO command-line options were ignored.
- "Seeds at pool optimum" used one seed's optimum for all seeds.
- Scoring the whole pool in one batch made memory grow with every iteration.

### Changed
- **BO results:**
  - 20 seeds instead of 5;
  - two control methods (vanilla GP and a model-free outlier heuristic);
  - paired Wilcoxon tests with Holm correction.
- **ML results:** a nested 5-fold benchmark with tuned models replaces the single 80/20 split. It adds Ridge, the BO loop's GP, TabPFN-3, TabICLv2, Chemprop, tail metrics and corrected t-tests.
- **Energies:** labelled as xTB binding free energies in kJ/mol. Only the 29 descriptors are DFT-level. The 6,850 molecules are described as a benchmark subset of a larger library.
- **Energy backend:** the in-house workflow is no longer imported by the repository; new molecules need an energy backend set with `BO_ENERGY_BACKEND`.

### Removed
- **Old ML code:** the single-split models (`ml_models/train_models.py`, `random_forest.py`, `xgboost_model.py`, `base_model.py`, `plot_model_results.py`), their results and `train_all_ml_models.sh`.
- **Script:** `scripts/recompute_holdout_metrics.py`.
