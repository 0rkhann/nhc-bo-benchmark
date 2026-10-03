# CPU budget decisions for the ML benchmark

Each benchmark job runs on a GitHub-hosted CPU runner (4 cores, 16 GB RAM) with a 340-minute limit. The spec's reduction rule:
- A cell's tuning grid plus its final refit must fit in 80% of that limit.
- If it does not, the cell runs its default configuration only, without inner tuning.
- If even that does not fit, it is reported as "not run".

Machine-readable decisions are in [`budget.json`](budget.json). `run.py` applies the `reduced` entries, and `plan_matrix.py` leaves the `not_run` cells out of the job matrix.

## Measurements

From the timing spike on 2026-10-02: workflow runs 36970456019, 36971902737 and 36972603924. Each entry is one inner fit and predict with the default configuration: 3,653 training rows and 1,827 predicted rows.

| Model | DFT descriptors | ChemBERTa-2 | Mordred |
|---|---|---|---|
| Random Forest | 2.9 s | 39.6 s | 104.4 s |
| XGBoost | 0.5 s | 8.4 s | 29.6 s |
| Exact GP | 755.3 s | 658.8 s | 701.6 s (523 columns after pruning) |
| TabPFN-3 (8 estimators) | 157.1 s | 417.5 s | 922.9 s |
| TabICLv2 (8 estimators, `batch_size=1`) | 52.2 s | 333.2 s | out of memory |

Chemprop: 244.8 s on SMILES only, and 479.5 s on SMILES plus the 29 DFT descriptors.

## Decisions

**TabPFN-3:** grid cost is about 18 × the 8-estimator fit (two configurations at 8 estimators and two at 16, each fitted 3 times), plus a refit.
- Descriptors (about 55 min) and ChemBERTa-2 (about 2.4 h) run the full grid.
- **Mordred is reduced to the default configuration** (8 estimators, temperature 0.9), because its grid plus refit is about 5.4 h, over the 4.5 h allowed.

**TabICLv2:** grid cost is about 21 × the 8-estimator fit (8, 16 and 32 estimators, each fitted 3 times).
- Descriptors (about 20 min) and ChemBERTa-2 (about 2.5 h) run the full grid.
- **Mordred is not run.** The runner ran out of memory at 1,197 columns with the default `batch_size=8`, with `batch_size=1`, and with disk offload.

**Exact GP:** a single fit at 5,480 rows is estimated at about 45 min, so it runs as planned.

**Chemprop, Random Forest on Mordred and XGBoost** run their full trial budgets, capped by wall time. Each cell's `trials.csv` and `n_trials_done` record how many trials finished.

## Refit reserve

`refit_estimate_s` in each adapter is set from these measurements, scaled to 5,480 rows and the largest configuration in its grid: TabPFN-3 1,500 s, TabICL 2,100 s, GP 2,700 s, Chemprop 2,700 s (raised from 1,500 s after the fold-2 timeouts), Random Forest 1,200 s, XGBoost 600 s. `run.py` keeps twice this time free for the final refit.

## Chemprop, SMILES plus DFT descriptors, fold 2

This fold hit the 350-minute job limit three times:
- in run 36973305220, before the tuning cap existed;
- in run 37045057608, with a 50-minute refit reserve;
- in run 37111375107, with a 90-minute refit reserve.

The other four folds finished. Under the spec's failure rule, the cell is reported with 4 of 5 folds and left out of the significance tests. Each fold completed only 2–4 Chemprop trials, so a GPU run is the way to finish this cell.
