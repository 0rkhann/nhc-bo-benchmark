420 runs; pool optimum per seed, by representation:
  dft_chemberta2: -41.83 to -40.29, top-1% threshold -20.36 to -19.97, pool size [6155]
  dft_descriptors: -41.83 to -36.31, top-1% threshold -20.41 to -19.92, pool size [6155]
  dft_mordred: -41.83 to -40.29, top-1% threshold -20.36 to -19.97, pool size [6155]

#### DFT descriptors

| Method | Final best (mean ± sd) | Hit pool optimum | Iter. to optimum (median) | Reached top 1% | Iter. to top 1% (median [IQR]) | Regret AUC (median) |
|---|---|---|---|---|---|---|
| fabo | -41.17 ± 1.33 | 16/20 | 41 | 20/20 | 6.5 [2–15] | 0.174 |
| pca | -40.94 ± 2.98 | 19/20 | 23 | 20/20 | 8.0 [2–11] | 0.156 |
| pls | -41.03 ± 2.60 | 19/20 | 21 | 20/20 | 6.0 [4–9] | 0.138 |
| opls | -41.48 ± 1.26 | 20/20 | 16 | 20/20 | 4.0 [1–7] | 0.103 |
| vanilla | -35.11 ± 9.91 | 14/20 | 77 | 18/20 | 26.0 [11–55] | 0.577 |
| random | -22.71 ± 4.59 | 0/20 | 100 | 14/20 | 51.0 [12–100] | 0.870 |
| outlier | -41.48 ± 1.26 | 20/20 | 28 | 20/20 | 27.0 [26–28] | 0.237 |

#### ChemBERTa-2

| Method | Final best (mean ± sd) | Hit pool optimum | Iter. to optimum (median) | Reached top 1% | Iter. to top 1% (median [IQR]) | Regret AUC (median) |
|---|---|---|---|---|---|---|
| fabo | -23.36 ± 1.16 | 0/20 | 100 | 20/20 | 24.0 [8–41] | 0.748 |
| pca | -23.53 ± 1.88 | 0/20 | 100 | 19/20 | 15.5 [4–40] | 0.713 |
| pls | -25.34 ± 3.80 | 0/20 | 100 | 20/20 | 18.0 [7–22] | 0.722 |
| opls | -24.83 ± 3.78 | 0/20 | 100 | 20/20 | 25.0 [11–38] | 0.737 |
| vanilla | -25.42 ± 8.31 | 3/20 | 100 | 12/20 | 57.5 [17–100] | 0.771 |
| random | -22.30 ± 4.94 | 0/20 | 100 | 16/20 | 30.5 [11–85] | 0.826 |
| outlier | -21.59 ± 0.86 | 0/20 | 100 | 19/20 | 7.0 [7–8] | 0.794 |

#### Mordred

| Method | Final best (mean ± sd) | Hit pool optimum | Iter. to optimum (median) | Reached top 1% | Iter. to top 1% (median [IQR]) | Regret AUC (median) |
|---|---|---|---|---|---|---|
| fabo | -24.51 ± 4.62 | 0/20 | 100 | 18/20 | 22.0 [9–53] | 0.776 |
| pca | -23.19 ± 5.33 | 1/20 | 100 | 17/20 | 63.0 [26–86] | 0.806 |
| pls | -27.33 ± 6.03 | 2/20 | 100 | 19/20 | 30.5 [9–41] | 0.698 |
| opls | -30.09 ± 5.39 | 2/20 | 100 | 20/20 | 31.0 [10–42] | 0.667 |
| vanilla | -26.21 ± 8.17 | 4/20 | 100 | 16/20 | 26.5 [11–76] | 0.772 |
| random | -22.30 ± 4.94 | 0/20 | 100 | 16/20 | 30.5 [11–85] | 0.826 |
| outlier | -20.23 ± 1.01 | 0/20 | 100 | 18/20 | 20.0 [18–21] | 0.831 |

#### iterations to top 1%: BO method vs `random` (paired by seed; negative = method better)

| Representation | Method | Median diff | Rank-biserial r | W / L / T | p | p (Holm) |
|---|---|---|---|---|---|---|
| DFT descriptors | fabo | -41 | -0.97 | 15/1/4 | 0.000642 | 0.00834 |
| DFT descriptors | pca | -36.5 | -0.92 | 15/2/3 | 0.000844 | 0.0101 |
| DFT descriptors | pls | -47 | -1.00 | 17/0/3 | 0.000291 | 0.00436 |
| DFT descriptors | opls | -47 | -0.97 | 16/1/3 | 0.00042 | 0.00588 |
| DFT descriptors | vanilla | +0 | -0.48 | 9/8/3 | 0.084 | 0.84 |
| ChemBERTa-2 | fabo | -7.5 | -0.46 | 11/7/2 | 0.0852 | 0.84 |
| ChemBERTa-2 | pca | +0 | -0.36 | 9/8/3 | 0.193 | 1 |
| ChemBERTa-2 | pls | -14.5 | -0.70 | 13/5/2 | 0.00956 | 0.105 |
| ChemBERTa-2 | opls | -3 | -0.29 | 10/8/2 | 0.286 | 1 |
| ChemBERTa-2 | vanilla | +0 | +0.26 | 8/8/4 | 0.352 | 1 |
| Mordred | fabo | -4.5 | -0.18 | 11/7/2 | 0.514 | 1 |
| Mordred | pca | +2 | +0.29 | 7/10/3 | 0.287 | 1 |
| Mordred | pls | -11 | -0.41 | 10/7/3 | 0.136 | 1 |
| Mordred | opls | -8 | -0.36 | 12/6/2 | 0.184 | 1 |
| Mordred | vanilla | +0 | -0.05 | 8/9/3 | 0.868 | 1 |

#### iterations to top 1%: BO method vs `outlier` (paired by seed; negative = method better)

| Representation | Method | Median diff | Rank-biserial r | W / L / T | p | p (Holm) |
|---|---|---|---|---|---|---|
| DFT descriptors | fabo | -15 | -1.00 | 17/0/3 | 0.000291 | 0.00428 |
| DFT descriptors | pca | -19 | -0.98 | 16/1/3 | 0.000375 | 0.00428 |
| DFT descriptors | pls | -19 | -1.00 | 17/0/3 | 0.00029 | 0.00428 |
| DFT descriptors | opls | -22 | -1.00 | 17/0/3 | 0.000286 | 0.00428 |
| DFT descriptors | vanilla | +1.5 | +0.42 | 7/10/3 | 0.124 | 0.619 |
| ChemBERTa-2 | fabo | +10 | +0.69 | 3/14/3 | 0.0129 | 0.103 |
| ChemBERTa-2 | pca | +4.5 | +0.79 | 4/12/4 | 0.0052 | 0.052 |
| ChemBERTa-2 | pls | +6.5 | +0.58 | 4/14/2 | 0.031 | 0.217 |
| ChemBERTa-2 | opls | +14 | +0.76 | 2/15/3 | 0.00558 | 0.052 |
| ChemBERTa-2 | vanilla | +42.5 | +0.99 | 1/16/3 | 0.000346 | 0.00428 |
| Mordred | fabo | +0 | +0.23 | 9/9/2 | 0.384 | 1 |
| Mordred | pca | +33.5 | +0.58 | 3/15/2 | 0.031 | 0.217 |
| Mordred | pls | +5 | +0.19 | 8/10/2 | 0.486 | 1 |
| Mordred | opls | +5.5 | +0.23 | 7/11/2 | 0.396 | 1 |
| Mordred | vanilla | -1.5 | +0.22 | 10/7/3 | 0.421 | 1 |

#### regret AUC: BO method vs `random` (paired by seed; negative = method better)

| Representation | Method | Median diff | Rank-biserial r | W / L / T | p | p (Holm) |
|---|---|---|---|---|---|---|
| DFT descriptors | fabo | -0.674 | -1.00 | 20/0/0 | 1.91e-06 | 2.86e-05 |
| DFT descriptors | pca | -0.694 | -1.00 | 20/0/0 | 1.91e-06 | 2.86e-05 |
| DFT descriptors | pls | -0.69 | -1.00 | 20/0/0 | 1.91e-06 | 2.86e-05 |
| DFT descriptors | opls | -0.739 | -1.00 | 20/0/0 | 1.91e-06 | 2.86e-05 |
| DFT descriptors | vanilla | -0.259 | -0.85 | 17/3/0 | 0.000322 | 0.00355 |
| ChemBERTa-2 | fabo | -0.0723 | -0.44 | 14/6/0 | 0.0897 | 0.448 |
| ChemBERTa-2 | pca | -0.0791 | -0.55 | 15/5/0 | 0.0296 | 0.177 |
| ChemBERTa-2 | pls | -0.109 | -0.65 | 17/3/0 | 0.00944 | 0.0849 |
| ChemBERTa-2 | opls | -0.0868 | -0.59 | 16/4/0 | 0.0192 | 0.135 |
| ChemBERTa-2 | vanilla | +0.000634 | -0.04 | 8/11/1 | 0.872 | 1 |
| Mordred | fabo | -0.0651 | -0.35 | 14/6/0 | 0.177 | 0.707 |
| Mordred | pca | +0.00131 | +0.12 | 9/10/1 | 0.658 | 1 |
| Mordred | pls | -0.125 | -0.63 | 17/3/0 | 0.0121 | 0.0966 |
| Mordred | opls | -0.135 | -0.73 | 17/3/0 | 0.00271 | 0.0271 |
| Mordred | vanilla | -0.0588 | -0.27 | 13/7/0 | 0.312 | 0.935 |

#### regret AUC: BO method vs `outlier` (paired by seed; negative = method better)

| Representation | Method | Median diff | Rank-biserial r | W / L / T | p | p (Holm) |
|---|---|---|---|---|---|---|
| DFT descriptors | fabo | -0.0546 | -0.48 | 13/7/0 | 0.0637 | 0.191 |
| DFT descriptors | pca | -0.107 | -0.77 | 17/3/0 | 0.00143 | 0.00859 |
| DFT descriptors | pls | -0.107 | -0.79 | 17/3/0 | 0.00102 | 0.00712 |
| DFT descriptors | opls | -0.14 | -0.96 | 19/1/0 | 1.34e-05 | 0.000174 |
| DFT descriptors | vanilla | +0.286 | +0.89 | 3/17/0 | 0.000134 | 0.00147 |
| ChemBERTa-2 | fabo | -0.0481 | -0.77 | 16/4/0 | 0.00143 | 0.00859 |
| ChemBERTa-2 | pca | -0.0683 | -0.87 | 16/4/0 | 0.00021 | 0.00189 |
| ChemBERTa-2 | pls | -0.0875 | -0.97 | 19/1/0 | 9.54e-06 | 0.000134 |
| ChemBERTa-2 | opls | -0.0724 | -0.86 | 16/4/0 | 0.000261 | 0.00209 |
| ChemBERTa-2 | vanilla | -0.00799 | -0.17 | 11/9/0 | 0.522 | 0.709 |
| Mordred | fabo | -0.0445 | -0.88 | 18/2/0 | 0.000168 | 0.00168 |
| Mordred | pca | -0.00225 | -0.24 | 12/7/1 | 0.355 | 0.709 |
| Mordred | pls | -0.169 | -0.91 | 19/1/0 | 6.29e-05 | 0.000755 |
| Mordred | opls | -0.17 | -0.98 | 19/1/0 | 5.72e-06 | 8.58e-05 |
| Mordred | vanilla | -0.0811 | -0.73 | 17/3/0 | 0.00271 | 0.0108 |

#### ML benchmark (5-fold nested CV, mean ± sd)

| Representation | Model | Folds | R² | RMSE | Spearman | Top-1% recall |
|---|---|---|---|---|---|---|
| dft_chemberta2 | gp | 5/5 | 0.142 ± 0.006 | 4.35 ± 0.14 | 0.410 ± 0.013 | 0.23 ± 0.09 |
| dft_chemberta2 | mean | 5/5 | -0.000 ± 0.001 | 4.70 ± 0.15 | — | 0.04 ± 0.06 |
| dft_chemberta2 | outlier | 5/5 | — | — | 0.060 ± 0.025 | 0.07 ± 0.07 |
| dft_chemberta2 | rf | 5/5 | 0.200 ± 0.016 | 4.20 ± 0.15 | 0.474 ± 0.009 | 0.31 ± 0.06 |
| dft_chemberta2 | ridge | 5/5 | 0.265 ± 0.021 | 4.03 ± 0.15 | 0.544 ± 0.021 | 0.30 ± 0.09 |
| dft_chemberta2 | tabicl | 5/5 | 0.299 ± 0.027 | 3.94 ± 0.17 | 0.580 ± 0.016 | 0.27 ± 0.09 |
| dft_chemberta2 | tabpfn | 5/5 | 0.310 ± 0.027 | 3.90 ± 0.16 | 0.590 ± 0.017 | 0.33 ± 0.06 |
| dft_chemberta2 | xgb | 5/5 | 0.205 ± 0.016 | 4.19 ± 0.16 | 0.478 ± 0.010 | 0.24 ± 0.06 |
| dft_descriptors | gp | 5/5 | 0.251 ± 0.035 | 4.07 ± 0.16 | 0.493 ± 0.026 | 0.46 ± 0.06 |
| dft_descriptors | mean | 5/5 | -0.000 ± 0.001 | 4.70 ± 0.15 | — | 0.04 ± 0.06 |
| dft_descriptors | outlier | 5/5 | — | — | 0.226 ± 0.030 | 0.44 ± 0.13 |
| dft_descriptors | rf | 5/5 | 0.267 ± 0.026 | 4.02 ± 0.15 | 0.509 ± 0.022 | 0.51 ± 0.12 |
| dft_descriptors | ridge | 5/5 | 0.203 ± 0.023 | 4.19 ± 0.14 | 0.427 ± 0.019 | 0.54 ± 0.08 |
| dft_descriptors | tabicl | 5/5 | 0.311 ± 0.019 | 3.90 ± 0.15 | 0.553 ± 0.017 | 0.56 ± 0.09 |
| dft_descriptors | tabpfn | 5/5 | 0.317 ± 0.024 | 3.89 ± 0.15 | 0.557 ± 0.018 | 0.60 ± 0.08 |
| dft_descriptors | xgb | 5/5 | 0.270 ± 0.028 | 4.02 ± 0.16 | 0.514 ± 0.022 | 0.56 ± 0.14 |
| dft_mordred | gp | 5/5 | 0.109 ± 0.007 | 4.44 ± 0.13 | 0.424 ± 0.021 | 0.24 ± 0.13 |
| dft_mordred | mean | 5/5 | -0.000 ± 0.001 | 4.70 ± 0.15 | — | 0.04 ± 0.06 |
| dft_mordred | outlier | 5/5 | — | — | 0.018 ± 0.041 | 0.11 ± 0.10 |
| dft_mordred | rf | 5/5 | 0.275 ± 0.025 | 4.00 ± 0.16 | 0.559 ± 0.017 | 0.26 ± 0.15 |
| dft_mordred | ridge | 5/5 | 0.290 ± 0.020 | 3.96 ± 0.16 | 0.572 ± 0.016 | 0.34 ± 0.11 |
| dft_mordred | tabpfn | 5/5 | 0.325 ± 0.024 | 3.86 ± 0.15 | 0.607 ± 0.018 | 0.33 ± 0.10 |
| dft_mordred | xgb | 5/5 | 0.281 ± 0.021 | 3.99 ± 0.16 | 0.567 ± 0.014 | 0.27 ± 0.15 |
| smiles | chemprop | 5/5 | 0.280 ± 0.017 | 3.99 ± 0.15 | 0.575 ± 0.020 | 0.36 ± 0.07 |

#### Facts

- Random-search histories identical (SMILES and energies) between ChemBERTa-2 and Mordred: 20/20 seeds
- Random-search histories identical between ChemBERTa-2 and descriptors: 0/20 seeds
- Outlier heuristic on descriptors: pool size 6155; median iteration reaching the pool optimum 28 (= top 0.45% of the pool by outlier score)
- Global energy optimum -41.8341: rank 30 of 6850 by max|z| (5.7 sigma on 'f+')
- Ten lowest-energy molecules: outlier-score rank (of 6850) = [30, 39, 43, 48, 96, 106, 113, 170, 204, 229]
- Spearman correlation between max|z| and energy over all molecules: -0.226
