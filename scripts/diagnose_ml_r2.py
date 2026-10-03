"""Why is hold-out R^2 low? Checks feature/target alignment (shuffled-target control), model capacity,
the learning curve and duplicated feature rows. Run from the repository root:
python scripts/diagnose_ml_r2.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

E = {r["SMILES"]: r["energy"] for r in json.load(open("data/dft_G.json"))}
y_all = np.array(list(E.values()))
print(f"target: n={len(y_all)} mean={y_all.mean():.2f} sd={y_all.std():.2f} skew={stats.skew(y_all):.2f} "
      f"kurtosis={stats.kurtosis(y_all):.1f} q01={np.quantile(y_all,.01):.2f} q99={np.quantile(y_all,.99):.2f}")

def load(name):
    df = pd.read_csv(f"data/{name}.csv")
    X = df.drop(columns="SMILES").apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    y = df["SMILES"].map(E).to_numpy(float)
    assert not np.isnan(y).any()
    return df["SMILES"].to_numpy(), X, y

cv = KFold(5, shuffle=True, random_state=0)
for name in ["dft_descriptors", "dft_mordred", "dft_chemberta2"]:
    smi, X, y = load(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_repo = RandomForestRegressor(200, max_features="sqrt", n_jobs=-1, random_state=42).fit(Xtr, ytr)
    rf_full = RandomForestRegressor(300, max_features=0.5, min_samples_leaf=2, n_jobs=-1, random_state=42).fit(Xtr, ytr)
    hgb = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, random_state=42).fit(Xtr, ytr)
    print(f"\n[{name}] X={X.shape}")
    for lab, m in [("RF as in repo", rf_repo), ("RF max_features=0.5", rf_full), ("HistGB", hgb)]:
        p = m.predict(Xte)
        print(f"  {lab:20s} hold-out R2={r2_score(yte,p):.3f}  Spearman={stats.spearmanr(yte,p)[0]:.3f}")
    # shuffled-target control: R2 should be ~0 if pipeline is sound
    yp = np.random.default_rng(0).permutation(ytr)
    print(f"  shuffled-y control   hold-out R2={r2_score(yte, RandomForestRegressor(200, n_jobs=-1, random_state=0).fit(Xtr, yp).predict(Xte)):.3f}")
    # learning curve
    lc = []
    for f in [0.1, 0.25, 0.5, 1.0]:
        n = int(len(ytr) * f)
        m = RandomForestRegressor(300, max_features=0.5, min_samples_leaf=2, n_jobs=-1, random_state=42).fit(Xtr[:n], ytr[:n])
        lc.append(f"{n}:{r2_score(yte, m.predict(Xte)):.3f}")
    print("  learning curve (n_train:R2)", " ".join(lc))
    # label-noise floor: molecules with identical feature vectors but different energies
    keys = pd.Series([hash(r.tobytes()) for r in np.round(X, 8)])
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"]
    sizes = g.size(); dup = sizes[sizes > 1]
    if len(dup):
        within = g.transform(lambda s: s - s.mean())[keys.isin(dup.index).to_numpy()]
        n_dup = int(dup.sum())
        print(f"  identical feature rows: {len(dup)} groups covering {n_dup} molecules; "
              f"within-group energy sd={within.std():.2f} (target sd {y.std():.2f})")
    else:
        print("  identical feature rows: none")
