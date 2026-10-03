import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

N_ROWS, N_INITIAL, SEED = 80, 10, 42


@pytest.fixture
def synthetic(tmp_path):
    """Small descriptor table + energy cache. Column f1 sits near -2500 (like TotalE),
    f2 is heavy tailed, so train-only scaling leaves pool points outside [0, 1]."""
    rng = np.random.default_rng(0)
    cols = {
        "f0": rng.normal(size=N_ROWS),
        "f1": -2500 + 5 * rng.normal(size=N_ROWS),
        "f2": rng.exponential(10, size=N_ROWS),
    }
    for j in range(3, 8):
        cols[f"f{j}"] = rng.normal(size=N_ROWS)
    df = pd.DataFrame(cols)
    y = -3 * df["f0"] + 0.2 * (df["f1"] + 2500) + 0.1 * df["f2"] + rng.normal(0, 0.1, N_ROWS)
    df.insert(0, "SMILES", [f"M{i}" for i in range(N_ROWS)])
    csv = tmp_path / "desc.csv"
    df.to_csv(csv, index=False)
    cache = tmp_path / "cache.json"
    energy = dict(zip(df.SMILES, (float(v) for v in y)))
    cache.write_text(json.dumps(energy))
    return {
        "csv": csv,
        "cache": cache,
        "df": df,
        "energy": energy,
        "results": tmp_path / "out",
    }


def split_like_pipeline(df, seed=SEED, n_initial=N_INITIAL, test_frac=0.1):
    """(init, pool, test) SMILES lists, same calls as the pipelines make."""
    smiles = df["SMILES"].tolist()
    X = df.drop(columns="SMILES")
    X_tmp, _, smi_tmp, smi_test = train_test_split(
        X, smiles, test_size=test_frac, random_state=seed, shuffle=True
    )
    _, _, smi_init, smi_pool = train_test_split(
        X_tmp, smi_tmp, train_size=n_initial, random_state=seed, shuffle=True
    )
    return smi_init, smi_pool, smi_test


def bo_config(results_dir, n_iter=3, seed=SEED, **fs):
    return {
        "data": {"n_initial": N_INITIAL, "test_frac": 0.1},
        "optimization": {
            "n_iter": n_iter,
            "kernel": "Matern",
            "acquisition": "EI",
            "beta": 1.0,
            "batch_size": 1,
        },
        "training": {"maxiter": 10, "n_restarts": 1},
        "random_state": seed,
        "results_dir": str(results_dir),
        "feature_selection": fs,
    }
