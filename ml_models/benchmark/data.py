"""Targets, representations and the shared outer folds, all keyed by SMILES."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPS = {"dft_descriptors": "dft_descriptors.csv", "dft_chemberta2": "dft_chemberta2.csv",
        "dft_mordred": "dft_mordred.csv"}


def load_target(path: Path | None = None) -> pd.Series:
    rows = json.loads((path or DATA / "dft_G.json").read_text())
    y = pd.Series({r["SMILES"]: float(r["energy"]) for r in rows})
    if not y.index.is_unique or y.isna().any():
        raise ValueError("dft_G.json has duplicate SMILES or missing energies")
    return y


def load_rep(rep: str, smiles: list[str]) -> np.ndarray:
    df = pd.read_csv(DATA / REPS[rep]).set_index("SMILES")
    missing = [s for s in smiles if s not in df.index]
    if missing:
        raise ValueError(f"{rep}: {len(missing)} SMILES from the folds are missing from {REPS[rep]}")
    X = df.loc[smiles].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError(f"{rep}: non-finite feature values")
    return X


def make_folds(smiles: list[str], k: int = 5, seed: int = 0) -> list[list[str]]:
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [[smiles[i] for i in test] for _, test in kf.split(smiles)]


def split(folds: list[list[str]], k: int, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    test = set(folds[k])
    is_test = np.array([s in test for s in smiles])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_hashes() -> dict[str, str]:
    files = ["dft_G.json", *REPS.values()]
    return {f: sha256(DATA / f) for f in files}


def save_folds(folds: list[list[str]], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"folds": folds, "data_hashes": data_hashes()}, indent=1))


def load_folds(path: Path) -> dict:
    return json.loads(Path(path).read_text())
