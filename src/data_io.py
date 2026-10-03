import json
from pathlib import Path
import numpy as np
import pandas as pd

def load_descriptors(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'SMILES' not in df.columns:
        raise ValueError("Input CSV must have a 'SMILES' column")
    return df

def load_or_init_cache(path: Path) -> dict[str, float]:
    if path.exists():
        raw = json.loads(path.read_text())
        # if the file is a list of { "smiles":..., "energy":... } records
        if isinstance(raw, list):
            return { rec['SMILES']: rec['energy'] for rec in raw }
        # otherwise assume it's already a dict
        return raw
    return {}

def append_to_cache(smiles: str, energy: float, cache_path: Path):
    cache = load_or_init_cache(cache_path)
    cache[smiles] = energy
    cache_path.write_text(json.dumps(cache, indent=2))

def pool_min(smiles: list, cache: dict) -> float:
    """Lowest cached energy among `smiles` (inf if none are cached).
    BO and the baselines call this once, on the initial candidate pool."""
    vals = [cache[s] for s in smiles if s in cache]
    return float(min(vals)) if vals else float("inf")


def prune_columns(X: pd.DataFrame, threshold: float) -> list[str]:
    """Names of the columns to keep: constant columns are dropped, then columns are
    visited in their original order and kept only if |Pearson r| <= threshold with
    every column already kept. Uses X only (no targets)."""
    X = X.loc[:, X.to_numpy(dtype=float).std(axis=0) > 0]
    corr = np.abs(np.corrcoef(X.to_numpy(dtype=float), rowvar=False))
    kept: list[int] = []
    for j in range(X.shape[1]):
        if not kept or corr[j, kept].max() <= threshold:
            kept.append(j)
    return list(X.columns[kept])
