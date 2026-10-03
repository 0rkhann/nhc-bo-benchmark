"""
Random search baseline implementation.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.data_io import append_to_cache, load_descriptors, load_or_init_cache, pool_min
from src.utils import logger, seed_everything


class RandomSearchPipeline:
    name = "random_search"  # results directory suffix / history file stem
    best_col = "best_random"

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.logger = logger

    def _pool_scores(self, X_full_df: pd.DataFrame, pool_raw_df: pd.DataFrame):
        """Per-pool-row score used by `_pick`; random search needs none."""
        return None

    def _pick(self, scores) -> int:
        """Index into the current pool of the molecule to evaluate next."""
        return int(np.random.randint(len(scores)))

    def run(self, input_csv: str, cache_path: Path, simulator):
        # 0) lock down all RNGs
        seed_everything(self.cfg["random_state"])

        # 1) load full descriptor set
        df = load_descriptors(Path(input_csv))
        smiles_all = df["SMILES"].tolist()
        X_full_df = df.drop(columns="SMILES")

        # 2) carve off test set
        X_tmp, X_test_df, smi_tmp, smi_test = train_test_split(
            X_full_df,
            smiles_all,
            test_size=self.cfg.get("test_frac", 0.1),
            random_state=self.cfg["random_state"],
            shuffle=True,
        )

        # 3) carve off n_initial for starting design
        X_init_df, X_pool_df, smi_init, smi_pool = train_test_split(
            X_tmp,
            smi_tmp,
            train_size=self.cfg["n_initial"],
            random_state=self.cfg["random_state"],
            shuffle=True,
        )

        # 4) reset indices
        train_raw_df = X_init_df.reset_index(drop=True)
        pool_raw_df = X_pool_df.reset_index(drop=True)
        test_raw_df = X_test_df.reset_index(drop=True)
        smi_train = smi_init.copy()

        # 5) fetch / simulate y for initial
        cache = load_or_init_cache(cache_path)
        train_y_raw = []
        for k, smi in enumerate(smi_train, start=1):
            if smi in cache:
                y_val = cache[smi]
                self.logger.info(f"Cache hit for initial sample: {smi}")
            else:
                self.logger.info(f"Computing Binding Free Energy for: {smi}")
                y_val = simulator.compute(smi, index=k, total=len(smi_train))
                append_to_cache(smi, float(y_val), cache_path)
                cache[smi] = y_val
            train_y_raw.append(float(y_val))

        print(f"Size of initial training set: {len(train_y_raw)}")
        print(f"Size of pool set: {len(smi_pool)}")
        print(f"Size of test set: {len(smi_test)}")

        # track best values; best_pool_min is fixed at the initial candidate pool,
        # exactly as in BOPipeline
        best_random = min(train_y_raw)
        best_pool_min = pool_min(smi_pool, cache)
        scores = self._pool_scores(X_full_df, pool_raw_df)
        if scores is None:
            scores = np.zeros(len(smi_pool))

        records = []

        # Random search loop
        for it in range(1, self.cfg["n_iter"] + 1):
            self.logger.info(f"=== {self.name} iter {it}/{self.cfg['n_iter']} ===")

            # a) random selection
            idx_next = self._pick(scores)
            scores = np.delete(scores, idx_next)
            smi_next = smi_pool.pop(idx_next)
            x_next = pool_raw_df.iloc[idx_next]

            # b) evaluate
            if smi_next in cache:
                y_next = cache[smi_next]
                self.logger.info(f"Cache hit for random sample: {smi_next}")
            else:
                y_next = simulator.compute(smi_next, index=it, total=self.cfg["n_iter"])
                append_to_cache(smi_next, float(y_next), cache_path)
                cache[smi_next] = y_next

            train_y_raw.append(float(y_next))
            best_random = min(best_random, y_next)

            # c) update raw dfs
            train_raw_df = pd.concat(
                [train_raw_df, x_next.to_frame().T], ignore_index=True
            )
            pool_raw_df = pool_raw_df.drop(idx_next).reset_index(drop=True)

            self.logger.info(
                f"Iter {it} → {self.best_col}={best_random:.3g}, best_pool_min={best_pool_min:.3g}"
            )

            records.append(
                {
                    "iter": it,
                    "smiles": smi_next,
                    "energy": float(y_next),
                    self.best_col: best_random,
                    "best_pool_min": best_pool_min,
                }
            )

        # Write out results
        dataset_name = Path(input_csv).stem
        dir_name = f"{dataset_name}_{self.name}"
        out_dir = (
            Path(self.cfg["results_dir"]) / dir_name / f"seed{self.cfg['random_state']}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out_dir / f"{self.name}_history.csv", index=False)
        self.logger.info(f"Finished {self.name} ✅  (results in {out_dir})")
