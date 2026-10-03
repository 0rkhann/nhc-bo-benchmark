"""Time one inner fit+predict per model and representation, to size the grids."""
import argparse
import json
import time

import numpy as np

from ml_models.benchmark import data, models
from ml_models.benchmark.run import _features

INNER_TRAIN = 3653           # two thirds of an outer training part (5,480 rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rep", required=True)
    a = ap.parse_args()
    y_all = data.load_target(); smiles = list(y_all.index); y = y_all.to_numpy()
    X = _features(a.rep, smiles)
    idx = np.random.default_rng(0).permutation(len(y))
    tr, va = idx[:INNER_TRAIN], idx[INNER_TRAIN:INNER_TRAIN + 1827]
    m = models.get(a.model)
    t0 = time.time()
    m.fit(X[tr], y[tr], m.defaults(), smiles=[smiles[i] for i in tr]).predict(X[va], smiles=[smiles[i] for i in va])
    print(json.dumps({"model": a.model, "rep": a.rep, "fit_predict_s": round(time.time() - t0, 1),
                      "n_train": INNER_TRAIN}))


if __name__ == "__main__":
    main()
