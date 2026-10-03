import numpy as np
import optuna
import pytest

pytest.importorskip("chemprop")
from ml_models.benchmark import models

SMI = ["CCO", "CCCO", "CCCCO", "CCN", "CCCN", "CCCCN", "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCOC",
       "CCCOC", "CC(C)O", "CC(C)N", "OCCO", "NCCN", "CC=O", "CCC=O", "CC(=O)O", "CCC(=O)O", "COC"] * 3


def small_params():
    return {"depth": 2, "message_hidden_dim": 64, "ffn_num_layers": 1, "ffn_hidden_dim": 64,
            "dropout": 0.0, "max_lr": 1e-3, "epochs": 15}


@pytest.mark.parametrize("n_xd", [0, 3])
def test_chemprop_fits_with_and_without_descriptors(n_xd):
    rng = np.random.default_rng(0)
    y = np.array([len(s) for s in SMI], dtype=float) + 0.1 * rng.normal(size=len(SMI))
    X = rng.normal(size=(len(SMI), n_xd))
    m = models.get("chemprop").fit(X[:50], y[:50], small_params(), smiles=SMI[:50])
    pred = m.predict(X[50:], smiles=SMI[50:])
    assert pred.shape == (10,) and np.isfinite(pred).all()


def test_chemprop_trains_only_on_given_rows():
    m = models.get("chemprop").fit(np.zeros((40, 0)), np.arange(40.0), small_params(), smiles=SMI[:40])
    assert m.n_train_seen + m.n_val_seen == 40      # its own split is never used


def test_search_space_keys():
    a = models.get("chemprop")
    p = a.search_space(optuna.trial.FixedTrial(a.defaults()))
    assert {"depth", "message_hidden_dim", "ffn_num_layers", "ffn_hidden_dim", "dropout", "max_lr"} <= set(p)
