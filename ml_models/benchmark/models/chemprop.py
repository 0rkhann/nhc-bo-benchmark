"""Chemprop D-MPNN on SMILES, optionally with molecule-level descriptors (x_d)."""
import os

import numpy as np


class Chemprop:
    name, tunable, grid = "chemprop", True, None
    refit_estimate_s = 2700
    MAX_EPOCHS, PATIENCE = 100, 15

    def defaults(self):
        return {"depth": 3, "message_hidden_dim": 300, "ffn_num_layers": 1, "ffn_hidden_dim": 300,
                "dropout": 0.0, "max_lr": 1e-3}

    def search_space(self, trial):
        return {"depth": trial.suggest_int("depth", 2, 6),
                "message_hidden_dim": trial.suggest_int("message_hidden_dim", 300, 1200, step=100),
                "ffn_num_layers": trial.suggest_int("ffn_num_layers", 1, 3),
                "ffn_hidden_dim": trial.suggest_int("ffn_hidden_dim", 300, 1200, step=100),
                "dropout": trial.suggest_float("dropout", 0.0, 0.4),
                "max_lr": trial.suggest_float("max_lr", 1e-4, 3e-3, log=True)}

    def _dataset(self, smiles, y, X):
        from chemprop import data, featurizers
        x_d = X if X.shape[1] else None
        dps = [data.MoleculeDatapoint.from_smi(s, None if y is None else [float(v)],
                                               x_d=None if x_d is None else x_d[i])
               for i, (s, v) in enumerate(zip(smiles, y if y is not None else [0.0] * len(smiles)))]
        return data.MoleculeDataset(dps, featurizers.SimpleMoleculeMolGraphFeaturizer())

    def fit(self, X, y, params, smiles=None):
        import lightning.pytorch as pl
        import torch
        from chemprop import data, models, nn
        from lightning.pytorch.callbacks import EarlyStopping

        torch.manual_seed(0); torch.set_num_threads(os.cpu_count())
        idx = np.random.default_rng(0).permutation(len(y))
        n_val = max(1, int(round(0.1 * len(y))))
        va, tr = idx[:n_val], idx[n_val:]
        self.n_train_seen, self.n_val_seen = len(tr), len(va)
        smiles = list(smiles)
        train = self._dataset([smiles[i] for i in tr], y[tr], X[tr])
        val = self._dataset([smiles[i] for i in va], y[va], X[va])
        y_scaler = train.normalize_targets(); val.normalize_targets(y_scaler)
        xd_tf = None
        if X.shape[1]:
            xd_scaler = train.normalize_inputs("X_d"); val.normalize_inputs("X_d", xd_scaler)
            xd_tf = nn.ScaleTransform.from_standard_scaler(xd_scaler)
        mp = nn.BondMessagePassing(d_h=params["message_hidden_dim"], depth=params["depth"], dropout=params["dropout"])
        ffn = nn.RegressionFFN(input_dim=mp.output_dim + X.shape[1], hidden_dim=params["ffn_hidden_dim"],
                               n_layers=params["ffn_num_layers"], dropout=params["dropout"],
                               output_transform=nn.UnscaleTransform.from_standard_scaler(y_scaler))
        self.model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=True, X_d_transform=xd_tf,
                                 max_lr=params["max_lr"], init_lr=params["max_lr"] / 10,
                                 final_lr=params["max_lr"] / 10)
        self.trainer = pl.Trainer(max_epochs=params.get("epochs", self.MAX_EPOCHS), accelerator="cpu",
                                  logger=False, enable_checkpointing=False, enable_progress_bar=False,
                                  callbacks=[EarlyStopping("val_loss", patience=self.PATIENCE)],
                                  deterministic=False)
        self.trainer.fit(self.model, data.build_dataloader(train, shuffle=True, seed=0),
                         data.build_dataloader(val, shuffle=False))
        return self

    def predict(self, X, smiles=None):
        from chemprop import data
        ds = self._dataset(list(smiles), None, X)        # X_d is scaled by the model's X_d_transform
        out = self.trainer.predict(self.model, data.build_dataloader(ds, shuffle=False))
        return np.concatenate([o.numpy().ravel() for o in out])
