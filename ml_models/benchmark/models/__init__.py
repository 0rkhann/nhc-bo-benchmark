"""Model registry. Heavy adapters are imported only when requested."""
import importlib

REGISTRY = {"mean": "mean.Mean", "ridge": "ridge.Ridge", "rf": "rf.RF", "xgb": "xgb.XGB",
            "gp": "gp.GP", "tabpfn": "tabpfn.TabPFN", "tabicl": "tabicl.TabICL",
            "chemprop": "chemprop.Chemprop", "outlier": "outlier.OutlierRank"}


def get(name: str):
    module, cls = REGISTRY[name].split(".")
    return getattr(importlib.import_module(f"{__name__}.{module}"), cls)()
