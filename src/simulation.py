"""
Energy backend for molecules that are not in the energy cache.

The reported runs never call it: every molecule is in data/dft_G.json. The in-house workflow that
produced those energies is not part of this repository. To compute new molecules, point
BO_ENERGY_BACKEND at a function that takes a SMILES string and returns its energy in kJ/mol:

    export BO_ENERGY_BACKEND="my_package.energies:binding_free_energy"
"""
import importlib
import logging
import os

BACKEND_ENV = "BO_ENERGY_BACKEND"


def _load_backend(spec: str):
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"{BACKEND_ENV} must look like 'module:function', got {spec!r}")
    fn = getattr(importlib.import_module(module_name), attr)
    if not callable(fn):
        raise TypeError(f"{spec!r} is not callable")
    return fn


class EnergySimulator:
    def __init__(self, options=None):
        self.options = options
        self.logger = logging.getLogger(__name__)
        spec = os.environ.get(BACKEND_ENV)
        self._backend = _load_backend(spec) if spec else None
        if self._backend is None:
            self.logger.info(f"No {BACKEND_ENV} set: energies come from the cache only.")

    def compute(self, smiles: str, index: int = None, total: int = None) -> float:
        progress = f"{index}/{total}: " if index is not None and total is not None else ""
        if self._backend is None:
            raise RuntimeError(
                f"Cannot compute {progress}{smiles}: it is not in the energy cache and no backend is set. "
                f"Add it to the cache or set {BACKEND_ENV}='module:function'."
            )
        self.logger.info(f"Computing {progress}{smiles}")
        energy = float(self._backend(smiles))
        self.logger.info(f"Computed energy for {smiles}: {energy:.2f} kJ/mol")
        return energy
