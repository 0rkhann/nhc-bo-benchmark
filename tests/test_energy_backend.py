"""EnergySimulator delegates to a backend named in BO_ENERGY_BACKEND and needs none for cached runs."""
import sys
import types

import pytest

from src.simulation import BACKEND_ENV, EnergySimulator


def test_without_backend_compute_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    sim = EnergySimulator()
    with pytest.raises(RuntimeError, match=BACKEND_ENV):
        sim.compute("C[n+]1[c-]n(C)cc1")


def test_backend_from_env_is_called_with_the_smiles(monkeypatch):
    calls = []
    mod = types.ModuleType("fake_energy_backend")
    mod.energy = lambda smiles: calls.append(smiles) or -12.5
    monkeypatch.setitem(sys.modules, "fake_energy_backend", mod)
    monkeypatch.setenv(BACKEND_ENV, "fake_energy_backend:energy")

    assert EnergySimulator().compute("C[n+]1[c-]n(C)cc1", index=1, total=3) == -12.5
    assert calls == ["C[n+]1[c-]n(C)cc1"]


@pytest.mark.parametrize("spec", ["no_colon", "missing_module_xyz:energy", "fake_energy_backend:nope"])
def test_bad_backend_spec_fails_at_startup(monkeypatch, spec):
    monkeypatch.setitem(sys.modules, "fake_energy_backend", types.ModuleType("fake_energy_backend"))
    monkeypatch.setenv(BACKEND_ENV, spec)
    with pytest.raises((ValueError, ImportError, AttributeError)):
        EnergySimulator()
