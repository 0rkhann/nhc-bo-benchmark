import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_package_imports():
    import ml_models.benchmark  # noqa: F401


def test_setup_declares_ml_extra_and_python_floor():
    tree = ast.parse((ROOT / "setup.py").read_text())
    src = (ROOT / "setup.py").read_text()
    assert 'python_requires=">=3.10"' in src
    assert '"ml"' in src and "requirements-ml.txt" in src


def test_ml_requirements_are_pinned():
    pins = dict(l.split("==") for l in (ROOT / "requirements-ml.txt").read_text().split()
                if "==" in l)
    assert pins == {"chemprop": "2.3.1", "tabpfn": "9.0.0", "tabicl": "2.2.0", "optuna": "5.0.0"}
