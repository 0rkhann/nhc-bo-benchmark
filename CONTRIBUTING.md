# Contributing

## Setup

Requires Python 3.8 or newer.

```bash
git clone https://github.com/0rkhann/nhc-bo-benchmark.git
cd nhc-bo-benchmark

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Install the project with its dev dependencies (pytest, black, flake8, mypy):

```bash
pip install -e ".[dev]"
```

Optional extras (`xgboost`, `pyopls`) can be added with `pip install -e ".[dev,optional]"`.

## Running tests

From the repository root:

```bash
pytest
```

Useful variations:

```bash
pytest path/to/test_file.py   # a single file
pytest -k "name_fragment"     # tests matching a name
pytest -x -v                  # stop at first failure, verbose
```

Note: the repository does not contain tests yet, so `pytest` will report that no tests were collected. Place new tests in a `tests/` directory in files named `test_*.py`.

## Code style

Before opening a pull request, format and lint your changes:

```bash
black .
flake8
```
