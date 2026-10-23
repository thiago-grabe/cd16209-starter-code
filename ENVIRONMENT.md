# Project environment

## Default Udacity Workspace

The target Workspace provides Python 3.13 and all project dependencies in the
`ml` environment. Learners do not run pip or create a virtual environment. The
terminal's `python` and the editor must use `/opt/conda/envs/ml/bin/python`, with
`/workspace/cd16209-starter-code` as the project root. Run project commands with
`python`, including `python scripts/check.py` and `python -m uvicorn main:app`.

If the project files or required software are missing, contact course support
before starting the project.

## Optional local development and reviewer setup

Use Python 3.13. From the supplied project's root on Linux or macOS:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip check
source .venv/bin/activate
```

After activation, use `python` for the project commands in README.md. On Windows,
create the environment with your Python 3.13 interpreter, use
`.venv\Scripts\python.exe` for installation, and activate with
`.venv\Scripts\Activate.ps1` in PowerShell. Installation may access a Python
package index; execution uses local data and artifacts. This is optional
local/reviewer provisioning, not a Workspace learner requirement.

`pyproject.toml` declares the setuptools backend for local/editable installation.
Python 3.13 is the tested interpreter; the metadata minimum does not establish
later-version compatibility. Direct dependencies are pinned in requirements.txt;
transitive dependencies are not a complete environment lock.

## Dependency scope

NumPy, pandas, and scikit-learn support model development; FastAPI, Pydantic, and
Uvicorn provide the JSON API; pytest and httpx support model and TestClient tests;
requests makes real HTTP calls; flake8 checks source. Basic Uvicorn does not need
its optional standard extras. No project imports require legacy aequitas/Flask,
plotting, notebook, multipart, or async-test packages. Workspace editor/notebook
infrastructure is separate from this project environment.

## Local quality validation

After completing the learner implementation and tests, run `python scripts/check.py`
from the project root. It uses the same interpreter for pytest and flake8, stops
on failure (including no tests), and prints success only after both pass.
The optional local `.venv` is excluded from linting, not unfinished source.
The pristine starter is intentionally incomplete and is not expected to pass.
`sanitycheck.py` remains optional interactive heuristic guidance, not a gate.
This is automated local quality validation, not hosted CI/CD or cloud deployment.

## Optional coverage tooling

Test coverage uses `coverage`, which `requirements.txt` does not include.
After activating the local environment, run
`python -m pip install -r requirements-dev.txt`, then
`python scripts/run_coverage.py` to write `evidence/coverage.txt`. This is
not a Workspace step, and `scripts/check.py` does not use it.
