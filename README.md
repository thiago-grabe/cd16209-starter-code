# Building and Serving an ML Model with FastAPI

Train and evaluate a Census income classifier, examine its performance across
categorical data slices, test the model and API, and serve real predictions with
FastAPI. Complete this project in a Udacity Workspace or equivalent local Python
environment. The data and model-card template are bundled; no third-party service
account, hosted runtime, remote repository, or public URL is required. The Workspace requires no learner package installation. Optional local/reviewer
setup may access a Python package index. Local Git is optional.

## Project implementation (learner)

This section documents the completed implementation that sections 1 to 5
ask for: a `HistGradientBoostingClassifier` trained on the bundled Census
data, its saved inference artifacts under `model/`, the FastAPI service in
`main.py`, the `request.py` client, the tests, and the evidence captured for
submission.

### Quickstart

In the Udacity Workspace the trained artifacts are already shipped under
`model/`, so there is nothing to install and nothing to retrain. From the
project root, run the quality gate and start the server:

```sh
python scripts/check.py
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

With the server running, open `http://127.0.0.1:8000/ui` in a browser or run
`python request.py` in a second terminal. Do not run the training command in
the Workspace; the shipped artifacts are the ones the tests and the evidence
files refer to.

For a local checkout, create the environment with the `venv` commands in
[ENVIRONMENT.md](ENVIRONMENT.md), or with the `uv` equivalents:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
```

`requirements-dev.txt` is optional: it only adds `coverage` for the report
described under "Test coverage" and is never needed by `scripts/check.py`.

### Training

```sh
python -m starter.train_model
```

The script loads `data/census.csv` through `starter.ml.data.load_data`, which
strips incidental whitespace from headers and values while keeping the
original feature names (hyphens included) and the `?` category. It holds out
a stratified 20% evaluation split (`random_state=42`), fits the
`OneHotEncoder` and the `LabelBinarizer` on the training split only, trains
the classifier, prints the held-out metrics (plus ready-to-paste markdown
rows for the model card), writes `slice_output.txt` and saves the artifacts
under `model/`. Options: `--data`, `--model-dir`, `--slice-output`,
`--test-size` (default 0.2) and `--random-state` (default 42).

| Item | Value |
| --- | --- |
| Model | `HistGradientBoostingClassifier` (`max_iter=200`, `learning_rate=0.1`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `early_stopping=False`, `random_state=42`) |
| Rows | 32,561 total: 26,048 training, 6,513 evaluation |
| Processed features | 108 (6 continuous + 102 one-hot columns) |
| Precision | 0.7855 |
| Recall | 0.6773 |
| F1 | 0.7274 |

The same numbers are stored in `model/metadata.json` (together with the
library versions used) and discussed in `model_card.md`. Retraining
regenerates `model/` (including `model/metadata.json`) and
`slice_output.txt`; the run is deterministic on one machine, but on another
platform or library build the last decimals may shift, in which case
`model_card.md` must be updated to match. Do not retrain in the Workspace.

### Model artifacts and loading

| File | Object | Produced by | Loaded by |
| --- | --- | --- | --- |
| `model/model.pkl` | trained `HistGradientBoostingClassifier` | `save_artifacts` during `python -m starter.train_model` | `load_artifacts` |
| `model/encoder.pkl` | fitted `OneHotEncoder` for the 8 categorical features | `save_artifacts` | `load_artifacts` |
| `model/lb.pkl` | fitted `LabelBinarizer` (`<=50K`, `>50K`) | `save_artifacts` | `load_artifacts` |
| `model/metadata.json` | feature order, categorical features, split, metrics, library versions, artifact names | `save_artifacts` | `load_artifacts` |

`save_artifacts` and `load_artifacts` live in `starter/ml/model.py`; the
three pickles are written with the standard library `pickle` module. Loading
in your own code:

```python
from pathlib import Path

from starter.ml.model import load_artifacts

artifacts = load_artifacts(Path("model"))
```

`load_artifacts` returns an `Artifacts` named tuple with `model`, `encoder`,
`lb` and `metadata`. It never retrains or refits anything; it raises
`FileNotFoundError` listing every missing file (with the hint to run the
training command), raises `ValueError` when the metadata's feature layout
differs from `starter.ml.data.FEATURE_COLUMNS` or `CAT_FEATURES`, and emits
a `UserWarning` when the installed scikit-learn differs from the training
version. `predict_labels(artifacts, frame)` runs the complete inference path
on a DataFrame of raw features and returns the labels and `P(>50K)`.

The API loads the same files once: `main.get_artifacts()` wraps
`load_artifacts(MODEL_DIR)` in `functools.lru_cache`, where `MODEL_DIR` is
the `model/` directory next to `main.py`, so the server works from any
working directory and from an extracted ZIP. The FastAPI lifespan handler
calls it at startup, so missing artifacts fail immediately instead of on the
first request, and the routes receive the cached object through `Depends`.

Pickle files can execute code when loaded; only load `model/*.pkl` from a
trusted source such as this repository or your own training run.

### Slice evaluation

`starter/ml/slices.py` holds the reusable slice function
`compute_slice_metrics(frame, y, preds, feature)`, which returns precision,
recall, F1 and the row count for every value of one categorical feature,
plus `compute_all_slices`, `format_slice_line` and `write_slice_output`.
Training runs it for all 8 categorical features on the held-out split and
writes `slice_output.txt`: three `# ` header lines (split description,
overall metrics, line format) followed by one self-describing line per
feature and value, 101 in total, for example:

```text
feature=workclass value=Private n=4597 precision=0.7937 recall=0.6482 f1=0.7136
```

Small slices are noisy, and `compute_model_metrics` uses `zero_division=1`,
so a slice with no positive labels and no positive predictions reports 1.0.
See `model_card.md` for the discussion of the slice results.

### API

Start the server from the project root:

```sh
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| GET | `/` | none | Welcome JSON (`message`, `docs`, `predict`, `ui`, `health`) |
| GET | `/health` | none | `{"status": "ok", "model_type": "HistGradientBoostingClassifier"}` |
| GET | `/features` | none | Feature lists, categories seen in training, the two example payloads |
| POST | `/predict` | none | Prediction for one record |
| POST | `/predict/batch` | `X-API-Key` | Predictions for 1 to 1000 records sent as `{"instances": [...]}`, plus `count` |
| GET | `/model/info` | `X-API-Key` | Contents of `model/metadata.json` |
| GET | `/ui` | none | Single-file web UI |
| GET | `/docs`, `/openapi.json` | none | Swagger UI and the OpenAPI schema |

`POST /predict` takes one record with the original Census feature names. The
example below is a real row of `data/census.csv`; it is also the first
example in the OpenAPI schema and in `/docs`:

```json
{
  "age": 44,
  "workclass": "Private",
  "fnlgt": 198282,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital-gain": 15024,
  "capital-loss": 0,
  "hours-per-week": 60,
  "native-country": "United-States"
}
```

Response (`probability_over_50k` rounded here to four decimals):

```json
{"prediction": ">50K", "probability_over_50k": 0.9997}
```

```sh
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 44, "workclass": "Private", "fnlgt": 198282,
       "education": "Bachelors", "education-num": 13,
       "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
       "relationship": "Husband", "race": "White", "sex": "Male",
       "capital-gain": 15024, "capital-loss": 0, "hours-per-week": 60,
       "native-country": "United-States"}'
```

Notes on the request model (`CensusInput` in `main.py`):

- The hyphenated names (`education-num`, `marital-status`, `capital-gain`,
  `capital-loss`, `hours-per-week`, `native-country`) are Pydantic aliases;
  the snake_case field names (`education_num` and so on) are accepted as
  well. Extra keys such as `salary` are ignored, so a CSV row converted to
  JSON can be posted as is. Leading and trailing whitespace in strings is
  stripped.
- Validation: a missing field, a wrong type or an out-of-range value (`age`
  0 to 120, `education-num` 1 to 16, `hours-per-week` 1 to 168, non-negative
  `fnlgt`, `capital-gain` and `capital-loss`) returns `422` with the field
  named in `detail[].loc`.
- Categories unseen in training (for example a new `native-country`) are
  accepted and encoded as all zeros by the fitted encoder
  (`handle_unknown="ignore"`), exactly as during evaluation.
- `prediction` is `<=50K` or `>50K` (the label binarizer's classes) and
  `probability_over_50k` is the model's probability of `>50K`.

### Authentication

`POST /predict/batch` and `GET /model/info` require the `X-API-Key` header;
every other route is public. The expected key is read from the
`CENSUS_API_KEY` environment variable on each request. When the variable is
unset, the development placeholder `census-dev-key` is accepted. That
placeholder is not a secret: it exists so a reviewer can exercise the
protected routes without any configuration. Set `CENSUS_API_KEY` before
starting Uvicorn to require your own key. A missing or wrong key returns
`401 Unauthorized` with the header `WWW-Authenticate: ApiKey`; keys are
compared with `secrets.compare_digest`. The scheme appears in the OpenAPI
document as `APIKeyHeader`, so `/docs` offers an "Authorize" button.

```sh
# Without the key: 401
curl -i http://127.0.0.1:8000/model/info

# With the key: 200 and the training metadata
curl -i -H "X-API-Key: census-dev-key" http://127.0.0.1:8000/model/info

# Batch prediction with the key
curl -s -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: census-dev-key" \
  -d '{"instances": [{"age": 44, "workclass": "Private", "fnlgt": 198282,
       "education": "Bachelors", "education-num": 13,
       "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
       "relationship": "Husband", "race": "White", "sex": "Male",
       "capital-gain": 15024, "capital-loss": 0, "hours-per-week": 60,
       "native-country": "United-States"}]}'

# Require your own key instead of the placeholder
CENSUS_API_KEY=my-key python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Web UI

`GET /ui` serves `frontend/index.html`, a single self-contained page with
inline CSS and JavaScript and no external assets, so it works offline. On
load it calls `GET /features` to fill the eight category dropdowns with the
values seen in training, offers "Load example" buttons for the `>50K` and
`<=50K` records, posts the form to `POST /predict` and shows the predicted
class, the probability and the raw JSON response; validation (`422`) and
server errors are displayed with the field and message returned by the API.
The page links to `/docs`. `/ui` and `/openapi.json` work offline; `/docs`
(Swagger UI) loads its scripts and styles from a CDN and needs internet
access.

### Real HTTP requests with request.py

With the server running, in a second terminal at the project root:

```sh
python request.py
python request.py --base-url http://127.0.0.1:8000
python request.py --base-url http://127.0.0.1:8000 --timeout 5
```

`request.py` uses `requests` (a session with `trust_env = False`, so proxy
variables are ignored) and imports nothing from the project. It calls
`GET /` and then `POST /predict` once per class with two real rows from the
data: the `>50K` example shown above and a `<=50K` example (copies of
`starter/examples.py`, kept in sync by a test). For every call it prints the
method and URL, the payload, the HTTP status and the response body, followed
by an `OK`, `UNEXPECTED` or `MISMATCH` verdict. `--base-url` defaults to
`http://127.0.0.1:8000` and `--timeout` (seconds per request) defaults to
10.

| Exit code | Meaning |
| --- | --- |
| 0 | every response had the expected status and prediction |
| 1 | an unexpected status, body or prediction (details printed above it) |
| 2 | the server could not be reached (connection error or timeout) |

### Tests and quality gate

All tests use the shipped artifacts and the real data; no prediction is
mocked.

| File | What it covers |
| --- | --- |
| `tests/conftest.py` | Fixtures: the loaded artifacts, a `TestClient` with the lifespan active, real data subsets, a 1,500-row CSV, example payloads and the API-key header |
| `tests/test_data.py` | `load_data` and `clean_frame` whitespace cleaning, missing-column error, `process_data` in training and inference mode, unknown categories, column-order sensitivity |
| `tests/test_model.py` | `train_model`, `inference`, `compute_model_metrics`, `save_artifacts`/`load_artifacts` round trip and error paths, `predict_labels`, consistency of the shipped metadata |
| `tests/test_slices.py` | Slice metrics on synthetic and real data, the line format, the file writer and the shipped `slice_output.txt` |
| `tests/test_train_model.py` | CLI defaults, end-to-end training on the 1,500-row subset into a temporary directory, reproducibility, missing data file |
| `tests/test_api.py` | `GET /`, `/health`, `/features`, `POST /predict` for both classes against the real model, parity with `predict_labels`, aliases and snake_case names, whitespace, `422` cases, OpenAPI aliases, examples and security scheme, fail-fast startup |
| `tests/test_auth.py` | `401`/`200` behaviour of `/model/info` and `/predict/batch`, `CENSUS_API_KEY` override, non-ASCII keys |
| `tests/test_frontend.py` | `/ui` serves the HTML form, `404` when the file is missing |
| `tests/test_request.py` | `request.py` output and exit codes with a fake HTTP client, payload parity with `starter/examples.py`, and one real round trip against a Uvicorn server on a free port |
| `tests/test_docs.py` | `submission_files.txt`, `model_card.md`, `evidence/openapi.json` and this README stay consistent with the code and artifacts |

Run the required gate from the project root:

```sh
python scripts/check.py
```

It runs `pytest` and then `flake8 --extend-exclude .venv` with the same
interpreter and prints `Local tests and lint passed.` when both succeed.
`pytest` is configured in `pyproject.toml` (`testpaths = ["tests"]`,
`pythonpath = ["."]`), so `python -m pytest -q` also works without an
editable install.

### Test coverage

Coverage is optional local tooling; the Workspace and `scripts/check.py`
never use it. Install `coverage` (pinned in `requirements-dev.txt`, which
also pulls in `requirements.txt`) and run the wrapper:

```sh
python -m pip install -r requirements-dev.txt
python scripts/run_coverage.py
```

With `uv`: `uv pip install --python .venv/bin/python -r requirements-dev.txt`.
The script runs `coverage erase`, `coverage run -m pytest -q` and
`coverage report -m` from the project root, prints their output and writes
the commands, output and exit statuses to `evidence/coverage.txt`
(`--output PATH` changes the location, `--html` also writes `htmlcov/`). The
configuration lives in `pyproject.toml` under `[tool.coverage.run]` and
`[tool.coverage.report]`: branch coverage over `main.py`, `request.py` and
`starter/`, with `tests/`, `scripts/`, `setup.py`, `sanitycheck.py` and
`.venv/` omitted, and `fail_under = 100`. The target is 100% of lines and
branches; the script exits nonzero below that threshold, and with status 2
when `coverage` is not installed. The quality gate itself is unchanged.

Current report (`evidence/coverage.txt`, 72 tests):

| File | Statements | Branches | Coverage |
| --- | --- | --- | --- |
| `main.py` | 132 | 4 | 100% |
| `request.py` | 73 | 4 | 100% |
| `starter/examples.py` | 4 | 0 | 100% |
| `starter/ml/data.py` | 41 | 8 | 100% |
| `starter/ml/model.py` | 73 | 14 | 100% |
| `starter/ml/slices.py` | 35 | 8 | 100% |
| `starter/train_model.py` | 63 | 2 | 100% |
| Total | 421 | 40 | 100% |

### Evidence files

| File | Contents |
| --- | --- |
| `evidence/validation.txt` | The `python scripts/check.py` command, its pytest and flake8 output and the line `exit status: 0` |
| `evidence/http.txt` | The Uvicorn command (started on port 8020 because ports 8000 and 8001 were taken by unrelated local services at capture time), `curl` calls to `GET /` and `POST /predict` for both classes, `python request.py --base-url http://127.0.0.1:8020` with its exit status, `GET /model/info` with and without the API key, and the server log |
| `evidence/openapi.json` | The schema exported from the running server with `curl -s http://127.0.0.1:8000/openapi.json` and pretty-printed with `python -m json.tool`; it contains the `CensusInput` examples, the hyphenated aliases and the `APIKeyHeader` security scheme |
| `evidence/coverage.txt` | The commands and report written by `python scripts/run_coverage.py` |

The files are captured from real runs against the shipped artifacts and are
never edited by hand. `tests/test_docs.py` compares `evidence/openapi.json`
with the live application when the file is present.

### Submission manifest

`submission_files.txt` names the files that the submission must carry in
addition to the fixed list in [SUBMISSION.md](SUBMISSION.md), one
project-relative path per line (comment lines start with `#`):

- `model/model.pkl`, `model/encoder.pkl`, `model/lb.pkl` and
  `model/metadata.json`: the artifacts loaded by `main.py`. Git ignores
  `*.pkl`, so the ZIP helper includes the pickles directly from disk only
  because they are listed here; for the GitHub route they are added with
  `git add -f` using these exact paths.
- `frontend/index.html`: the web UI served at `/ui`.
- `requirements-dev.txt`, `scripts/run_coverage.py` and
  `evidence/coverage.txt`: the optional coverage tooling and its report.

`scripts/package_submission.py` reads this list when building the ZIP, and
`tests/test_docs.py` checks that every listed file exists (evidence files
excepted) and that the artifacts are present. Generated `submission*.zip`
archives are ignored by Git.

## 1. Set up the environment

Open a terminal in the project root: the directory containing `requirements.txt`,
`setup.py`, `main.py`, and `data/census.csv`. Keep all commands below in this root.
`starter/` is the Python package; for example, preprocessing lives at
`starter.ml.data`.

The preinstalled Udacity Workspace is the default development and submission
path. Use its Python 3.13 `ml` environment; do not create a new virtual environment
or install packages in the Workspace. Open `/workspace/cd16209-starter-code` and
confirm `python --version` reports Python 3.13. The terminal and editor should use
`/opt/conda/envs/ml/bin/python`.

If the updated source or required software is missing, ask course support to
provision the image rather than installing it yourself. Image deployment and
reset verification are tracked separately; these instructions are the target
configuration, not proof that every existing Workspace has been updated.

For optional local development or reviewer reproduction, follow the isolated
installation commands in [ENVIRONMENT.md](ENVIRONMENT.md), activate that
environment, and use `python` for the commands below. `requirements.txt` remains
the pinned dependency source for maintainers and local/reviewer setup.

## 2. Complete the model and evaluation

The starter is intentionally unfinished. Complete `starter/train_model.py` and
the functions in `starter/ml/model.py`; use the provided preprocessing in
`starter/ml/data.py` and add your own supporting code as needed.

- Inspect `data/census.csv` and clean incidental whitespace in headers and values.
  Preserve the feature names, including their hyphens, and the meaning of values.
- Split data for training and evaluation (or use appropriate cross-validation).
  Fit preprocessing on training data only; transform held-out data with those
  fitted objects to avoid leakage.
- Train a classifier and report precision, recall, and F1 on held-out data.
- Choose at least one categorical feature and report precision, recall, and F1
  for each distinct value in the evaluation data. Write these slice results to
  `slice_output.txt`, identifying the feature and value for each slice.
- Save the trained model and all fitted preprocessing objects needed for inference
  under `model/`. A combined artifact or separate files are both acceptable.
  Document their filenames and loading procedure in your submitted README.
- Complete `model_card.md` using the bundled `model_card_template.md`, including
  the model's intended use, evaluation results, and limitations.

After implementing the training path, run:

```sh
python -m starter.train_model
```

Training must produce the artifacts used by the API. Inference must load these
artifacts without retraining or fitting new preprocessing. Include the artifacts
in every submission route even if Git ignores them.

## 3. Complete the API and tests

Implement the FastAPI application named `app` in `main.py`:

- GET `/` returns a welcome message.
- A separate POST route accepts features and returns a real model prediction.
  Choose and document the route and response format.
- Use type hints and a Pydantic request model with an example. Use field aliases
  for hyphenated Census names rather than renaming those features in the dataset.
- Load the saved model and fitted preprocessing using paths relative to the
  project, without remote storage or absolute paths from your own machine.

Write completed tests under `tests/`: unit tests covering at least three
model-related functions, and at least three API tests. The API tests must check
GET status/body and separate POST cases for each prediction class. Both POST
tests must exercise the actual trained model and preprocessing, not a mocked or
hard-coded prediction. Check response status and prediction content.

Run the local quality gate:

```sh
python scripts/check.py
```

It runs pytest and then flake8, stopping on failure (including no tests). Fix
failures in your implementation; do not weaken tests or exclude unfinished code
to obtain a pass. The pristine starter is not expected to pass. This is local
automated quality validation, not hosted CI/CD or automatic cloud deployment.
`sanitycheck.py` is an optional interactive, heuristic advisory tool and is not
part of this gate.

## 4. Run and query the local API

After training and implementing the API, start the server:

```sh
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Leave it running. In a second terminal at the same project root, run the
`request.py` script you implement:

```sh
python request.py
```

Use `requests` in that script and implement a `--base-url` argument defaulting
to `http://127.0.0.1:8000`. For example, the following invocation must use the
specified server: `python request.py --base-url http://127.0.0.1:8000`.
Call GET `/` and your inference POST route with valid input,
print each HTTP status and response body (including the inference result), and
exit nonzero on an unexpected result. Use a timeout so a failed server does not
leave the script waiting indefinitely. This script must call the running server;
TestClient tests alone do not demonstrate real HTTP execution.

The API's local `/docs` page can help inspect the request model. Export its
`/openapi.json` schema and verify that your input example and aliases appear.

## 5. Capture evidence and submit

Follow [SUBMISSION.md](SUBMISSION.md) for the required files on every route.
Create `evidence/validation.txt` (command, actual pytest/flake8 output and exit
status zero), `evidence/http.txt` (real GET/POST commands, statuses and bodies),
and `evidence/openapi.json` (request example and aliases). Record every trained
model/preprocessing artifact and extra supporting file in `submission_files.txt`.

Submit through Workspace by default after checking that its submission includes
all required files. GitHub and ZIP are optional alternatives with the same
source, tests, data, trained artifacts, documentation, and evidence requirements.
No external account is needed for the Workspace or ZIP route. Save your work and
keep a downloaded backup.

For optional ZIP submission, run the following from the project root:

```sh
python scripts/package_submission.py submission.zip
```

The helper includes listed Git-ignored artifacts and requires a new output
filename. Extract and test the
archive's `submission/` root using the local/reviewer setup in ENVIRONMENT.md.
For every route, inference must load submitted artifacts without retraining or
accessing the original working directory. See SUBMISSION.md for route-specific
inclusion and reproduction checks.
