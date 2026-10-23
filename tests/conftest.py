"""Shared fixtures: the real trained artifacts, real data and a client."""

import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from starter.examples import HIGH_INCOME_EXAMPLE, LOW_INCOME_EXAMPLE
from starter.ml.data import CAT_FEATURES, LABEL, load_data, process_data
from starter.ml.model import load_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "census.csv"
MODEL_DIR = PROJECT_ROOT / "model"
SMALL_CSV_ROWS = 1500


@pytest.fixture(autouse=True)
def default_api_key(monkeypatch):
    """Tests assume the development key unless they set the variable."""
    monkeypatch.delenv(main.API_KEY_ENV, raising=False)


@pytest.fixture(scope="session")
def artifacts():
    """The trained model and fitted preprocessing shipped under model/."""
    return load_artifacts(MODEL_DIR)


@pytest.fixture(scope="session")
def client():
    """Test client whose lifespan loads the real artifacts once."""
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def census_df():
    """The cleaned bundled dataset."""
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def small_df(census_df):
    """A small frame containing both classes for quick fits."""
    return census_df.groupby(LABEL).head(150).reset_index(drop=True)


@pytest.fixture(scope="module")
def processed_small(small_df):
    """process_data(training=True) output for small_df."""
    return process_data(
        small_df, categorical_features=CAT_FEATURES, label=LABEL,
        training=True,
    )


@pytest.fixture
def high_income_payload():
    """A fresh copy of the >50K example, keyed by Census names."""
    return copy.deepcopy(HIGH_INCOME_EXAMPLE)


@pytest.fixture
def low_income_payload():
    """A fresh copy of the <=50K example, keyed by Census names."""
    return copy.deepcopy(LOW_INCOME_EXAMPLE)


@pytest.fixture
def api_key_headers():
    """Headers carrying the development API key."""
    return {main.API_KEY_HEADER_NAME: main.DEFAULT_DEV_API_KEY}


@pytest.fixture(scope="session")
def small_csv(tmp_path_factory):
    """Header plus the first 1500 raw rows, whitespace preserved."""
    with DATA_PATH.open(encoding="utf-8") as handle:
        lines = [next(handle) for _ in range(SMALL_CSV_ROWS + 1)]
    path = tmp_path_factory.mktemp("data") / "census_small.csv"
    path.write_text("".join(lines), encoding="utf-8")
    return path
