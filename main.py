"""FastAPI application serving the Census income classifier.

Run from the project root::

    python -m uvicorn main:app --host 127.0.0.1 --port 8000

The trained model and its fitted preprocessing are loaded once from the
``model/`` directory next to this file; nothing is retrained or refitted.
``GET /`` and ``POST /predict`` are public. ``POST /predict/batch`` and
``GET /model/info`` require the ``X-API-Key`` header.
"""

import functools
import os
import secrets
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import Body, Depends, FastAPI, HTTPException, Security, status
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from starter.examples import (
    EXAMPLES,
    HIGH_INCOME_EXAMPLE,
    LOW_INCOME_EXAMPLE,
)
from starter.ml.data import CAT_FEATURES, CONTINUOUS_FEATURES, FEATURE_COLUMNS
from starter.ml.model import Artifacts, load_artifacts, predict_labels

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
FRONTEND_FILE = PROJECT_ROOT / "frontend" / "index.html"

API_KEY_ENV = "CENSUS_API_KEY"
API_KEY_HEADER_NAME = "X-API-Key"
# Development placeholder, not a secret: set CENSUS_API_KEY to override it.
DEFAULT_DEV_API_KEY = "census-dev-key"

WELCOME_MESSAGE = "Welcome to the Census Income Prediction API"
# Upper bound for the open-ended counts: keeps values inside numpy int64
# and matches the 32-bit range of the source data.
MAX_COUNT = 2_147_483_647

SalaryClass = Literal["<=50K", ">50K"]


# --------------------------------------------------------------------------
# Request and response models
# --------------------------------------------------------------------------
class CensusInput(BaseModel):
    """One person's Census features, named exactly as in the dataset."""

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [HIGH_INCOME_EXAMPLE, LOW_INCOME_EXAMPLE],
        },
    )

    age: int = Field(ge=0, le=120, description="Age in years")
    workclass: str = Field(description="Type of employer, e.g. Private")
    fnlgt: int = Field(
        ge=0, le=MAX_COUNT, description="Census final sampling weight"
    )
    education: str = Field(description="Highest education level")
    education_num: int = Field(
        alias="education-num",
        ge=1,
        le=16,
        description="Education level encoded as a number",
    )
    marital_status: str = Field(
        alias="marital-status", description="Marital status"
    )
    occupation: str = Field(description="Occupation category")
    relationship: str = Field(description="Relationship within household")
    race: str = Field(description="Race as recorded in the Census")
    sex: str = Field(description="Sex as recorded in the Census")
    capital_gain: int = Field(
        alias="capital-gain",
        ge=0,
        le=MAX_COUNT,
        description="Capital gains in USD",
    )
    capital_loss: int = Field(
        alias="capital-loss",
        ge=0,
        le=MAX_COUNT,
        description="Capital losses in USD",
    )
    hours_per_week: int = Field(
        alias="hours-per-week",
        ge=1,
        le=168,
        description="Hours worked per week",
    )
    native_country: str = Field(
        alias="native-country", description="Country of origin"
    )


class PredictionResponse(BaseModel):
    """Prediction for one record."""

    prediction: SalaryClass = Field(description="Predicted salary class")
    probability_over_50k: float = Field(
        ge=0.0,
        le=1.0,
        description="Model probability that the salary exceeds 50K",
    )


class BatchPredictionRequest(BaseModel):
    """Several records to score in one call."""

    instances: list[CensusInput] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Predictions in the same order as the submitted instances."""

    predictions: list[PredictionResponse]
    count: int


class MessageResponse(BaseModel):
    """Welcome message with the main routes of the API."""

    message: str
    docs: str
    predict: str
    ui: str
    health: str


class HealthResponse(BaseModel):
    """Liveness information."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    model_type: str


class FeaturesResponse(BaseModel):
    """Accepted features, known categories and example payloads."""

    feature_columns: list[str]
    continuous_features: list[str]
    categorical_features: list[str]
    categories: dict[str, list[str]]
    examples: dict[str, dict[str, Any]]


class ModelInfo(BaseModel):
    """Training metadata saved next to the model artifacts."""

    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    hyperparameters: dict[str, Any]
    label: str
    label_classes: list[str]
    positive_label: str
    feature_columns: list[str]
    categorical_features: list[str]
    continuous_features: list[str]
    n_features_processed: int
    n_rows: int
    n_train: int
    n_test: int
    split: dict[str, Any]
    metrics: dict[str, float]
    versions: dict[str, str]
    artifacts: dict[str, str]
    slice_output: str


# --------------------------------------------------------------------------
# Artifact loading and authentication
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_artifacts() -> Artifacts:
    """Load the saved model and fitted preprocessing once and reuse them."""
    return load_artifacts(MODEL_DIR)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Fail fast at startup when the trained artifacts are missing."""
    get_artifacts()
    yield


ArtifactsDep = Annotated[Artifacts, Depends(get_artifacts)]

api_key_header = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    auto_error=False,
    description=(
        "API key for the protected routes. The development placeholder "
        "'census-dev-key' is accepted unless CENSUS_API_KEY is set."
    ),
)


def expected_api_key() -> str:
    """Return the configured API key, read at request time."""
    return os.environ.get(API_KEY_ENV, DEFAULT_DEV_API_KEY)


def require_api_key(
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> str:
    """Reject requests that lack a valid ``X-API-Key`` header."""
    provided = (api_key or "").encode("utf-8")
    expected = expected_api_key().encode("utf-8")
    if not api_key or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


# --------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------
def payloads_to_frame(items: Sequence[CensusInput]) -> pd.DataFrame:
    """Build a DataFrame in training column order from request models."""
    records = [item.model_dump(by_alias=True) for item in items]
    return pd.DataFrame(records, columns=FEATURE_COLUMNS)


def predictions_for(
    items: Sequence[CensusInput], artifacts: Artifacts
) -> list[PredictionResponse]:
    """Score request models with the saved model and preprocessing."""
    labels, probabilities = predict_labels(
        artifacts, payloads_to_frame(items)
    )
    return [
        PredictionResponse(prediction=label, probability_over_50k=prob)
        for label, prob in zip(labels, probabilities, strict=True)
    ]


# --------------------------------------------------------------------------
# Application and routes
# --------------------------------------------------------------------------
app = FastAPI(
    title="Census Income Prediction API",
    version="1.0.0",
    description=(
        "Predicts whether a person's annual income exceeds 50K from Census "
        "features using a trained HistGradientBoosting classifier. "
        "GET / and POST /predict are public; POST /predict/batch and "
        "GET /model/info require the X-API-Key header."
    ),
    lifespan=lifespan,
)

PREDICT_EXAMPLES = {
    "high_income": {
        "summary": "Likely to earn more than 50K",
        "value": HIGH_INCOME_EXAMPLE,
    },
    "low_income": {
        "summary": "Likely to earn 50K or less",
        "value": LOW_INCOME_EXAMPLE,
    },
}


@app.get("/", summary="Welcome", tags=["General"])
def read_root() -> MessageResponse:
    """Return a welcome message and the main routes of this API."""
    return MessageResponse(
        message=WELCOME_MESSAGE,
        docs="/docs",
        predict="/predict",
        ui="/ui",
        health="/health",
    )


@app.get("/health", summary="Liveness check", tags=["General"])
def health(artifacts: ArtifactsDep) -> HealthResponse:
    """Confirm the API is up and which model type is loaded."""
    return HealthResponse(
        status="ok", model_type=artifacts.metadata["model_type"]
    )


@app.get("/features", summary="Accepted features", tags=["Model"])
def features(artifacts: ArtifactsDep) -> FeaturesResponse:
    """List the features, the categories seen in training and examples."""
    categories = {
        feature: [str(value) for value in values]
        for feature, values in zip(
            CAT_FEATURES, artifacts.encoder.categories_, strict=True
        )
    }
    return FeaturesResponse(
        feature_columns=list(FEATURE_COLUMNS),
        continuous_features=list(CONTINUOUS_FEATURES),
        categorical_features=list(CAT_FEATURES),
        categories=categories,
        examples=EXAMPLES,
    )


@app.post(
    "/predict",
    summary="Predict whether income exceeds 50K",
    tags=["Inference"],
)
def predict(
    payload: Annotated[CensusInput, Body(openapi_examples=PREDICT_EXAMPLES)],
    artifacts: ArtifactsDep,
) -> PredictionResponse:
    """Run the trained model on one record.

    Hyphenated Census names (``education-num``, ``marital-status``,
    ``capital-gain``, ``capital-loss``, ``hours-per-week``,
    ``native-country``) are accepted as sent; snake_case field names work
    too. Categories unseen during training are encoded as all zeros.
    """
    return predictions_for([payload], artifacts)[0]


@app.post(
    "/predict/batch",
    summary="Predict several records (API key required)",
    tags=["Inference"],
    dependencies=[Security(require_api_key)],
)
def predict_batch(
    payload: BatchPredictionRequest, artifacts: ArtifactsDep
) -> BatchPredictionResponse:
    """Score up to 1000 records in one request."""
    predictions = predictions_for(payload.instances, artifacts)
    return BatchPredictionResponse(
        predictions=predictions, count=len(predictions)
    )


@app.get(
    "/model/info",
    summary="Training metadata (API key required)",
    tags=["Model"],
    dependencies=[Security(require_api_key)],
)
def model_info(artifacts: ArtifactsDep) -> ModelInfo:
    """Return the metadata saved with the trained artifacts."""
    return ModelInfo(**artifacts.metadata)


@app.get("/ui", summary="Web user interface", tags=["Frontend"])
def ui() -> HTMLResponse:
    """Serve the single-page frontend that calls ``POST /predict``."""
    if not FRONTEND_FILE.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend file not found",
        )
    return HTMLResponse(FRONTEND_FILE.read_text(encoding="utf-8"))
