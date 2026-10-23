"""Tests for the FastAPI application using the real trained artifacts."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
from starter.examples import HIGH_INCOME_EXAMPLE, LOW_INCOME_EXAMPLE
from starter.ml.data import FEATURE_COLUMNS
from starter.ml.model import predict_labels

ALIASES = [
    "education-num",
    "marital-status",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]


def test_get_root_returns_welcome_message(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == main.WELCOME_MESSAGE
    assert body["message"].startswith("Welcome")
    assert body["predict"] == "/predict"
    assert body["docs"] == "/docs"


def test_get_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "model_type": "HistGradientBoostingClassifier",
    }


def test_post_predict_high_income(client, high_income_payload):
    response = client.post("/predict", json=high_income_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == ">50K"
    assert 0.5 < body["probability_over_50k"] <= 1.0


def test_post_predict_low_income(client, low_income_payload):
    response = client.post("/predict", json=low_income_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "<=50K"
    assert 0.0 <= body["probability_over_50k"] < 0.5


@pytest.mark.parametrize(
    "payload", [HIGH_INCOME_EXAMPLE, LOW_INCOME_EXAMPLE], ids=["high", "low"]
)
def test_post_predict_matches_direct_inference(client, artifacts, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    labels, probabilities = predict_labels(artifacts, pd.DataFrame([payload]))
    body = response.json()
    assert body["prediction"] == labels[0]
    assert body["probability_over_50k"] == pytest.approx(
        probabilities[0], abs=1e-9
    )


def test_post_predict_accepts_field_names(client, high_income_payload):
    aliased = client.post("/predict", json=high_income_payload).json()
    snake_case = {
        key.replace("-", "_"): value
        for key, value in high_income_payload.items()
    }
    response = client.post("/predict", json=snake_case)
    assert response.status_code == 200
    assert response.json() == aliased


def test_post_predict_strips_whitespace(client, high_income_payload):
    clean = client.post("/predict", json=high_income_payload).json()
    padded = dict(
        high_income_payload, workclass=" Private", education="Bachelors "
    )
    response = client.post("/predict", json=padded)
    assert response.status_code == 200
    assert response.json() == clean


def test_post_predict_missing_field_returns_422(client, high_income_payload):
    del high_income_payload["education-num"]
    response = client.post("/predict", json=high_income_payload)
    assert response.status_code == 422
    assert "education-num" in response.json()["detail"][0]["loc"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", "forty"),
        ("hours-per-week", 0),
        ("education-num", 17),
        ("fnlgt", 10**30),
    ],
)
def test_post_predict_wrong_type_and_bounds_return_422(
    client, high_income_payload, field, value
):
    high_income_payload[field] = value
    response = client.post("/predict", json=high_income_payload)
    assert response.status_code == 422
    assert field in response.json()["detail"][0]["loc"]


def test_post_predict_ignores_extra_label_key(client, high_income_payload):
    """A pasted CSV row still works because the salary key is ignored."""
    high_income_payload["salary"] = ">50K"
    response = client.post("/predict", json=high_income_payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"


def test_get_features_is_public(client, artifacts):
    response = client.get("/features")
    assert response.status_code == 200
    body = response.json()
    assert body["feature_columns"] == FEATURE_COLUMNS
    assert body["categories"]["sex"] == ["Female", "Male"]
    assert "?" in body["categories"]["workclass"]
    assert body["examples"]["high_income"] == HIGH_INCOME_EXAMPLE
    assert len(body["categories"]) == len(artifacts.encoder.categories_)


def test_openapi_has_aliases_examples_and_security(client):
    schema = client.get("/openapi.json").json()
    post = schema["paths"]["/predict"]["post"]
    request_body = post["requestBody"]["content"]["application/json"]
    component_name = request_body["schema"]["$ref"].rsplit("/", 1)[-1]
    component = schema["components"]["schemas"][component_name]
    assert set(ALIASES) <= set(component["properties"])
    assert set(ALIASES) <= set(component["required"])
    assert not any("_" in name for name in component["properties"])
    assert component["examples"] == [HIGH_INCOME_EXAMPLE, LOW_INCOME_EXAMPLE]
    examples = request_body["examples"]
    assert examples["high_income"]["value"] == HIGH_INCOME_EXAMPLE
    assert examples["low_income"]["value"] == LOW_INCOME_EXAMPLE
    security = schema["components"]["securitySchemes"]["APIKeyHeader"]
    assert security["name"] == "X-API-Key"


def test_startup_fails_fast_without_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_DIR", tmp_path)
    main.get_artifacts.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="train_model"):
            with TestClient(main.app):
                pass
    finally:
        main.get_artifacts.cache_clear()
