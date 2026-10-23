"""Tests for the API-key protected routes."""

import pytest
from fastapi import HTTPException

import main
from starter.examples import HIGH_INCOME_EXAMPLE, LOW_INCOME_EXAMPLE
from starter.ml.data import FEATURE_COLUMNS


def test_model_info_requires_api_key(client):
    response = client.get("/model/info")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "ApiKey"
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_model_info_rejects_wrong_key(client):
    response = client.get("/model/info", headers={"X-API-Key": "nope"})
    assert response.status_code == 401


def test_model_info_with_valid_key(client, api_key_headers, artifacts):
    response = client.get("/model/info", headers=api_key_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"] == artifacts.metadata["metrics"]
    assert body["feature_columns"] == FEATURE_COLUMNS
    assert body["model_type"] == "HistGradientBoostingClassifier"


def test_api_key_env_override(client, monkeypatch, api_key_headers):
    monkeypatch.setenv(main.API_KEY_ENV, "from-env")
    stale = client.get("/model/info", headers=api_key_headers)
    assert stale.status_code == 401
    response = client.get("/model/info", headers={"X-API-Key": "from-env"})
    assert response.status_code == 200


def test_non_ascii_key_is_rejected_not_500():
    with pytest.raises(HTTPException) as info:
        main.require_api_key("clé")
    assert info.value.status_code == 401


def test_require_api_key_returns_valid_key():
    key = main.DEFAULT_DEV_API_KEY
    assert main.require_api_key(key) == key


def test_batch_predict_returns_both_classes(client, api_key_headers):
    payload = {"instances": [HIGH_INCOME_EXAMPLE, LOW_INCOME_EXAMPLE]}
    response = client.post(
        "/predict/batch", json=payload, headers=api_key_headers
    )
    assert response.status_code == 200
    body = response.json()
    predictions = [item["prediction"] for item in body["predictions"]]
    assert predictions == [">50K", "<=50K"]
    assert body["count"] == 2


def test_batch_predict_requires_key_and_rejects_empty(
    client, api_key_headers
):
    payload = {"instances": [HIGH_INCOME_EXAMPLE]}
    assert client.post("/predict/batch", json=payload).status_code == 401
    response = client.post(
        "/predict/batch", json={"instances": []}, headers=api_key_headers
    )
    assert response.status_code == 422
