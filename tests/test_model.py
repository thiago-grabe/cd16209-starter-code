"""Tests for training, inference, metrics and artifact persistence."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted

from starter.examples import HIGH_INCOME_EXAMPLE, LOW_INCOME_EXAMPLE
from starter.ml.data import CAT_FEATURES, CONTINUOUS_FEATURES, FEATURE_COLUMNS
from starter.ml.model import (
    ARTIFACT_FILES,
    METADATA_FILE,
    Artifacts,
    compute_model_metrics,
    inference,
    load_artifacts,
    predict_labels,
    save_artifacts,
    train_model,
)


@pytest.fixture(scope="module")
def small_model(processed_small):
    X, y, _, _ = processed_small
    return train_model(X, y)


def minimal_metadata(**overrides):
    metadata = {
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_features": list(CAT_FEATURES),
        "versions": {"scikit_learn": sklearn.__version__},
    }
    metadata.update(overrides)
    return metadata


def test_train_model_returns_fitted_classifier(small_model, processed_small):
    X, y, _, _ = processed_small
    assert isinstance(small_model, HistGradientBoostingClassifier)
    check_is_fitted(small_model)
    assert small_model.random_state == 42
    assert small_model.early_stopping is False
    assert small_model.n_iter_ == 200
    assert list(small_model.classes_) == [0, 1]
    accuracy = (small_model.predict(X) == y).mean()
    majority_rate = max(y.mean(), 1 - y.mean())
    assert accuracy > majority_rate


def test_train_model_is_deterministic(processed_small):
    X, y, _, _ = processed_small
    first = train_model(X, y).predict_proba(X)
    second = train_model(X, y).predict_proba(X)
    assert np.array_equal(first, second)


def test_inference_returns_binary_predictions(small_model, processed_small):
    X, _, _, _ = processed_small
    preds = inference(small_model, X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)) <= {0, 1}
    assert np.array_equal(preds, small_model.predict(X))


def test_compute_model_metrics_known_values():
    perfect = np.array([1, 0, 1, 0])
    assert compute_model_metrics(perfect, perfect) == (1.0, 1.0, 1.0)

    precision, recall, fbeta = compute_model_metrics(
        np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])
    )
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(0.5)

    # zero_division=1: no positives predicted or present counts as 1.0.
    zeros = np.array([0, 0])
    assert compute_model_metrics(zeros, zeros) == (1.0, 1.0, 1.0)
    precision, recall, _ = compute_model_metrics(
        np.array([1, 0, 1]), np.array([0, 0, 0])
    )
    assert precision == 1.0
    assert recall == 0.0


def test_save_and_load_artifacts_roundtrip(
    tmp_path, small_model, processed_small
):
    X, _, encoder, lb = processed_small
    metadata = minimal_metadata(metrics={"f1": 0.5})
    model_dir = tmp_path / "model"
    written = save_artifacts(model_dir, small_model, encoder, lb, metadata)
    assert set(written) == {"model", "encoder", "lb", "metadata"}
    for path in written.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    expected_names = set(ARTIFACT_FILES.values()) | {METADATA_FILE}
    assert {path.name for path in model_dir.iterdir()} == expected_names

    loaded = load_artifacts(model_dir)
    assert isinstance(loaded, Artifacts)
    assert loaded.metadata == metadata
    assert np.array_equal(loaded.model.predict(X), small_model.predict(X))
    for original, restored in zip(
        encoder.categories_, loaded.encoder.categories_, strict=True
    ):
        assert list(original) == list(restored)
    assert list(loaded.lb.classes_) == list(lb.classes_)


def test_load_artifacts_missing_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.pkl") as info:
        load_artifacts(tmp_path)
    assert "train_model" in str(info.value)

    (tmp_path / "model.pkl").write_bytes(b"x")
    (tmp_path / "lb.pkl").write_bytes(b"x")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r": encoder\.pkl\."):
        load_artifacts(tmp_path)


def test_load_artifacts_rejects_mismatched_feature_columns(
    tmp_path, small_model, processed_small
):
    _, _, encoder, lb = processed_small
    metadata = minimal_metadata(
        feature_columns=list(reversed(FEATURE_COLUMNS))
    )
    save_artifacts(tmp_path, small_model, encoder, lb, metadata)
    with pytest.raises(ValueError, match="feature_columns"):
        load_artifacts(tmp_path)


def test_load_artifacts_rejects_mismatched_categorical_features(
    tmp_path, small_model, processed_small
):
    _, _, encoder, lb = processed_small
    metadata = minimal_metadata(categorical_features=CAT_FEATURES[:-1])
    save_artifacts(tmp_path, small_model, encoder, lb, metadata)
    with pytest.raises(ValueError, match="categorical_features"):
        load_artifacts(tmp_path)


def test_load_artifacts_warns_on_sklearn_version_mismatch(
    tmp_path, small_model, processed_small
):
    _, _, encoder, lb = processed_small
    metadata = minimal_metadata(versions={"scikit_learn": "0.0.0"})
    save_artifacts(tmp_path, small_model, encoder, lb, metadata)
    with pytest.warns(UserWarning, match="0.0.0"):
        loaded = load_artifacts(tmp_path)
    assert loaded.metadata["versions"]["scikit_learn"] == "0.0.0"


def test_predict_labels_returns_expected_classes(artifacts):
    frame = pd.DataFrame([HIGH_INCOME_EXAMPLE, LOW_INCOME_EXAMPLE])
    labels, probabilities = predict_labels(artifacts, frame)
    assert labels == [">50K", "<=50K"]
    assert probabilities[0] > 0.5 > probabilities[1]
    assert all(type(label) is str for label in labels)
    assert all(type(probability) is float for probability in probabilities)


def test_predict_labels_rejects_missing_column(artifacts):
    frame = pd.DataFrame([HIGH_INCOME_EXAMPLE]).drop(columns=["age"])
    with pytest.raises(ValueError, match="age"):
        predict_labels(artifacts, frame)


def test_predict_labels_accepts_unknown_category_silently(artifacts):
    row = dict(HIGH_INCOME_EXAMPLE, workclass="Mars")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        labels, probabilities = predict_labels(artifacts, pd.DataFrame([row]))
    assert labels[0] in {"<=50K", ">50K"}
    assert 0.0 <= probabilities[0] <= 1.0


def test_shipped_metadata_is_consistent(artifacts):
    metadata = artifacts.metadata
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert metadata["categorical_features"] == CAT_FEATURES
    assert metadata["continuous_features"] == CONTINUOUS_FEATURES
    assert len(artifacts.encoder.categories_) == len(CAT_FEATURES)
    n_categories = sum(len(values) for values in artifacts.encoder.categories_)
    n_features = n_categories + len(CONTINUOUS_FEATURES)
    assert artifacts.model.n_features_in_ == n_features
    assert metadata["n_features_processed"] == n_features
    assert metadata["n_train"] + metadata["n_test"] == metadata["n_rows"]
    assert metadata["n_rows"] == 32561
    assert set(metadata["metrics"]) == {"precision", "recall", "f1"}
    for value in metadata["metrics"].values():
        assert 0.0 <= value <= 1.0
    assert metadata["metrics"]["f1"] > 0.6
    assert metadata["label_classes"] == ["<=50K", ">50K"]
    assert metadata["positive_label"] == ">50K"
    for version in metadata["versions"].values():
        assert isinstance(version, str) and version
    dumped = json.dumps(metadata)
    assert "/Users/" not in dumped
    assert "C:\\" not in dumped
