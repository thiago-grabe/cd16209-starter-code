"""Model training, inference, metrics and artifact persistence."""

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from starter.ml.data import CAT_FEATURES, FEATURE_COLUMNS, process_data

DEFAULT_RANDOM_STATE = 42

# File names under the model directory; metadata.json completes the set.
ARTIFACT_FILES = {
    "model": "model.pkl",
    "encoder": "encoder.pkl",
    "lb": "lb.pkl",
}
METADATA_FILE = "metadata.json"


class Artifacts(NamedTuple):
    """Trained model plus the fitted preprocessing it was trained with."""

    model: HistGradientBoostingClassifier
    encoder: OneHotEncoder
    lb: LabelBinarizer
    metadata: dict[str, Any]


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> HistGradientBoostingClassifier:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    random_state : int
        Seed that makes the fit reproducible.
    Returns
    -------
    model : HistGradientBoostingClassifier
        Trained machine learning model.
    """
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(
    y: np.ndarray, preds: np.ndarray
) -> tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(
    model: HistGradientBoostingClassifier, X: np.ndarray
) -> np.ndarray:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : HistGradientBoostingClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def save_artifacts(
    model_dir: str | Path,
    model: HistGradientBoostingClassifier,
    encoder: OneHotEncoder,
    lb: LabelBinarizer,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Persist the model, fitted preprocessing and metadata under model_dir.

    Returns the path of every file written, keyed by artifact name.
    """
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    objects = {"model": model, "encoder": encoder, "lb": lb}
    written: dict[str, Path] = {}
    for name, obj in objects.items():
        path = directory / ARTIFACT_FILES[name]
        with path.open("wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        written[name] = path
    metadata_path = directory / METADATA_FILE
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written["metadata"] = metadata_path
    return written


def load_artifacts(model_dir: str | Path) -> Artifacts:
    """Load the trained model and fitted preprocessing saved by training.

    Nothing is retrained or refitted here. A ``FileNotFoundError`` lists
    every missing file; a ``ValueError`` reports metadata whose feature
    layout differs from the one this code expects.
    """
    directory = Path(model_dir)
    expected = [*ARTIFACT_FILES.values(), METADATA_FILE]
    missing = [name for name in expected if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts in {directory}: {', '.join(missing)}. "
            "Run: python -m starter.train_model"
        )
    loaded: dict[str, Any] = {}
    for name, filename in ARTIFACT_FILES.items():
        with (directory / filename).open("rb") as handle:
            loaded[name] = pickle.load(handle)
    metadata = json.loads(
        (directory / METADATA_FILE).read_text(encoding="utf-8")
    )
    if list(metadata.get("feature_columns", [])) != FEATURE_COLUMNS:
        raise ValueError(
            "metadata.json feature_columns do not match FEATURE_COLUMNS"
        )
    if list(metadata.get("categorical_features", [])) != CAT_FEATURES:
        raise ValueError(
            "metadata.json categorical_features do not match CAT_FEATURES"
        )
    trained_with = metadata.get("versions", {}).get("scikit_learn")
    if trained_with != sklearn.__version__:
        warnings.warn(
            f"Artifacts were trained with scikit-learn {trained_with} but "
            f"{sklearn.__version__} is installed; predictions may differ.",
            UserWarning,
            stacklevel=2,
        )
    return Artifacts(
        model=loaded["model"],
        encoder=loaded["encoder"],
        lb=loaded["lb"],
        metadata=metadata,
    )


def predict_labels(
    artifacts: Artifacts, frame: pd.DataFrame
) -> tuple[list[str], list[float]]:
    """Predict salary labels and P(>50K) for rows of raw Census features.

    The frame is reordered to the training column order before the fitted
    encoder transforms it. Categories unseen during training are encoded as
    all zeros by the fitted encoder (``handle_unknown="ignore"``), exactly
    as during evaluation.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    X, _, _, _ = process_data(
        frame[FEATURE_COLUMNS],
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=artifacts.encoder,
        lb=artifacts.lb,
    )
    preds = inference(artifacts.model, X)
    probabilities = artifacts.model.predict_proba(X)[:, 1]
    labels = artifacts.lb.inverse_transform(preds)
    return (
        [str(label) for label in labels],
        [float(probability) for probability in probabilities],
    )
