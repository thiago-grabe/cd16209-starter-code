"""Data loading, cleaning and preprocessing for the Census income model."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

LABEL = "salary"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

CONTINUOUS_FEATURES = [
    "age",
    "fnlgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

# Column order of data/census.csv without the label. process_data keeps the
# continuous columns in DataFrame order, so every caller must use this order.
FEATURE_COLUMNS = [
    "age",
    "workclass",
    "fnlgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip incidental whitespace from column names and text values.

    Feature names (including their hyphens) and the meaning of every value
    are preserved; only leading and trailing whitespace is removed. A
    cleaned copy is returned and the input frame is left untouched.
    """
    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    text_columns = cleaned.select_dtypes(include=["object", "string"])
    for column in text_columns.columns:
        cleaned[column] = cleaned[column].str.strip()
    return cleaned


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the Census CSV, clean whitespace and order the columns.

    Raises ``ValueError`` when an expected column is missing so that a wrong
    file is reported by name instead of failing later inside the encoder.
    """
    frame = clean_frame(pd.read_csv(path, skipinitialspace=True))
    expected = FEATURE_COLUMNS + [LABEL]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in {path}: {', '.join(missing)}"
        )
    return frame[expected]


def process_data(
    X, categorical_features=[], label=None, training=True,
    encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features
    and a label binarizer for the labels. This can be used in either training
    or inference/validation.

    Note: depending on the type of model used, you may want to add in
    functionality that scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array
        will be returned for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns
        the encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns
        the binarizer passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(categorical_features, axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
