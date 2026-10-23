"""Tests for data loading, cleaning and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from starter.ml.data import (
    CAT_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_COLUMNS,
    LABEL,
    clean_frame,
    load_data,
    process_data,
)
from starter.ml.model import predict_labels


def test_load_data_cleans_whitespace_and_keeps_names(census_df):
    assert list(census_df.columns) == FEATURE_COLUMNS + [LABEL]
    assert census_df.shape == (32561, 15)
    for column in census_df.select_dtypes(include="object").columns:
        assert (census_df[column] == census_df[column].str.strip()).all()
    assert "education-num" in census_df.columns
    assert "?" in set(census_df["workclass"])
    assert set(census_df[LABEL]) == {"<=50K", ">50K"}
    assert all(
        dtype.kind == "i" for dtype in census_df[CONTINUOUS_FEATURES].dtypes
    )
    assert not census_df.isna().any().any()


def test_clean_frame_strips_text_columns_only():
    frame = pd.DataFrame(
        {" workclass": [" State-gov", "Private ", " ? "], "age": [39, 50, 38]}
    )
    cleaned = clean_frame(frame)
    assert list(cleaned.columns) == ["workclass", "age"]
    assert cleaned["workclass"].tolist() == ["State-gov", "Private", "?"]
    assert cleaned["age"].tolist() == [39, 50, 38]
    assert cleaned["age"].dtype.kind == "i"
    # The input frame is left untouched.
    assert list(frame.columns) == [" workclass", "age"]
    assert frame[" workclass"].tolist() == [" State-gov", "Private ", " ? "]


def test_load_data_rejects_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("age, workclass\n39, Private\n", encoding="utf-8")
    with pytest.raises(ValueError, match="salary"):
        load_data(path)


def test_process_data_training_fits_encoder_and_binarizer(
    small_df, processed_small
):
    X, y, encoder, lb = processed_small
    n_categories = sum(len(values) for values in encoder.categories_)
    assert X.shape == (len(small_df), len(CONTINUOUS_FEATURES) + n_categories)
    assert set(np.unique(y)) <= {0, 1}
    assert list(lb.classes_) == ["<=50K", ">50K"]
    assert X.dtype.kind == "f"


def test_process_data_inference_reuses_fitted_objects(
    small_df, processed_small
):
    X_train, _, encoder, lb = processed_small
    X_inf, y_inf, encoder_out, lb_out = process_data(
        small_df,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    assert encoder_out is encoder
    assert lb_out is lb
    assert np.array_equal(X_inf, X_train)
    assert y_inf.shape == (len(small_df),)

    unseen = small_df.head(1).drop(columns=[LABEL]).copy()
    unseen["native-country"] = "Atlantis"
    X_unseen, y_unseen, _, _ = process_data(
        unseen,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    # The last block of one-hot columns belongs to native-country.
    n_country = len(encoder.categories_[-1])
    assert X_unseen.shape == (1, X_train.shape[1])
    assert not X_unseen[0, -n_country:].any()
    assert y_unseen.size == 0


def test_process_data_output_depends_on_column_order(small_df, artifacts):
    features = small_df.drop(columns=[LABEL]).head(5)
    reversed_frame = features[list(reversed(FEATURE_COLUMNS))]
    kwargs = {
        "categorical_features": CAT_FEATURES,
        "label": None,
        "training": False,
        "encoder": artifacts.encoder,
        "lb": artifacts.lb,
    }
    X_ordered, _, _, _ = process_data(features, **kwargs)
    X_reversed, _, _, _ = process_data(reversed_frame, **kwargs)
    assert not np.array_equal(X_ordered, X_reversed)
    # predict_labels pins the column order, so the result is unchanged.
    assert predict_labels(artifacts, reversed_frame) == predict_labels(
        artifacts, features
    )
