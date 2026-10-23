"""Tests for the slice evaluation helpers and the shipped slice file."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starter.ml.data import CAT_FEATURES, LABEL
from starter.ml.slices import (
    SLICE_LINE_RE,
    compute_all_slices,
    compute_slice_metrics,
    format_slice_line,
    write_slice_output,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLICE_OUTPUT = PROJECT_ROOT / "slice_output.txt"


def labels_and_shifted_preds(frame):
    y = (frame[LABEL] == ">50K").to_numpy().astype(int)
    return y, np.roll(y, 1)


def test_compute_slice_metrics_synthetic():
    frame = pd.DataFrame({"color": ["red", "red", "blue", "blue", "blue"]})
    y = np.array([1, 0, 1, 1, 0])
    preds = np.array([1, 1, 1, 0, 0])
    rows = compute_slice_metrics(frame, y, preds, "color")
    assert [row["value"] for row in rows] == ["blue", "red"]
    assert [row["n"] for row in rows] == [3, 2]
    blue, red = rows
    assert blue["precision"] == 1.0
    assert blue["recall"] == 0.5
    assert blue["f1"] == pytest.approx(2 / 3)
    assert red["precision"] == 0.5
    assert red["recall"] == 1.0
    assert all(row["feature"] == "color" for row in rows)


def test_compute_slice_metrics_covers_every_value_and_counts(small_df):
    y, preds = labels_and_shifted_preds(small_df)
    rows = compute_slice_metrics(small_df, y, preds, "education")
    assert {row["value"] for row in rows} == set(
        small_df["education"].astype(str)
    )
    assert sum(row["n"] for row in rows) == len(small_df)
    for row in rows:
        for key in ("precision", "recall", "f1"):
            assert 0.0 <= row[key] <= 1.0
        assert type(row["n"]) is int


def test_compute_slice_metrics_rejects_misaligned_inputs():
    frame = pd.DataFrame({"color": ["red", "blue"]})
    with pytest.raises(ValueError, match="same number of rows"):
        compute_slice_metrics(frame, np.array([1]), np.array([1, 0]), "color")
    with pytest.raises(ValueError, match="Unknown feature"):
        compute_slice_metrics(
            frame, np.array([1, 0]), np.array([1, 0]), "size"
        )


def test_compute_all_slices_keeps_feature_order(small_df):
    y, preds = labels_and_shifted_preds(small_df)
    rows = compute_all_slices(small_df, y, preds, ["sex", "race"])
    features = [row["feature"] for row in rows]
    assert features[0] == "sex"
    assert features[-1] == "race"
    assert features == sorted(features, key=["sex", "race"].index)
    expected = small_df["sex"].nunique() + small_df["race"].nunique()
    assert len(rows) == expected


def test_format_slice_line_exact():
    row = {
        "feature": "sex", "value": "Male", "n": 10,
        "precision": 0.5, "recall": 1.0, "f1": 2 / 3,
    }
    line = format_slice_line(row)
    assert line == (
        "feature=sex value=Male n=10 precision=0.5000 recall=1.0000 "
        "f1=0.6667"
    )
    assert SLICE_LINE_RE.fullmatch(line)


def test_write_slice_output_with_and_without_header(tmp_path):
    rows = [{
        "feature": "sex", "value": "Male", "n": 1,
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
    }]
    with_header = tmp_path / "nested" / "with_header.txt"
    write_slice_output(with_header, rows, ["first header", "second header"])
    text = with_header.read_text(encoding="utf-8")
    assert text.startswith("# first header\n# second header\n")
    assert text.endswith("\n")
    metric_lines = [
        line for line in text.splitlines() if not line.startswith("#")
    ]
    assert metric_lines == [format_slice_line(rows[0])]

    plain = tmp_path / "plain.txt"
    write_slice_output(plain, rows)
    assert plain.read_text(encoding="utf-8") == (
        format_slice_line(rows[0]) + "\n"
    )


def test_shipped_slice_output_is_complete(artifacts):
    text = SLICE_OUTPUT.read_text(encoding="utf-8")
    text.encode("ascii")
    lines = text.splitlines()
    headers = [line for line in lines if line.startswith("#")]
    matches = [
        SLICE_LINE_RE.fullmatch(line)
        for line in lines
        if not line.startswith("#")
    ]
    assert matches
    assert all(matches)
    assert {match.group(1) for match in matches} == set(CAT_FEATURES)
    counts = {}
    for match in matches:
        feature, n = match.group(1), int(match.group(3))
        counts[feature] = counts.get(feature, 0) + n
    assert set(counts.values()) == {artifacts.metadata["n_test"]}
    overall = [header for header in headers if header.startswith("# Overall")]
    assert len(overall) == 1
    metrics = artifacts.metadata["metrics"]
    assert f"precision={metrics['precision']:.4f}" in overall[0]
    assert f"f1={metrics['f1']:.4f}" in overall[0]
