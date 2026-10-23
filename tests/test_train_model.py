"""Tests for the training script."""

import json

import pandas as pd
import pytest

import main as api
from starter import train_model as training
from starter.examples import HIGH_INCOME_EXAMPLE, LOW_INCOME_EXAMPLE
from starter.ml.data import CAT_FEATURES, FEATURE_COLUMNS
from starter.ml.model import load_artifacts, predict_labels
from starter.ml.slices import SLICE_LINE_RE


def run_small_training(small_csv, out_dir):
    return training.main([
        "--data", str(small_csv),
        "--model-dir", str(out_dir / "model"),
        "--slice-output", str(out_dir / "slice_output.txt"),
    ])


def metric_lines(path):
    text = path.read_text(encoding="utf-8")
    return [line for line in text.splitlines() if not line.startswith("#")]


def test_parse_args_defaults():
    args = training.parse_args([])
    root = training.PROJECT_ROOT
    assert args.data == root / "data" / "census.csv"
    assert args.model_dir == root / "model"
    assert args.slice_output == root / "slice_output.txt"
    assert args.test_size == 0.2
    assert args.random_state == 42
    # The API loads from the same project-relative directory.
    assert api.MODEL_DIR == root / "model"


def test_main_trains_and_writes_all_outputs_on_subset(
    small_csv, tmp_path, capsys
):
    assert run_small_training(small_csv, tmp_path) == 0
    out = capsys.readouterr().out
    assert "precision=" in out
    assert "| F1 |" in out

    model_dir = tmp_path / "model"
    for name in ("model.pkl", "encoder.pkl", "lb.pkl", "metadata.json"):
        assert (model_dir / name).is_file()
    metadata = json.loads(
        (model_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["n_rows"] == 1500
    assert metadata["n_train"] + metadata["n_test"] == 1500
    assert set(metadata["metrics"]) == {"precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in metadata["metrics"].values())
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert metadata["slice_output"] == "slice_output.txt"
    assert metadata["model_type"] == "HistGradientBoostingClassifier"
    assert metadata["hyperparameters"]["random_state"] == 42

    lines = metric_lines(tmp_path / "slice_output.txt")
    matches = [SLICE_LINE_RE.fullmatch(line) for line in lines]
    assert matches
    assert all(matches)
    assert {match.group(1) for match in matches} == set(CAT_FEATURES)

    artifacts = load_artifacts(model_dir)
    frame = pd.DataFrame([HIGH_INCOME_EXAMPLE, LOW_INCOME_EXAMPLE])
    labels, probabilities = predict_labels(artifacts, frame)
    assert set(labels) <= {"<=50K", ">50K"}
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_training_is_reproducible(small_csv, tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    assert run_small_training(small_csv, first_dir) == 0
    assert run_small_training(small_csv, second_dir) == 0
    first = load_artifacts(first_dir / "model")
    second = load_artifacts(second_dir / "model")
    assert first.metadata["metrics"] == second.metadata["metrics"]
    frame = pd.DataFrame([HIGH_INCOME_EXAMPLE, LOW_INCOME_EXAMPLE])
    _, first_probabilities = predict_labels(first, frame)
    _, second_probabilities = predict_labels(second, frame)
    assert first_probabilities == pytest.approx(
        second_probabilities, abs=1e-9
    )
    assert metric_lines(first_dir / "slice_output.txt") == metric_lines(
        second_dir / "slice_output.txt"
    )


def test_main_returns_1_when_data_missing(tmp_path, capsys):
    missing = tmp_path / "nope.csv"
    status = training.main([
        "--data", str(missing), "--model-dir", str(tmp_path / "model"),
    ])
    assert status == 1
    assert str(missing) in capsys.readouterr().err
    assert not (tmp_path / "model").exists()
