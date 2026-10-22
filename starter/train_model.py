"""Train the Census income classifier and save its inference artifacts.

Run from the project root::

    python -m starter.train_model

The script loads and cleans ``data/census.csv``, holds out a stratified
20% evaluation split, fits the preprocessing on the training split only,
trains the model, reports held-out precision, recall and F1, writes
per-slice metrics for every categorical feature to ``slice_output.txt``
and saves the model plus fitted preprocessing under ``model/``.
"""

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from starter.ml.data import (
    CAT_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_COLUMNS,
    LABEL,
    load_data,
    process_data,
)
from starter.ml.model import (
    ARTIFACT_FILES,
    compute_model_metrics,
    inference,
    save_artifacts,
    train_model,
)
from starter.ml.slices import compute_all_slices, write_slice_output

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "census.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_SLICE_OUTPUT = PROJECT_ROOT / "slice_output.txt"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line options for training."""
    parser = argparse.ArgumentParser(
        description="Train the Census income classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV file with the Census data",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="directory that receives the model and preprocessing files",
    )
    parser.add_argument(
        "--slice-output",
        type=Path,
        default=DEFAULT_SLICE_OUTPUT,
        help="text file that receives the per-slice metrics",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="fraction of rows held out for evaluation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="seed for the split and the model",
    )
    return parser.parse_args(argv)


def build_metadata(
    model: Any,
    lb: Any,
    metrics: tuple[float, float, float],
    counts: dict[str, int],
    test_size: float,
    random_state: int,
    slice_output_name: str,
) -> dict[str, Any]:
    """Collect what the API, the model card and the tests need to know."""
    precision, recall, fbeta = metrics
    # Round-trip through JSON so every hyperparameter is a plain type.
    hyperparameters = json.loads(json.dumps(model.get_params(), default=str))
    return {
        "model_type": type(model).__name__,
        "hyperparameters": hyperparameters,
        "label": LABEL,
        "label_classes": [str(cls) for cls in lb.classes_],
        "positive_label": str(lb.classes_[1]),
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_features": list(CAT_FEATURES),
        "continuous_features": list(CONTINUOUS_FEATURES),
        "n_features_processed": int(counts["n_features_processed"]),
        "n_rows": int(counts["n_rows"]),
        "n_train": int(counts["n_train"]),
        "n_test": int(counts["n_test"]),
        "split": {
            "test_size": test_size,
            "random_state": random_state,
            "stratify": LABEL,
        },
        "metrics": {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(fbeta), 4),
        },
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "artifacts": {
            "model": ARTIFACT_FILES["model"],
            "encoder": ARTIFACT_FILES["encoder"],
            "label_binarizer": ARTIFACT_FILES["lb"],
        },
        "slice_output": slice_output_name,
    }


def run_training(
    data_path: Path = DEFAULT_DATA_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
    slice_output: Path = DEFAULT_SLICE_OUTPUT,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Train, evaluate and save the model; return the metadata written."""
    data = load_data(data_path)
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[LABEL],
    )

    # Preprocessing is fitted on the training split only; the held-out
    # split is transformed with the fitted encoder and label binarizer.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train, random_state=random_state)
    preds = inference(model, X_test)
    metrics = compute_model_metrics(y_test, preds)
    precision, recall, fbeta = metrics

    slice_rows = compute_all_slices(test, y_test, preds, CAT_FEATURES)
    header = [
        f"Slice metrics on the held-out evaluation split "
        f"(n_test={len(test)}, test_size={test_size}, "
        f"random_state={random_state}, stratified by {LABEL})",
        f"Overall held-out metrics: precision={precision:.4f} "
        f"recall={recall:.4f} f1={fbeta:.4f}",
        "Format: feature=<name> value=<category> n=<rows> "
        "precision=<p> recall=<r> f1=<f1>",
    ]
    write_slice_output(slice_output, slice_rows, header)

    counts = {
        "n_features_processed": X_train.shape[1],
        "n_rows": len(data),
        "n_train": len(train),
        "n_test": len(test),
    }
    metadata = build_metadata(
        model,
        lb,
        metrics,
        counts,
        test_size,
        random_state,
        Path(slice_output).name,
    )
    save_artifacts(model_dir, model, encoder, lb, metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    """Command line entry point; returns the process exit code."""
    args = parse_args(argv)
    if not args.data.is_file():
        print(f"Data file not found: {args.data}", file=sys.stderr)
        return 1
    metadata = run_training(
        data_path=args.data,
        model_dir=args.model_dir,
        slice_output=args.slice_output,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    metrics = metadata["metrics"]
    print(
        f"rows={metadata['n_rows']} n_train={metadata['n_train']} "
        f"n_test={metadata['n_test']} "
        f"n_features_processed={metadata['n_features_processed']}"
    )
    print(
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
    )
    print("Markdown rows for model_card.md:")
    print(f"| Precision | {metrics['precision']:.4f} |")
    print(f"| Recall | {metrics['recall']:.4f} |")
    print(f"| F1 | {metrics['f1']:.4f} |")
    print(f"Artifacts written to {args.model_dir}")
    print(f"Slice metrics written to {args.slice_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
