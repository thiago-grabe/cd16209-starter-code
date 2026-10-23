"""Model performance on slices of a categorical feature."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from starter.ml.model import compute_model_metrics

# Every metric line written by write_slice_output matches this pattern.
SLICE_LINE_RE = re.compile(
    r"^feature=(\S+) value=(\S+) n=(\d+) "
    r"precision=(\d\.\d{4}) recall=(\d\.\d{4}) f1=(\d\.\d{4})$"
)


def compute_slice_metrics(
    frame: pd.DataFrame,
    y: np.ndarray,
    preds: np.ndarray,
    feature: str,
) -> list[dict[str, Any]]:
    """Compute precision, recall and F1 for every value of one feature.

    ``frame`` holds the raw (unencoded) evaluation rows, ``y`` the binarized
    labels and ``preds`` the model predictions aligned with those rows. One
    dictionary per distinct value is returned, sorted by value.
    """
    if feature not in frame.columns:
        raise ValueError(f"Unknown feature: {feature}")
    y = np.asarray(y)
    preds = np.asarray(preds)
    if len(frame) != len(y) or len(y) != len(preds):
        raise ValueError(
            "frame, y and preds must have the same number of rows"
        )
    values = frame[feature].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for value in sorted(set(values)):
        mask = values == value
        precision, recall, fbeta = compute_model_metrics(
            y[mask], preds[mask]
        )
        rows.append(
            {
                "feature": feature,
                "value": value,
                "n": int(mask.sum()),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(fbeta),
            }
        )
    return rows


def compute_all_slices(
    frame: pd.DataFrame,
    y: np.ndarray,
    preds: np.ndarray,
    features: Sequence[str],
) -> list[dict[str, Any]]:
    """Run compute_slice_metrics for several features, in the given order."""
    rows: list[dict[str, Any]] = []
    for feature in features:
        rows.extend(compute_slice_metrics(frame, y, preds, feature))
    return rows


def format_slice_line(row: dict[str, Any]) -> str:
    """Render one slice as a self-describing single line of text."""
    return (
        f"feature={row['feature']} value={row['value']} n={row['n']} "
        f"precision={row['precision']:.4f} recall={row['recall']:.4f} "
        f"f1={row['f1']:.4f}"
    )


def write_slice_output(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
    header_lines: Sequence[str] = (),
) -> None:
    """Write header comments and one formatted line per slice to path."""
    lines = [f"# {line}" for line in header_lines]
    lines.extend(format_slice_line(row) for row in rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
