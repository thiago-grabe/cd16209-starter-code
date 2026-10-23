"""Consistency checks between documentation, manifest and artifacts."""

import json
import re
from pathlib import Path

import pytest

import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCKED_PARTS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".nox", ".cache", ".aws", ".ssh",
    ".azure", ".config", "node_modules", "credentials", "secrets",
}
REQUIRED_MANIFEST_ENTRIES = {
    "model/model.pkl",
    "model/encoder.pkl",
    "model/lb.pkl",
    "model/metadata.json",
    "frontend/index.html",
}
ALIASES = {
    "education-num", "marital-status", "capital-gain", "capital-loss",
    "hours-per-week", "native-country",
}


def manifest_entries():
    text = (PROJECT_ROOT / "submission_files.txt").read_text(encoding="utf-8")
    entries = [line.strip() for line in text.splitlines()]
    return [entry for entry in entries if entry and not entry.startswith("#")]


def test_submission_manifest_lists_artifacts_and_supporting_files():
    entries = manifest_entries()
    assert REQUIRED_MANIFEST_ENTRIES <= set(entries)
    for entry in entries:
        assert "\\" not in entry
        assert not entry.startswith("/")
        assert not set(entry.split("/")) & BLOCKED_PARTS
        if not entry.startswith("evidence/"):
            assert (PROJECT_ROOT / entry).is_file(), entry


def test_model_card_complete_and_consistent(artifacts):
    template = (PROJECT_ROOT / "model_card_template.md").read_text(
        encoding="utf-8"
    )
    card = (PROJECT_ROOT / "model_card.md").read_text(encoding="utf-8")
    headings = [
        line for line in template.splitlines() if line.startswith("## ")
    ]
    assert headings
    for heading in headings:
        assert heading in card
    assert "HistGradientBoostingClassifier" in card
    metadata = artifacts.metadata
    plain = card.replace(",", "")
    assert str(metadata["n_train"]) in plain
    assert str(metadata["n_test"]) in plain
    rows = (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))
    for key, label in rows:
        pattern = rf"\|\s*{label}\s*\|\s*([0-9]\.[0-9]+)\s*\|"
        match = re.search(pattern, card)
        assert match, f"{label} row missing from the metrics table"
        assert abs(float(match.group(1)) - metadata["metrics"][key]) <= 0.005


def test_evidence_openapi_matches_app_structure():
    path = PROJECT_ROOT / "evidence" / "openapi.json"
    if not path.is_file():
        pytest.skip("evidence/openapi.json has not been exported yet")
    exported = json.loads(path.read_text(encoding="utf-8"))
    live = main.app.openapi()
    assert set(exported["paths"]) == set(live["paths"])
    component = exported["components"]["schemas"]["CensusInput"]
    assert ALIASES <= set(component["properties"])
    assert len(component["examples"]) == 2
    assert "APIKeyHeader" in exported["components"]["securitySchemes"]


def test_readme_documents_artifacts_and_routes():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    needles = (
        "model/model.pkl",
        "model/encoder.pkl",
        "model/lb.pkl",
        "model/metadata.json",
        "load_artifacts",
        "/predict",
        "python -m uvicorn main:app --host 127.0.0.1 --port 8000",
    )
    for needle in needles:
        assert needle in readme, needle
