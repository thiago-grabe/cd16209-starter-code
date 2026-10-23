"""Tests for the static web UI served at /ui."""

import main
from starter.ml.data import FEATURE_COLUMNS


def test_ui_serves_html(client):
    response = client.get("/ui")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    text = response.text
    assert "<form" in text
    assert "/predict" in text
    assert "/features" in text
    for name in FEATURE_COLUMNS:
        assert f'name="{name}"' in text


def test_ui_returns_404_when_file_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "FRONTEND_FILE", tmp_path / "missing.html")
    response = client.get("/ui")
    assert response.status_code == 404
    assert response.json()["detail"] == "Frontend file not found"
