import json

from fastapi.testclient import TestClient

from src.serving import app as serving_app

# Not used as a context manager, so the app's lifespan doesn't start a real
# scoring server; the tests replace the scoring server calls instead.
client = TestClient(serving_app.app)


def test_health(monkeypatch):
    monkeypatch.setattr(serving_app, "scoring_server_health", lambda: 200)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(serving_app, "MODEL_PATH", tmp_path)
    response = client.get("/version")
    assert response.json() == {"model_version": "unknown", "trained_at": None}


def test_version_reads_version_file(monkeypatch, tmp_path):
    info = {"model_version": "2", "trained_at": "2026-11-01T10:00:00+00:00"}
    (tmp_path / "version.json").write_text(json.dumps(info))
    monkeypatch.setattr(serving_app, "MODEL_PATH", tmp_path)
    response = client.get("/version")
    assert response.json() == info
