import base64
import json
from datetime import datetime, timezone

import pytest
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


def test_health_loading_while_scoring_server_starts(monkeypatch):
    monkeypatch.setattr(serving_app, "scoring_server_health", lambda: None)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


def test_health_error_when_scoring_server_died(monkeypatch):
    class DeadProcess:
        def poll(self):
            return 1

    monkeypatch.setattr(serving_app, "scoring_proc", DeadProcess())
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "error"}


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


IMAGE_BYTES = b"\xff\xd8\xff\xe0 fake jpeg \x00\xfe"


@pytest.fixture
def scoring_calls(monkeypatch):
    """Replace the scoring server with one that records what it was sent."""
    calls = []

    def fake_invoke(payload):
        calls.append(payload)
        return 200, b'{"predictions": {"predictions": [1]}}', "application/json"

    monkeypatch.setattr(serving_app, "invoke_scoring_server", fake_invoke)
    return calls


@pytest.mark.parametrize("path", ["/invocations", "/infer"])
def test_raw_image_is_sent_as_base64(scoring_calls, path):
    response = client.post(path, content=IMAGE_BYTES, headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 200
    assert response.json() == {"predictions": {"predictions": [1]}}
    sent = json.loads(scoring_calls[0])
    assert base64.b64decode(sent["instances"][0]) == IMAGE_BYTES


def test_json_is_passed_through(scoring_calls):
    body = b'{"instances": ["abc"]}'
    client.post("/invocations", content=body, headers={"Content-Type": "application/json"})
    assert scoring_calls == [body]


def test_empty_body_is_rejected(scoring_calls):
    response = client.post("/invocations", content=b"", headers={"Content-Type": "image/jpeg"})
    assert response.status_code == 400
    assert scoring_calls == []


def test_scoring_server_down_returns_502(monkeypatch):
    def unreachable(payload):
        raise ConnectionError("Scoring server not reachable")

    monkeypatch.setattr(serving_app, "invoke_scoring_server", unreachable)
    response = client.post("/invocations", content=IMAGE_BYTES)
    assert response.status_code == 502
    assert response.json()["status"] == "error"


def test_drift_without_logs_does_not_crash(monkeypatch, tmp_path):
    # No inference log and no baseline in an empty directory
    monkeypatch.chdir(tmp_path)
    response = client.get("/drift")
    assert response.status_code == 200
    assert response.json()["num_samples"] == 0
    assert response.json()["overall_drift_detected"] is False


def test_drift_reads_inference_log(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts").mkdir()
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "prediction": 1, "confidence": 0.9}
    (tmp_path / "artifacts" / "inference_log.json").write_text(json.dumps([entry, entry]))
    body = client.get("/drift").json()
    assert body["num_samples"] == 2
    assert body["class_distribution"]["class_1_count"] == 2


def test_drift_rejects_days_below_one():
    assert client.get("/drift", params={"days": 0}).status_code == 422
