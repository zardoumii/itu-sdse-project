"""Send a sample image to the real app and check it gets a prediction.

This starts the real MLflow scoring server, so it needs an exported model in
models/model (`python -m src.training.run`, then `python -m src.deployment.run`).
It is skipped when there is no model, e.g. in CI before training has run.
"""

import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.serving import app as serving_app

MODEL_PATH = Path("models/model").resolve()

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH / "MLmodel").exists(), reason="no exported model in models/model"
)


def sample_image() -> bytes:
    # A first test image from data/test if it was downloaded, otherwise a
    # generated one at the camera's native size (1600x768)
    test_images = sorted(Path("data/test").glob("*.jpg"))
    if test_images:
        return test_images[0].read_bytes()

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1600, 768), color=(120, 130, 140)).save(buffer, "JPEG")
    return buffer.getvalue()


def test_sample_image_gets_prediction(monkeypatch, tmp_path):
    image = sample_image()
    monkeypatch.setattr(serving_app, "MODEL_PATH", MODEL_PATH)
    # The model writes its inference log relative to the working directory;
    # keep it out of the project
    monkeypatch.chdir(tmp_path)

    # Using the client as a context manager runs the lifespan, which starts
    # the real scoring server
    with TestClient(serving_app.app) as client:
        for _ in range(120):
            if client.get("/health").status_code == 200:
                break
            time.sleep(1)
        else:
            pytest.fail("scoring server did not become healthy within 120s")

        response = client.post("/invocations", content=image, headers={"Content-Type": "image/jpeg"})

    assert response.status_code == 200
    result = response.json()["predictions"]
    assert result["predictions"][0] in (0, 1)
    assert 0.0 <= result["confidences"][0] <= 1.0
