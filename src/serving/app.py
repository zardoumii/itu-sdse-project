import json
from pathlib import Path

from fastapi import FastAPI

MODEL_PATH = Path("models/model")

app = FastAPI()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict:
    # version.json is written next to the exported model by the pipeline
    version_file = MODEL_PATH / "version.json"
    if version_file.exists():
        return json.loads(version_file.read_text())
    return {"model_version": "unknown", "trained_at": None}
