"""FastAPI app in front of the MLflow scoring server.

The scoring server runs as a subprocess (see scoring_server.py); this app is
what gets exposed on port 5001.
"""

import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from src.serving.drift import (
    DEFAULT_DAYS,
    DRIFT_BASELINE_STATS_PATH,
    INFERENCE_LOG_PATH,
    detect_drift,
    load_baseline_stats,
    load_inference_logs,
)
from src.serving.scoring_server import (
    invoke_scoring_server,
    scoring_server_health,
    start_scoring_server,
    stop_scoring_server,
)

MODEL_PATH = Path("models/model")

# Set while the app is running (see lifespan)
scoring_proc = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scoring_proc
    scoring_proc = start_scoring_server(MODEL_PATH)
    yield
    stop_scoring_server(scoring_proc)
    scoring_proc = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if scoring_proc is not None and scoring_proc.poll() is not None:
        return JSONResponse({"status": "error"}, status_code=503)
    if scoring_server_health() == 200:
        return JSONResponse({"status": "ok"})
    # The model takes a while to load after startup
    return JSONResponse({"status": "loading"}, status_code=503)


@app.get("/version")
def version() -> dict:
    # version.json is written next to the exported model by the pipeline
    version_file = MODEL_PATH / "version.json"
    if version_file.exists():
        return json.loads(version_file.read_text())
    return {"model_version": "unknown", "trained_at": None}


def to_scoring_payload(body: bytes, content_type: str) -> bytes:
    """Turn a request body into the JSON the scoring server accepts.

    JSON is passed through unchanged. Anything else is treated as a raw
    image: the scoring server only takes JSON, so it is sent base64 encoded,
    which the model wrapper decodes.
    """
    if content_type.split(";")[0].strip().lower() == "application/json":
        return body
    return json.dumps({"instances": [base64.b64encode(body).decode("ascii")]}).encode()


@app.post("/invocations")
@app.post("/infer")
async def invocations(request: Request) -> Response:
    body = await request.body()
    if not body:
        return JSONResponse({"status": "error", "detail": "Request body is empty"}, status_code=400)

    payload = to_scoring_payload(body, request.headers.get("content-type", ""))
    try:
        # Run the blocking call in a thread so /health keeps answering during inference
        status, content, media_type = await run_in_threadpool(invoke_scoring_server, payload)
    except ConnectionError as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=502)
    return Response(content=content, status_code=status, media_type=media_type)


@app.get("/drift")
def drift(days: int = Query(default=DEFAULT_DAYS, ge=1)) -> dict:
    """Drift checks over the requests the model logged in the last `days` days."""
    logs = load_inference_logs(INFERENCE_LOG_PATH)
    baseline = load_baseline_stats(DRIFT_BASELINE_STATS_PATH)
    return detect_drift(logs, baseline, days=days)
