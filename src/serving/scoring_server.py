"""Start and stop the MLflow scoring server for the exported model.

The scoring server runs as a subprocess on an internal port. The FastAPI app
in app.py sits in front of it and is the only thing exposed to the outside.
"""

import os
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = 5002


def start_scoring_server(model_path: Path, host: str = UPSTREAM_HOST, port: int = UPSTREAM_PORT) -> subprocess.Popen:
    env = os.environ.copy()
    # `mlflow models serve` runs `uvicorn` from PATH, so put this
    # interpreter's bin directory first
    bin_dir = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([bin_dir, env.get("PATH", "")])

    return subprocess.Popen(
        [
            sys.executable, "-m", "mlflow", "models", "serve",
            "--model-uri", str(model_path),
            "--host", host,
            "--port", str(port),
            "--workers", "1",
            "--env-manager", "local",
        ],
        env=env,
        # mlflow starts the real server as a child process. A new session
        # gives them their own process group, so stop_scoring_server can
        # stop both instead of leaving the child running.
        start_new_session=True,
    )


def stop_scoring_server(proc: subprocess.Popen, timeout: float = 10) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def scoring_server_health(host: str = UPSTREAM_HOST, port: int = UPSTREAM_PORT) -> int | None:
    """Status code of the scoring server's /health, or None if it is not reachable."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def invoke_scoring_server(
    payload: bytes, host: str = UPSTREAM_HOST, port: int = UPSTREAM_PORT, timeout: float = 300
) -> tuple[int, bytes, str]:
    """POST a JSON payload to the scoring server's /invocations.

    Returns (status code, body, content type). Error responses from the
    scoring server are returned as they are; ConnectionError is raised if
    it can't be reached at all.
    """
    request = urllib.request.Request(
        f"http://{host}:{port}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "application/json")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ConnectionError(f"Scoring server not reachable: {e}") from e
