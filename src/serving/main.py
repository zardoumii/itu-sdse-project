"""Start the serving app: `python -m src.serving.main`.

Run it from the project root (or /app in the Docker image), where
models/model is. The app starts the MLflow scoring server itself.
"""

import uvicorn

from src.serving.app import app

HOST = "0.0.0.0"
# Must match targetPort and the probe ports in infra/azure/app.yaml
PORT = 5001


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
