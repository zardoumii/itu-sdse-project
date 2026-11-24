# Vial defect classification – MLOps pipeline

Group project for ITU BDS MLOps (Data Science in Production), 2026.

A CNN classifies images of glass vials from a production line as clean or
defective. We are turning the original monolithic notebooks
(`notebooks/01_training.ipynb` and `notebooks/02_deployment.ipynb`) into a
modular Python project, with a Dagger workflow (Go) run by GitHub Actions that
trains the model and deploys it to DigitalOcean.

The original task description is in the course repository:
[lasselundstenjensen/itu-sdse-project](https://github.com/lasselundstenjensen/itu-sdse-project).

## Group members

- Erik Aymerich
- Benjamin
- Youssef

## Project structure

```text
.
├── data/                  # metadata csvs tracked with DVC; images are downloaded, not in git
│   ├── images/
│   └── test/
├── docker/                # Dockerfiles
├── docs/                  # architecture diagrams and dataset description
├── infra/azure/           # Azure Container Apps spec
├── notebooks/             # original notebooks, being split into src/
├── src/
│   ├── common/            # code shared by training and deployment (MLflow setup)
│   ├── training/          # training pipeline
│   ├── deployment/        # model selection, promotion and export
│   └── serving/           # inference service (/health, /invocations, /drift)
├── tests/
└── requirements.txt
```

## Serving

All commands are run from the project root. Serving needs an exported model
in `models/model`: run the training pipeline and then the deployment step
(`python -m src.deployment.run`), which writes the model there together with
`version.json` and `drift_baseline_stats.json`.

### Run locally

```bash
.venv/bin/pip install -r requirements-serving.txt
.venv/bin/python -m src.serving.main
```

This starts the API on port 5001. It runs the MLflow scoring server for the
model as a subprocess on port 5002, which is only reachable from inside the
app. Loading the model takes a few seconds; until then `/health` answers
`503 {"status": "loading"}`.

### Run with Docker

```bash
docker build -f docker/Dockerfile.model-serve -t model-serve .
docker run -p 5001:5001 model-serve
```

The image uses CPU-only PyTorch and contains `src/` and `models/model`. This
is the image that runs on Azure (`infra/azure/app.yaml`).

### Endpoints

| Endpoint | What it does |
| --- | --- |
| `GET /health` | `{"status": "ok"}` once the model is loaded, otherwise 503 |
| `GET /version` | version and training time of the served model |
| `POST /invocations` | prediction for one image |
| `POST /infer` | same as `/invocations` |
| `GET /drift?days=7` | drift checks over the requests of the last `days` days |

`/invocations` takes the raw image file as the request body. The JSON format
of the original notebook, `{"instances": ["<base64 image>"]}` with
`Content-Type: application/json`, still works too.

```bash
curl -X POST --data-binary @data/test/<image>.jpg \
  -H "Content-Type: image/jpeg" http://localhost:5001/invocations
```

```json
{"predictions": {"predictions": [1], "confidences": [0.9312], "raw_outputs": [0.9312], "num_samples": 1}}
```

A prediction of `1` means defective, `0` means clean. Input that is not a
readable image gets a `400`.

The model logs every request to `artifacts/inference_log.json`, and `/drift`
compares those requests with the baseline from training (two-sample KS tests
on the confidences and the predicted classes). The log lives inside the
container, so it starts empty after every deploy.

| Module | What it does |
| --- | --- |
| `src/serving/app.py` | the FastAPI app and its endpoints |
| `src/serving/scoring_server.py` | starts, checks and stops the MLflow scoring server |
| `src/serving/drift.py` | drift detection over the inference log |
| `src/serving/main.py` | starts the app with uvicorn on port 5001 |

### Tests

```bash
.venv/bin/python -m pytest tests/serving
```

`tests/serving/test_sample_image.py` sends an image to the real model and is
skipped when there is no exported model in `models/model`.
