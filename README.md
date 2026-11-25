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

## Training

All commands are run from the project root.

### Setup

Requires Python 3.13.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Get the data

The metadata CSVs are tracked with DVC; the images listed in them are then
downloaded from the course object store (public, no credentials needed).

```bash
.venv/bin/dvc pull
.venv/bin/python -m src.training.data
```

### Train the model

```bash
.venv/bin/python -m src.training.run
```

This runs the whole training job in one MLflow run (about 20 seconds on a
laptop CPU): stratified 70/15/15 split, training with early stopping,
threshold tuning on the validation set, test-set evaluation and the drift
baseline. It produces:

- `artifacts/`: the model checkpoint (`vial_cnn_model.pth`),
  `model_results.json`, `drift_baseline_stats.json`, `class_names.json` and
  `image_statistics.json`
- `mlflow-state/`: the MLflow runs (SQLite) and the model artifact
  `model_pyfunc`, registered as `cnn_model` in Staging with the alias `best`

Both folders are generated output and are not committed.

| Module | What it does |
| --- | --- |
| `src/training/config.py` | paths and hyperparameters |
| `src/training/data.py` | downloads the images and loads `(image_path, label)` pairs |
| `src/training/preprocessing.py` | transforms, dataset, split, dataloaders, image statistics |
| `src/training/model.py` | the `SimpleCNN` model |
| `src/training/train.py` | training loop and threshold tuning |
| `src/training/evaluate.py` | test-set metrics |
| `src/training/drift_baseline.py` | statistics the serving drift check compares against |
| `src/training/wrapper.py` | `SimpleLoggingWrapper`, the MLflow model that is served |
| `src/training/run.py` | runs all of the above in order |

### Tests

```bash
.venv/bin/python -m pytest tests/training
```
