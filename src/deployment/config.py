"""Settings for the deployment step: model selection, promotion and export.

Paths are relative to the project root, same as in the training pipeline.
"""

from pathlib import Path

MODEL_NAME = "cnn_model"
EXPERIMENT_NAME = "vial_image_classification"
# Name the training run logs the pyfunc model under
ARTIFACT_PATH = "model_pyfunc"

PRODUCTION_ALIAS = "Production"

ARTIFACT_DIR = Path("artifacts")
MLFLOW_STATE_DIR = Path("mlflow-state")
MODEL_RESULTS_PATH = ARTIFACT_DIR / "model_results.json"

# The exported model ends up here; this is what gets baked into the serving image
EXPORT_DIR = Path("models/model")
