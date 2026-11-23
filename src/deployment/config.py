"""Settings for the deployment step: model selection, promotion and export.

Names and paths match src/training/config.py. All paths are absolute and
resolved from the project root, so the code works no matter which directory
it is run from.
"""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "cnn_model"
EXPERIMENT_NAME = "vial_image_classification"
# Name the training run logs the pyfunc model under (PYFUNC_ARTIFACT_PATH in src/training/run.py)
PYFUNC_ARTIFACT_PATH = "model_pyfunc"

PRODUCTION_ALIAS = "Production"

ARTIFACT_DIR = PROJ_ROOT / "artifacts"
MLFLOW_STATE_DIR = PROJ_ROOT / "mlflow-state"
MODEL_RESULTS_PATH = ARTIFACT_DIR / "model_results.json"

# The exported model ends up here; this is what gets baked into the serving image
EXPORT_DIR = PROJ_ROOT / "models" / "model"
