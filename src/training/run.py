"""Run the whole training pipeline: `python -m src.training.run`.

Downloads nothing: run `dvc pull` and `python -m src.training.data` first.
"""

import json
import logging
import time
from pathlib import Path

import mlflow
import torch
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking import MlflowClient

from src.training.config import (
    ARTIFACT_DIR,
    CLASS_NAMES,
    CLASS_NAMES_PATH,
    IMAGE_STATISTICS_PATH,
    MODEL_NAME,
    MODEL_RESULTS_PATH,
    NUM_CLASSES,
    PROJ_ROOT,
    PYTORCH_MODEL_PATH,
)
from src.training.model import SimpleCNN
from src.training.wrapper import SimpleLoggingWrapper

logger = logging.getLogger(__name__)

# The deployment step (src/deployment) looks the model up under this name
PYFUNC_ARTIFACT_PATH = "model_pyfunc"


def save_training_artifacts(model, threshold, test_results, image_stats, artifact_dir=ARTIFACT_DIR):
    """Write the checkpoint, test results, class names and image statistics."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": SimpleCNN.__name__,
            "input_shape": (3, 64, 64),
            "output_shape": (1,),
            "threshold": threshold,
            "num_classes": NUM_CLASSES,
        },
        artifact_dir / PYTORCH_MODEL_PATH.name,
    )
    for path, content in (
        (MODEL_RESULTS_PATH, test_results),
        (CLASS_NAMES_PATH, CLASS_NAMES),
        (IMAGE_STATISTICS_PATH, image_stats),
    ):
        (artifact_dir / path.name).write_text(json.dumps(content, indent=2))
    logger.info("Training artifacts saved to %s", artifact_dir)


def log_pyfunc_model(model):
    """Log the CNN wrapped in SimpleLoggingWrapper as the run's model artifact.

    The wrapper keeps its default threshold (0.5), as in the notebook. The
    src/ package is bundled with the model (code_paths) so it can be loaded
    where src is not importable, e.g. in the serving container.
    """
    mlflow.pyfunc.log_model(
        PYFUNC_ARTIFACT_PATH,
        python_model=SimpleLoggingWrapper(model=model),
        code_paths=[str(PROJ_ROOT / "src")],
    )
    logger.info("Logged pyfunc model as %s", PYFUNC_ARTIFACT_PATH)


def wait_until_ready(model_name, model_version, max_attempts=10):
    """Wait until a registered model version is READY. Returns True if it is."""
    client = MlflowClient()
    for attempt in range(max_attempts):
        try:
            details = client.get_model_version(name=model_name, version=model_version)
            if ModelVersionStatus.from_string(details.status) == ModelVersionStatus.READY:
                return True
        except Exception as e:
            logger.warning("Checking model status failed (attempt %d/%d): %s", attempt + 1, max_attempts, e)
        time.sleep(1)
    logger.warning("Model %s version %s did not become READY", model_name, model_version)
    return False


def register_model(run_id, model_name=MODEL_NAME):
    """Register the run's pyfunc model, move it to Staging and alias it "best"."""
    model_details = mlflow.register_model(f"runs:/{run_id}/{PYFUNC_ARTIFACT_PATH}", model_name)
    wait_until_ready(model_details.name, model_details.version)

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_details.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(model_name, model_details.version, "deployed_to", "staging")
    client.set_model_version_tag(model_name, model_details.version, "task", "classification")
    client.set_registered_model_alias(model_name, "best", str(model_details.version))
    logger.info("Registered %s version %s (Staging, alias best)", model_name, model_details.version)
    return model_details
