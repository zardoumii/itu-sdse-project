"""Run the whole training pipeline: `python -m src.training.run`.

Downloads nothing: run `dvc pull` and `python -m src.training.data` first.
"""

import json
import logging
from pathlib import Path

import torch

from src.training.config import (
    ARTIFACT_DIR,
    CLASS_NAMES,
    CLASS_NAMES_PATH,
    IMAGE_STATISTICS_PATH,
    MODEL_RESULTS_PATH,
    NUM_CLASSES,
    PYTORCH_MODEL_PATH,
)
from src.training.model import SimpleCNN

logger = logging.getLogger(__name__)


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
