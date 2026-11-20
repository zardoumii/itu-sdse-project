"""The served model: an MLflow pyfunc wrapper around the trained CNN.

It accepts image paths, raw bytes or base64 strings, applies the validation
transforms, returns predictions with confidences, and appends every request
to an inference log that the serving side's drift check reads.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import mlflow.pyfunc
import torch

from src.training.config import THRESHOLD
from src.training.model import SimpleCNN
from src.training.preprocessing import define_transforms

logger = logging.getLogger(__name__)

# Relative on purpose: the wrapper is pickled into the model and the path is
# resolved where the model is served. src/serving/drift.py reads this file.
DEFAULT_INFERENCE_LOG_PATH = Path("artifacts/inference_log.json")


class SimpleLoggingWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper that logs inference requests and returns predictions."""

    def __init__(self, model=None, model_path=None, log_path=DEFAULT_INFERENCE_LOG_PATH, threshold=THRESHOLD):
        self.model = model
        self.model_path = model_path
        self.log_path = Path(log_path)
        self.threshold = threshold
        # Same as the validation transforms: resize and normalize, no augmentation
        self.transform = define_transforms()[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def get_raw_model(self):
        """Return the underlying PyTorch model."""
        return self.model

    def load_context(self, context):
        """Called by MLflow when the model is loaded. Nothing to load when the
        CNN was pickled together with the wrapper."""
        if self.model is None:
            if self.model_path and Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.model = self._create_model_from_checkpoint(checkpoint)
            else:
                raise FileNotFoundError("Could not find model file")
        self.model.to(self.device)
        self.model.eval()
        logger.info("SimpleLoggingWrapper loaded on %s", self.device)

    def _create_model_from_checkpoint(self, checkpoint):
        architecture = checkpoint.get("architecture", "SimpleCNN")
        if architecture != "SimpleCNN":
            raise ValueError(f"Unknown architecture: {architecture}")
        model = SimpleCNN()
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        return model

    def _get_timestamp_iso(self):
        return datetime.now(timezone.utc).isoformat()

    def _log_request(self, request_data):
        """Append one entry to the JSON inference log. Never fails a prediction."""
        try:
            logs = []
            if self.log_path.exists():
                try:
                    logs = json.loads(self.log_path.read_text())
                except json.JSONDecodeError:
                    logs = []
                if not isinstance(logs, list):
                    logs = []
            logs.append(request_data)
            self.log_path.write_text(json.dumps(logs, indent=2))
        except Exception as e:
            logger.warning("Failed to log request: %s", e)
