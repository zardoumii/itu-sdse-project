"""The served model: an MLflow pyfunc wrapper around the trained CNN.

It accepts image paths, raw bytes or base64 strings, applies the validation
transforms, returns predictions with confidences, and appends every request
to an inference log that the serving side's drift check reads.
"""

import base64
import json
import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import torch
from PIL import Image

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

    def preprocess_image(self, image_input):
        """Turn one image into a (1, 3, 64, 64) tensor.

        Accepts a file path (str or Path), raw bytes, or a base64 string (the
        portable format for containerized serving). A str is used as a path if
        that file exists, otherwise decoded as base64.
        """
        if isinstance(image_input, bytes):
            image = Image.open(BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, str):
            try:
                is_file = Path(image_input).exists()
            except OSError:  # e.g. a base64 string too long to be a file name
                is_file = False
            if is_file:
                image = Image.open(image_input).convert("RGB")
            else:
                try:
                    image = Image.open(BytesIO(base64.b64decode(image_input))).convert("RGB")
                except Exception as e:
                    raise ValueError(
                        f"Cannot process string input '{image_input[:50]}...': "
                        f"not an existing file path and not valid base64 image data: {e}"
                    )
        elif isinstance(image_input, Path):
            if not image_input.exists():
                raise FileNotFoundError(f"File not found: {image_input}")
            image = Image.open(image_input).convert("RGB")
        else:
            raise TypeError(
                f"Unsupported image input type: {type(image_input)}. Expected str, Path, or bytes."
            )

        return self.transform(image).unsqueeze(0)

    def _extract_inputs(self, model_input):
        """Normalize the accepted request shapes into a flat list of images.

        Accepts {"instances": [...]}, {"data": [...]}, any other dict with one
        image or list value, a list/tuple/array, or a single image. Nested
        lists are flattened one level.
        """
        if isinstance(model_input, dict):
            if "instances" in model_input:
                image_inputs = model_input["instances"]
            elif "data" in model_input:
                image_inputs = model_input["data"]
            else:
                for value in model_input.values():
                    if isinstance(value, (list, str, np.ndarray, bytes, Path)):
                        image_inputs = [value] if isinstance(value, (str, Path, np.ndarray, bytes)) else value
                        break
                else:
                    raise ValueError(f"Could not extract image inputs from input: {model_input}")
        elif isinstance(model_input, (list, tuple, np.ndarray)):
            image_inputs = model_input
        else:
            image_inputs = [model_input]

        if isinstance(image_inputs, np.ndarray):
            image_inputs = image_inputs.tolist()

        flat = []
        for item in image_inputs:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    def predict(self, context, model_input):
        """Predict every image in the request and log each one.

        Returns predictions (1 = defective), confidences and raw outputs
        (sigmoid probability, rounded to 4 decimals) and the number of
        samples. An image that fails gets None instead of failing the request.
        """
        if self.model is None:
            self.load_context(context)

        image_inputs = self._extract_inputs(model_input)
        predictions, confidences, raw_outputs = [], [], []
        timestamp = self._get_timestamp_iso()

        for image_input in image_inputs:
            try:
                image_tensor = self.preprocess_image(image_input).to(self.device)
                with torch.no_grad():
                    prob = torch.sigmoid(self.model(image_tensor)).item()
                pred = 1 if prob >= self.threshold else 0

                predictions.append(pred)
                confidences.append(round(float(prob), 4))
                raw_outputs.append(round(float(prob), 4))

                self._log_request({
                    "timestamp": timestamp,
                    "image_path": "<bytes>" if isinstance(image_input, bytes) else str(image_input),
                    "prediction": pred,
                    "confidence": float(prob),
                    "raw_output": float(prob),
                })
            except Exception as e:
                logger.warning("Error processing image %s: %s", str(image_input)[:80], e)
                predictions.append(None)
                confidences.append(None)
                raw_outputs.append(None)

        return {
            "predictions": predictions,
            "confidences": confidences,
            "raw_outputs": raw_outputs,
            "num_samples": len(image_inputs),
        }
