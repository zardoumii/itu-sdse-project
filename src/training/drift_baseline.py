"""Baseline statistics that the serving side compares live traffic against.

The model is run over the validation set once, and we keep what live data
can later be compared with: image statistics, class and prediction counts,
prediction confidences and the embeddings of the fc1 layer.
"""

import logging

import numpy as np
import torch

from src.training.config import THRESHOLD
from src.training.train import get_device

logger = logging.getLogger(__name__)


def collect_model_outputs(model, data_loader, device=None, feature_layer="fc1", threshold=THRESHOLD):
    """Run the model over data_loader and collect what the baseline is built from.

    Returns a dict with per-batch channel means/stds of the input images, the
    labels, predictions and confidences per image, and the per-batch outputs
    of feature_layer (captured with a forward hook).
    """
    device = device or get_device()
    if not hasattr(model, feature_layer):
        raise ValueError(f"Model does not have layer '{feature_layer}'")

    embeddings = []
    handle = getattr(model, feature_layer).register_forward_hook(
        lambda module, inputs, output: embeddings.append(output.detach().cpu().numpy())
    )

    outputs = {"image_means": [], "image_stds": [], "labels": [], "preds": [], "confidences": []}
    model.eval()
    try:
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                probs = torch.sigmoid(model(images)).cpu().numpy().flatten()

                outputs["image_means"].append(images.mean(dim=(0, 2, 3)).cpu().numpy())
                outputs["image_stds"].append(images.std(dim=(0, 2, 3)).cpu().numpy())
                outputs["labels"].extend(int(label) for label in labels.view(-1))
                outputs["preds"].extend((probs >= threshold).astype(int).tolist())
                outputs["confidences"].extend(probs)
    finally:
        handle.remove()

    outputs["embeddings"] = embeddings
    return outputs
