"""Baseline statistics that the serving side compares live traffic against.

The model is run over the validation set once, and we keep what live data
can later be compared with: image statistics, class and prediction counts,
prediction confidences and the embeddings of the fc1 layer.
"""

import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import torch

from src.training.config import DRIFT_BASELINE_STATS_PATH, THRESHOLD
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


def compute_baseline_statistics(model, data_loader, device=None, feature_layer="fc1", threshold=THRESHOLD):
    """Summarize the model outputs on data_loader into the drift baseline.

    The keys are read by the serving side (src/serving/drift.py), so keep
    the format unchanged.
    """
    outputs = collect_model_outputs(model, data_loader, device, feature_layer, threshold)
    image_means = np.array(outputs["image_means"])
    image_stds = np.array(outputs["image_stds"])
    labels, preds, confidences = outputs["labels"], outputs["preds"], outputs["confidences"]

    stats = {
        "image_statistics": {
            "mean_r": float(np.mean(image_means[:, 0])),
            "mean_g": float(np.mean(image_means[:, 1])),
            "mean_b": float(np.mean(image_means[:, 2])),
            "std_r": float(np.mean(image_stds[:, 0])),
            "std_g": float(np.mean(image_stds[:, 1])),
            "std_b": float(np.mean(image_stds[:, 2])),
        },
        "class_distribution": {
            "class_0_count": labels.count(0),
            "class_1_count": labels.count(1),
            "total": len(labels),
        },
        "prediction_distribution": {
            "pred_0_count": preds.count(0),
            "pred_1_count": preds.count(1),
            "total": len(preds),
        },
        "confidence_statistics": {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
        },
        "embedding_statistics": {"mean": [], "std": []},
    }

    if outputs["embeddings"]:
        embeddings = np.concatenate(outputs["embeddings"], axis=0)
        stats["embedding_statistics"] = {
            "mean": np.mean(embeddings, axis=0).tolist(),
            "std": np.std(embeddings, axis=0).tolist(),
            "shape": int(embeddings.shape[1]),
        }

    logger.info(
        "Drift baseline: classes 0=%d 1=%d, confidence mean %.4f",
        labels.count(0), labels.count(1), stats["confidence_statistics"]["mean"],
    )
    return stats


def save_baseline_statistics(stats, path=DRIFT_BASELINE_STATS_PATH):
    """Write the baseline to JSON, and log it to MLflow when a run is active."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))
    logger.info("Drift baseline saved to %s", path)

    if mlflow.active_run():
        image_stats = stats["image_statistics"]
        mlflow.log_metrics({
            **{f"baseline_image_{key}": value for key, value in image_stats.items()},
            "baseline_class_0_count": stats["class_distribution"]["class_0_count"],
            "baseline_class_1_count": stats["class_distribution"]["class_1_count"],
            "baseline_confidence_mean": stats["confidence_statistics"]["mean"],
            "baseline_confidence_std": stats["confidence_statistics"]["std"],
        })
        mlflow.log_artifact(str(path), "drift_detection")
