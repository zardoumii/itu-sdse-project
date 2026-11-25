"""Test-set evaluation of the trained CNN."""

import logging

import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.training.config import THRESHOLD
from src.training.train import get_device

logger = logging.getLogger(__name__)


def evaluate_cnn_model(model, test_loader, device=None, threshold=THRESHOLD):
    """Return accuracy, F1, precision, recall, ROC-AUC, confusion matrix and
    classification report on the test set. Logged to MLflow when a run is active.

    zero_division=0 gives the same values sklearn falls back to when a class is
    never predicted, without the UndefinedMetricWarning.
    """
    device = device or get_device()
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            all_probs.extend(torch.sigmoid(logits).view(-1).cpu().numpy())
            all_labels.extend(labels.float().view(-1).numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= threshold).astype(float)

    results = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_score": f1_score(all_labels, all_preds, average="binary", zero_division=0),
        "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="binary", zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "classification_report": classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0,
        ),
    }

    logger.info(
        "Test (threshold %s): accuracy %.4f, F1 %.4f, precision %.4f, recall %.4f, ROC-AUC %.4f, "
        "confusion matrix [[TN, FP], [FN, TP]] = %s",
        threshold, results["accuracy"], results["f1_score"], results["precision"],
        results["recall"], results["roc_auc"], results["confusion_matrix"],
    )
    if mlflow.active_run():
        mlflow.log_metrics({
            "test_accuracy": results["accuracy"],
            "test_f1_score": results["f1_score"],
            "test_precision": results["precision"],
            "test_recall": results["recall"],
            "test_roc_auc": results["roc_auc"],
        })
    return results
