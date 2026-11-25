import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.evaluate import evaluate_cnn_model

EXPECTED_KEYS = {
    "accuracy", "f1_score", "precision", "recall", "roc_auc",
    "confusion_matrix", "classification_report",
}


class PerfectModel(nn.Module):
    """Logit +5 for all-ones images, -5 for all-zeros images."""

    def forward(self, x):
        return (x.mean(dim=(1, 2, 3)) * 10 - 5).view(-1, 1)


def make_loader(labels):
    images = torch.stack([torch.full((3, 64, 64), float(label)) for label in labels])
    return DataLoader(TensorDataset(images, torch.tensor(labels)), batch_size=2)


def test_evaluate_returns_the_expected_metric_keys():
    results = evaluate_cnn_model(PerfectModel(), make_loader([0, 1, 0, 1]), device=torch.device("cpu"))
    assert set(results) == EXPECTED_KEYS


def test_evaluate_scores_a_perfect_model_as_perfect():
    results = evaluate_cnn_model(PerfectModel(), make_loader([0, 1, 1, 0, 1]), device=torch.device("cpu"))
    assert results["accuracy"] == results["f1_score"] == results["roc_auc"] == 1.0
    assert results["confusion_matrix"] == [[2, 0], [0, 3]]


def test_evaluate_handles_a_class_that_is_never_predicted():
    # Threshold above every probability: nothing is predicted defective
    results = evaluate_cnn_model(
        PerfectModel(), make_loader([0, 1, 0, 1]), device=torch.device("cpu"), threshold=1.0,
    )
    assert results["precision"] == 0.0 and results["recall"] == 0.0
    assert results["confusion_matrix"] == [[2, 0], [2, 0]]
