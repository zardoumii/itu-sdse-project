"""Training loop for the SimpleCNN."""

import logging

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.training.config import THRESHOLD

logger = logging.getLogger(__name__)


def get_device():
    """Use the GPU when there is one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_criterion(train_labels, device):
    """BCE-with-logits loss weighted for the class imbalance.

    Most images are defective (label 1), so the positive class is
    down-weighted by the negative/positive count ratio.
    """
    num_positive = int(sum(train_labels))
    num_negative = len(train_labels) - num_positive
    pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32, device=device)
    logger.info(
        "Training class counts: clean=%d, defective=%d -> pos_weight=%.4f",
        num_negative, num_positive, pos_weight.item(),
    )
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch(model, loader, criterion, optimizer, device, threshold=THRESHOLD):
    """One pass over the training data. Returns (loss, accuracy, f1)."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device).view(-1, 1)

        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (torch.sigmoid(logits) >= threshold).float().view(-1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return (
        running_loss / len(loader),
        float(accuracy),
        f1_score(all_labels, all_preds, average="binary"),
    )
