"""Training loop for the SimpleCNN."""

import logging

import torch
import torch.nn as nn

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
