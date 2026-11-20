"""Training loop for the SimpleCNN."""

import logging

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.training.config import (
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    NUM_EPOCHS,
    RANDOM_STATE,
    SCHEDULER_PATIENCE,
    THRESHOLD,
)
from src.training.model import SimpleCNN

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


def validate_epoch(model, loader, criterion, device, threshold=THRESHOLD):
    """One pass over the validation data without updating the model.

    Returns (loss, accuracy, f1, f1_macro). Macro F1 averages both classes,
    so a model that predicts "defective" for everything scores low on it.
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device).view(-1, 1)

            logits = model(images)
            running_loss += criterion(logits, labels).item()
            predicted = (torch.sigmoid(logits) >= threshold).float().view(-1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return (
        running_loss / len(loader),
        float(accuracy),
        f1_score(all_labels, all_preds, average="binary"),
        f1_score(all_labels, all_preds, average="macro"),
    )


def train_cnn_model(
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    device=None,
    random_seed=RANDOM_STATE,
):
    """Train a SimpleCNN and return (model with the best weights, history).

    The best epoch is the one with the highest validation macro F1. The
    learning rate is halved when macro F1 plateaus, and training stops early
    after EARLY_STOP_PATIENCE epochs without improvement. Metrics are logged
    to MLflow when a run is active.
    """
    device = device or get_device()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model = SimpleCNN().to(device)
    criterion = make_criterion(train_loader.dataset.labels, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE,
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "train_f1": [], "val_f1": [], "val_f1_macro": [],
        "best_val_f1_macro": 0.0, "best_model_state": None,
    }
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc, val_f1, val_f1_macro = validate_epoch(
            model, val_loader, criterion, device,
        )

        for key, value in (
            ("train_loss", train_loss), ("val_loss", val_loss),
            ("train_accuracy", train_acc), ("val_accuracy", val_acc),
            ("train_f1", train_f1), ("val_f1", val_f1), ("val_f1_macro", val_f1_macro),
        ):
            history[key].append(value)

        scheduler.step(val_f1_macro)

        if val_f1_macro > history["best_val_f1_macro"]:
            history["best_val_f1_macro"] = val_f1_macro
            # Copy the weights so the saved best state does not keep changing
            history["best_model_state"] = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                logger.info(
                    "Early stopping: no macro F1 improvement for %d epochs", EARLY_STOP_PATIENCE,
                )
                break

        logger.info(
            "Epoch %d/%d (lr=%.6f): train loss %.4f acc %.4f f1 %.4f | "
            "val loss %.4f acc %.4f f1 %.4f macro f1 %.4f",
            epoch + 1, num_epochs, optimizer.param_groups[0]["lr"],
            train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, val_f1_macro,
        )

        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "train_f1_score": train_f1,
                    "val_f1_score": val_f1,
                    "val_f1_macro": val_f1_macro,
                },
                step=epoch,
            )

    if history["best_model_state"] is not None:
        model.load_state_dict(history["best_model_state"])
    logger.info("Best validation macro F1: %.4f", history["best_val_f1_macro"])
    return model, history
