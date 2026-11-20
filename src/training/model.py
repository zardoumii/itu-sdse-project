"""CNN that classifies glass vial images as clean or defective."""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """Two conv layers and a fully connected head for binary classification.

    Input: 64x64 RGB images (resized from the native 1600x768).
    Output: one logit per image (positive = defective, negative = clean).
    There is no sigmoid: training uses BCEWithLogitsLoss, and callers apply
    torch.sigmoid to get probabilities.

    Keep the layer names: saved checkpoints are keyed by them, and the drift
    baseline reads the fc1 embeddings.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Two 2x2 max pools: 64x64 -> 32x32 -> 16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
