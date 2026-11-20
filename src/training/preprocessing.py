"""Image preprocessing for training: transforms, dataset, split and dataloaders."""

import logging

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.training.config import IMAGE_SIZE, RANDOM_STATE, TRANSFORM_MEAN, TRANSFORM_STD

logger = logging.getLogger(__name__)


def define_transforms():
    """Return (train_transforms, val_transforms).

    Images are resized from the native camera resolution to IMAGE_SIZE and
    normalized with the ImageNet statistics. Only the training transforms
    apply augmentation (flip, rotation, colour jitter).
    """
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
    ])

    return train_transforms, val_transforms


class GlassVialDataset(Dataset):
    """Loads vial images as RGB and their labels, and applies the transform."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            image = img.convert("RGB") if img.mode != "RGB" else img.copy()

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


def split_dataset(image_dataset, train_ratio=0.70, val_ratio=0.15, random_seed=RANDOM_STATE):
    """Stratified split into (train, val, test) lists of (image_path, label).

    Each class is shuffled and split on its own, so every set keeps the same
    class balance. Whatever is left after train and val goes to test.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    class_0 = [(path, label) for path, label in image_dataset if label == 0]
    class_1 = [(path, label) for path, label in image_dataset if label == 1]
    np.random.shuffle(class_0)
    np.random.shuffle(class_1)

    def split_class(data):
        train_size = int(len(data) * train_ratio)
        val_size = int(len(data) * val_ratio)
        return (
            data[:train_size],
            data[train_size:train_size + val_size],
            data[train_size + val_size:],
        )

    train_0, val_0, test_0 = split_class(class_0)
    train_1, val_1, test_1 = split_class(class_1)

    train_dataset = train_0 + train_1
    val_dataset = val_0 + val_1
    test_dataset = test_0 + test_1
    np.random.shuffle(train_dataset)
    np.random.shuffle(val_dataset)
    np.random.shuffle(test_dataset)

    for name, subset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        labels = [label for _, label in subset]
        logger.info(
            "%s: %d images (class 0: %d, class 1: %d)",
            name, len(subset), labels.count(0), labels.count(1),
        )
    return train_dataset, val_dataset, test_dataset
