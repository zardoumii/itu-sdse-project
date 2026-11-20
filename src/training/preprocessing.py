"""Image preprocessing for training: transforms, dataset, split and dataloaders."""

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.training.config import IMAGE_SIZE, TRANSFORM_MEAN, TRANSFORM_STD


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
