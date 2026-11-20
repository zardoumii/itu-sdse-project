"""Image preprocessing for training: transforms, dataset, split and dataloaders."""

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
