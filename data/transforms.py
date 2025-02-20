"""
Data augmentation transforms for the road segmentation dataset.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_transforms():
    """
    Returns the augmentation pipeline for training data.

    Returns:
        A.Compose: Composition of transformations to be applied
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])


def get_validation_transforms():
    """
    Returns the augmentation pipeline for validation data.

    Returns:
        A.Compose: Composition of transformations to be applied
    """
    return A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])