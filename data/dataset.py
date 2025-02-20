"""
Dataset class for road segmentation.
"""
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import config


class RoadSegmentationDataset(Dataset):
    """
    Dataset class for road segmentation task.
    """

    def __init__(self, split, csv_file, data_parent_dir=None, transform=None):
        """
        Initialize the dataset.

        Args:
            split (str): Data split ('train', 'train_val', 'test')
            csv_file (str): Path to the CSV file with annotations
            data_parent_dir (str): Parent directory containing the images
            transform: Optional transform to be applied to samples
        """
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["split"] == split]
        self.data_parent_dir = data_parent_dir or config.DATA_ROOT
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (image_path, image_tensor, mask_tensor)
        """
        img_path = os.path.join(self.data_parent_dir, self.data.iloc[idx]['sat_image_path'])
        mask_path = os.path.join(self.data_parent_dir, self.data.iloc[idx]['mask_path'])

        # Read image and mask
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Ensure mask is properly formatted
        mask = mask.permute(2, 0, 1) / 255

        return img_path, image.float(), mask.float()


def get_data_loaders(train_transform, valid_transform):
    """
    Create and return data loaders for training and validation.

    Args:
        train_transform: Transforms to apply to training data
        valid_transform: Transforms to apply to validation data

    Returns:
        tuple: (train_loader, valid_loader)
    """
    # Initialize datasets
    train_dataset = RoadSegmentationDataset(
        split="train",
        csv_file=config.TRAIN_SPLIT_CSV,
        transform=train_transform
    )

    valid_dataset = RoadSegmentationDataset(
        split="train_val",
        csv_file=config.VAL_SPLIT_CSV,
        transform=valid_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    return train_loader, valid_loader