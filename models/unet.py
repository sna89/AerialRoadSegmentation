"""
U-Net model definition for road segmentation.
"""
import segmentation_models_pytorch as smp
import torch.nn as nn
import config


def get_model():
    """
    Create and initialize a U-Net model for road segmentation.

    Returns:
        nn.Module: Initialized U-Net model
    """
    model = smp.Unet(
        encoder_name=config.MODEL_NAME,
        encoder_weights=config.ENCODER_PRETRAINED,
        in_channels=3,  # RGB input
        classes=config.NUM_CLASSES,  # Background and road
        dropout=config.DROPOUT
    )

    return model


def get_loss_functions():
    """
    Get the loss functions for training.

    Returns:
        list: List of loss function instances
    """
    return [smp.losses.DiceLoss(mode='binary')]


def get_optimizer(model):
    """
    Create an optimizer for the model.

    Args:
        model (nn.Module): Model to optimize

    Returns:
        torch.optim.Optimizer: Optimizer instance
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )