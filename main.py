import os

import torch
from torch.utils.tensorboard import SummaryWriter

import config
from data.dataset import get_data_loaders
from data.transforms import get_training_transforms, get_validation_transforms
from models.unet import get_model, get_loss_functions, get_optimizer
from train import train_epoch
from validation import validate


def main():
    """Main training function."""
    # Create checkpoint directory
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    # Initialize tensorboard writer
    writer = SummaryWriter(config.TENSORBOARD_DIR)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get data transforms
    train_transform = get_training_transforms()
    valid_transform = get_validation_transforms()

    # Get data loaders
    train_loader, valid_loader = get_data_loaders(train_transform, valid_transform)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(valid_loader.dataset)}")

    # Initialize model
    model = get_model().to(device)
    criterion_list = get_loss_functions()
    optimizer = get_optimizer(model)

    # Training loop
    best_val_iou = 0

    for epoch in range(config.NUM_EPOCHS):
        print(f'Epoch {epoch + 1}/{config.NUM_EPOCHS}')

        # Training
        train_loss, train_iou = train_epoch(
            model, train_loader, optimizer, criterion_list, device, epoch, writer
        )

        # Validation
        val_loss, val_iou = validate(
            model, valid_loader, criterion_list, device, epoch, writer
        )

        # Log epoch metrics
        writer.add_scalar('Epoch/train_loss', train_loss, epoch)
        writer.add_scalar('Epoch/val_loss', val_loss, epoch)
        writer.add_scalar('Epoch/train_iou', train_iou, epoch)
        writer.add_scalar('Epoch/val_iou', val_iou, epoch)

        metric_log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_iou': train_iou,
            'val_iou': val_iou
        }

        # Log hyperparameters along with metrics
        writer.add_hparams(
            hparam_dict=config.HPARAMS,  # Hyperparameters
            metric_dict=metric_log  # Final metrics
        )

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
            }, config.MODEL_SAVE_PATH)
            print(f"Saved new best model with validation IoU: {val_iou:.4f}")

        print(f'Epoch {epoch + 1}/{config.NUM_EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')

    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()