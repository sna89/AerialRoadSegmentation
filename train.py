"""
Training script for road segmentation model.
"""
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.metrics import calc_iou
from utils.visualization import log_predictions_to_tensorboard


def train_epoch(model, loader, optimizer, criterion_list, device, epoch, writer):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Model to train
        loader (DataLoader): DataLoader for training data
        optimizer (Optimizer): Optimizer for training
        criterion_list (list): List of loss functions
        device (torch.device): Device to train on
        epoch (int): Current epoch number
        writer (SummaryWriter): TensorBoard writer

    Returns:
        tuple: (average_loss, average_iou)
    """
    model.train()
    epoch_loss = 0
    iou_score = 0
    num_batches = len(loader)

    for i, (paths, images, masks) in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs_logits = model(images)

        # Take the second channel (road class) for binary prediction
        outputs = outputs_logits[:, 1].unsqueeze(1)

        loss = 0
        for criterion in criterion_list:
            loss += criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        batch_iou = calc_iou(outputs_logits, masks)

        # Get current loss and IoU values
        batch_loss = loss.item()
        batch_iou = batch_iou.item()

        # Log to TensorBoard at each iteration
        current_iter = epoch * num_batches + i
        writer.add_scalar('Iteration/train_loss', batch_loss, current_iter)
        writer.add_scalar('Iteration/train_iou', batch_iou, current_iter)

        # Accumulate metrics
        epoch_loss += batch_loss
        iou_score += batch_iou

        # Log visualizations periodically
        if i % 100 == 0:
            log_predictions_to_tensorboard(
                writer, images, masks, outputs_logits, 'Train', current_iter
            )

            print(f'Epoch {epoch}, Iter {i}/{num_batches}, '
                  f'Loss: {batch_loss:.4f}, IoU: {batch_iou:.4f}')

    return epoch_loss / num_batches, iou_score / num_batches

