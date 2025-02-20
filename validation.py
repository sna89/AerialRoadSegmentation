import torch
from tqdm import tqdm
from utils.metrics import calc_iou
from utils.visualization import log_predictions_to_tensorboard


def validate(model, loader, criterion_list, device, epoch, writer):
    """
    Validate the model on the validation set.

    Args:
        model (nn.Module): Model to validate
        loader (DataLoader): DataLoader for validation data
        criterion_list (list): List of loss functions
        device (torch.device): Device to validate on
        epoch (int): Current epoch number
        writer (SummaryWriter): TensorBoard writer

    Returns:
        tuple: (average_loss, average_iou)
    """
    model.eval()

    val_loss = 0
    iou_score = 0
    num_batches = len(loader)

    with torch.no_grad():
        for i, (paths, images, masks) in enumerate(tqdm(loader, desc=f"Validation Epoch {epoch}")):
            images = images.to(device)
            masks = masks.to(device)

            outputs_logits = model(images)

            # Take the second channel (road class) for binary prediction
            outputs = outputs_logits[:, 1].unsqueeze(1)
            loss = 0

            for criterion in criterion_list:
                loss += criterion(outputs, masks)

            batch_iou = calc_iou(outputs_logits, masks)

            # Get current loss and IoU values
            batch_loss = loss.item()
            batch_iou = batch_iou.item()

            # Log to TensorBoard at each iteration
            current_iter = epoch * num_batches + i
            writer.add_scalar('Iteration/val_loss', batch_loss, current_iter)
            writer.add_scalar('Iteration/val_iou', batch_iou, current_iter)

            # Accumulate metrics
            val_loss += batch_loss
            iou_score += batch_iou

            # Log visualizations periodically
            if i % 100 == 0:
                log_predictions_to_tensorboard(
                    writer, images, masks, outputs_logits, 'Val', current_iter
                )

                print(f'Validation Epoch {epoch}, Iter {i}/{num_batches}, '
                      f'Loss: {batch_loss:.4f}, IoU: {batch_iou:.4f}')

    return val_loss / num_batches, iou_score / num_batches

