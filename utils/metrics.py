"""
Evaluation metrics for segmentation models.
"""
import torch
import torch.nn.functional as F


def calc_iou(pred, masks):
    """
    Calculate Intersection over Union (IoU) for segmentation predictions.

    Args:
        pred (torch.Tensor): Model predictions (logits) [B, C, H, W]
        masks (torch.Tensor): Ground truth masks [B, 1, H, W]

    Returns:
        torch.Tensor: IoU score for the foreground class
    """
    pred = F.softmax(pred, dim=1)
    pred_masks = (pred > 0.5).float()

    # Calculate intersection and union for the foreground (class 1)
    foreground = 1.0  # our foreground label

    # Create boolean masks for the foreground
    pred_foreground = (pred_masks == foreground)
    true_foreground = (masks == foreground)

    # Compute intersection and union
    intersection = (pred_foreground[:, 1, :, :] & true_foreground).sum().float()
    union = (pred_foreground[:, 1, :, :] | true_foreground).sum().float()

    # Calculate IoU, adding a small epsilon to avoid division by zero
    iou_foreground = (intersection + 1e-8) / (union + 1e-8)
    return iou_foreground


def get_evaluation_metrics(outputs_logits, masks):
    """
    Calculate multiple evaluation metrics.

    Args:
        outputs_logits (torch.Tensor): Model predictions (logits)
        masks (torch.Tensor): Ground truth masks

    Returns:
        dict: Dictionary containing various metrics
    """
    # Currently we only have IoU, but more metrics can be added here
    iou = calc_iou(outputs_logits, masks)

    return {
        "iou": iou.item()
    }