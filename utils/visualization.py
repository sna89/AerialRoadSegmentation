import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def show_image_and_mask(image_path, mask_path):
    """
    Display an image and its corresponding mask side by side.

    Args:
        image_path (str): Path to the image file
        mask_path (str): Path to the mask file
    """
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the image
    axes[0].imshow(image)
    axes[0].set_title('Satellite Image')
    axes[0].axis('off')

    # Display the mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Road Mask')
    axes[1].axis('off')

    # Adjust layout and display the images
    plt.tight_layout()
    plt.show()


def log_predictions_to_tensorboard(writer, images, masks, outputs_logits, prefix, current_iter):
    """
    Log images, true masks, and predicted masks to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer instance
        images (torch.Tensor): Input images
        masks (torch.Tensor): Ground truth masks
        outputs_logits (torch.Tensor): Model output logits
        prefix (str): Prefix for the TensorBoard tag
        current_iter (int): Current iteration number
    """
    # Normalize images for visualization
    image = images[:1]
    image = (image - image.min()) / (image.max() - image.min() + 0.0001)

    # Get prediction masks from logits
    pred_masks = (F.softmax(outputs_logits, dim=1) > 0.5).float()
    pred_masks = pred_masks[:, 1, :, :].unsqueeze(1)

    # Log to TensorBoard
    writer.add_images(f'{prefix}/Input', image, current_iter)
    writer.add_images(f'{prefix}/True_Mask', masks[:1], current_iter)
    writer.add_images(f'{prefix}/Pred_Mask', pred_masks, current_iter)

