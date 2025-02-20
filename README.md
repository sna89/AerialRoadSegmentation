# 🚀 Aerial Road Segmentation

![Road Segmentation Dataset Cover](/dataset_cover.png/)

## 📌 Overview
This project focuses on **road segmentation** using the [DeepGlobe Road Extraction Dataset](http://deepglobe.org/challenge.html) and leverages the `segmentation-models-pytorch (smp)` package to apply **deep learning-based semantic segmentation**.

The current implementation utilizes a **UNet** architecture with a **ResNet-34** encoder, optimized with **Dice Loss**. 
Various augmentations (rotations, flipping) and regularization techniques (dropout, weight decay) are applied to improve performance.

Model hyperparameters and configurations can be set up by updating the config file.
Training monitoring and evaluation is monitored using TensorBoard.

To run the project and train a model:
1. Install requirements.txt.
2. Download dataset.
3. Update configuration.
4. Run main file
