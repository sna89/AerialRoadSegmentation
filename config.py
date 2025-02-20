"""
Configuration parameters for the road segmentation project.
"""
import os
from datetime import datetime

# Paths
DATA_ROOT = "/data/"
METADATA_CSV = os.path.join(DATA_ROOT, "metadata.csv")
TRAIN_SPLIT_CSV = os.path.join(DATA_ROOT, "train_split.csv")
VAL_SPLIT_CSV = os.path.join(DATA_ROOT, "val_split.csv")

# Training parameters
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Model parameters
MODEL_NAME = "resnet34"
ENCODER_PRETRAINED = None
DROPOUT = 0.2
NUM_CLASSES = 2

# Experiment tracking
EXPERIMENT_NAME = f"road_segmentation_{MODEL_NAME}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
TENSORBOARD_DIR = os.path.join("runs", "road_segmentation_experiment", MODEL_NAME,
                             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
MODEL_SAVE_PATH = f"checkpoints/best_model_{MODEL_NAME}.pth"

# Create a dictionary of hyperparameters for logging
HPARAMS = {
    "BatchSize": BATCH_SIZE,
    "LearningRate": LEARNING_RATE,
    "Epochs": NUM_EPOCHS,
    "Model": MODEL_NAME,
    "Dropout": DROPOUT,
    "WeightDecay": WEIGHT_DECAY,
    "EncoderPretrained": ENCODER_PRETRAINED
}