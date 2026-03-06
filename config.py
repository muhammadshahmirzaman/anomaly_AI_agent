"""
config.py — Centralized configuration for the Showroom Anomaly Detection Pipeline.

All hardcoded paths, hyperparameters, device settings, and thresholds are defined here.
Modify this file to customize the pipeline behavior.
"""

import os
import torch


# =============================================================================
# DEVICE CONFIGURATION — automatically selects GPU if available
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# PATH CONFIGURATION — hardcoded paths for training and inference
# =============================================================================
# INPUT: Place your normal showroom videos here for training
NORMAL_VIDEO_DIR = os.path.join(".", "data", "showroom_normal")

# INPUT: Place your abnormal showroom videos here for inference
ABNORMAL_VIDEO_DIR = os.path.join(".", "data", "showroom_abnormal")

# OUTPUT: Results (annotated videos, CSV files, model weights) are saved here
RESULTS_DIR = os.path.join(".", "results")

# Path where the trained anomaly model weights will be saved/loaded
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "anomaly_model.pth")

# Supported video file extensions
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")

# =============================================================================
# YOLOv8 DETECTION SETTINGS
# =============================================================================
# YOLOv8 model variant (n=nano, s=small, m=medium, l=large, x=extra-large)
YOLO_MODEL = "yolov8n.pt"

# Minimum confidence threshold for person detections
YOLO_CONFIDENCE = 0.5

# COCO class ID for "person"
PERSON_CLASS_ID = 0

# =============================================================================
# BYTETRACK TRACKER SETTINGS
# =============================================================================
# Tracking confidence threshold
TRACK_CONFIDENCE = 0.5

# Tracker type used by ultralytics
TRACKER_TYPE = "bytetrack.yaml"

# =============================================================================
# ANOMALY MODEL SETTINGS
# =============================================================================
# Input size for cropped person regions fed to the autoencoder (width, height)
CROP_SIZE = (64, 128)

# Number of input channels (3 for RGB)
INPUT_CHANNELS = 3

# Latent dimension of the autoencoder bottleneck
LATENT_DIM = 128

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
# Number of training epochs
NUM_EPOCHS = 50

# Batch size for training
BATCH_SIZE = 64

# Learning rate for the Adam optimizer
LEARNING_RATE = 1e-3

# How often to print training progress (every N batches)
LOG_INTERVAL = 10

# Frame sampling rate: extract every Nth frame during training to avoid redundancy
TRAIN_FRAME_SKIP = 5

# =============================================================================
# INFERENCE SETTINGS
# =============================================================================
# Anomaly score threshold — persons above this score are flagged as anomalous
ANOMALY_THRESHOLD = 0.02

# Frame sampling rate for inference (1 = process every frame)
INFER_FRAME_SKIP = 1

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================
# Bounding box colors (BGR format for OpenCV)
COLOR_NORMAL = (0, 255, 0)       # Green for normal persons
COLOR_ANOMALY = (0, 0, 255)      # Red for anomalous persons

# Bounding box thickness
BBOX_THICKNESS = 2

# Font scale for text overlays
FONT_SCALE = 0.5

# =============================================================================
# NORMALIZATION — ImageNet statistics used to normalize input crops
# =============================================================================
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def ensure_dirs():
    """Create required directories if they don't exist."""
    os.makedirs(NORMAL_VIDEO_DIR, exist_ok=True)
    os.makedirs(ABNORMAL_VIDEO_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
