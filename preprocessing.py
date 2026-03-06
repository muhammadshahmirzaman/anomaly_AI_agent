"""
preprocessing.py — Frame extraction, cropping, resizing, and normalization utilities.

This module handles all image preprocessing steps required by the pipeline:
1. Extract frames from video files using OpenCV.
2. Crop detected person regions from frames.
3. Resize crops to a fixed size for the anomaly model.
4. Normalize crops using ImageNet statistics.
5. Convert crops to PyTorch tensors ready for model input.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T

import config


# =============================================================================
# Define the torchvision transform pipeline for person crops
# =============================================================================
# This transform is applied to each cropped person region before feeding
# it to the anomaly autoencoder. It resizes, converts to tensor [0,1],
# and normalizes using ImageNet mean/std.
transform_pipeline = T.Compose([
    T.ToPILImage(),                                     # Convert numpy (H,W,C) to PIL Image
    T.Resize((config.CROP_SIZE[1], config.CROP_SIZE[0])),  # Resize to (height, width)
    T.ToTensor(),                                       # Convert to tensor [0, 1]
    T.Normalize(mean=config.NORMALIZE_MEAN,             # Normalize with ImageNet stats
                std=config.NORMALIZE_STD),
])

# Inverse normalization for reconstructed images (used in visualization if needed)
inverse_normalize = T.Compose([
    T.Normalize(
        mean=[-m / s for m, s in zip(config.NORMALIZE_MEAN, config.NORMALIZE_STD)],
        std=[1.0 / s for s in config.NORMALIZE_STD]
    ),
])


def extract_frames(video_path, frame_skip=1):
    """
    Generator that yields frames from a video file.

    Args:
        video_path (str): Path to the video file.
        frame_skip (int): Yield every Nth frame (1 = all frames).

    Yields:
        tuple: (frame_index, frame) where frame is a BGR numpy array (H, W, 3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Only yield every Nth frame based on the skip rate
        if frame_idx % frame_skip == 0:
            yield frame_idx, frame
        frame_idx += 1

    cap.release()


def get_video_properties(video_path):
    """
    Get video properties (FPS, width, height, total frames).

    Args:
        video_path (str): Path to the video file.

    Returns:
        dict: Dictionary with keys 'fps', 'width', 'height', 'total_frames'.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    props = {
        "fps": int(cap.get(cv2.CAP_PROP_FPS)) or 30,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return props


def crop_person(frame, bbox):
    """
    Crop a person region from the frame using the bounding box coordinates.

    Args:
        frame (np.ndarray): Full video frame (H, W, 3) in BGR.
        bbox (tuple): Bounding box as (x1, y1, x2, y2) in pixel coordinates.

    Returns:
        np.ndarray: Cropped person region in RGB format, or None if crop is invalid.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    # Clamp coordinates to frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Ensure valid crop dimensions
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    crop = frame[y1:y2, x1:x2]
    # Convert BGR (OpenCV) to RGB (for torchvision transforms)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop_rgb


def prepare_crop(crop_rgb):
    """
    Apply the full preprocessing pipeline to a cropped person region.

    Steps:
        1. Resize to the configured CROP_SIZE.
        2. Convert to PyTorch tensor [0, 1].
        3. Normalize using ImageNet statistics.

    Args:
        crop_rgb (np.ndarray): Cropped person image in RGB format (H, W, 3).

    Returns:
        torch.Tensor: Preprocessed tensor of shape (C, H, W), ready for model input.
    """
    tensor = transform_pipeline(crop_rgb)
    return tensor
