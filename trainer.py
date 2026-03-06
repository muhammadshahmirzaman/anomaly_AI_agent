"""
trainer.py — Self-supervised training of the anomaly autoencoder on normal videos.

Training pipeline:
1. Scan all videos in the NORMAL_VIDEO_DIR folder.
2. For each video:
   a. Extract frames (with frame skipping to reduce redundancy).
   b. Run YOLOv8 person detection on each frame.
   c. Crop all detected person regions and preprocess them.
3. Train the convolutional autoencoder on the collected crops using MSE loss.
4. Save the trained model weights to RESULTS_DIR.

The model learns to reconstruct normal person appearances. During inference,
abnormal appearances will produce higher reconstruction errors (anomaly scores).
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from detector import PersonDetector
from preprocessing import extract_frames, crop_person, prepare_crop
from anomaly_model import AnomalyAutoencoder


def collect_training_crops():
    """
    Collect all person crops from normal showroom videos.

    Iterates all videos in NORMAL_VIDEO_DIR, extracts frames, detects persons
    with YOLOv8, crops and preprocesses each detected person region.

    Returns:
        torch.Tensor: Stacked tensor of all preprocessed crops, shape (N, C, H, W).
    """
    video_dir = config.NORMAL_VIDEO_DIR

    # Find all video files in the normal videos directory
    video_files = [
        os.path.join(video_dir, f)
        for f in sorted(os.listdir(video_dir))
        if f.lower().endswith(config.VIDEO_EXTENSIONS)
    ]

    if not video_files:
        raise FileNotFoundError(
            f"No videos found in {video_dir}. "
            f"Please place normal showroom videos in this folder."
        )

    print(f"[Trainer] Found {len(video_files)} normal video(s) for training.")

    # Initialize YOLOv8 detector for person detection
    detector = PersonDetector()
    all_crops = []

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        print(f"[Trainer] Processing: {video_name}")
        crop_count = 0

        # Extract frames with skipping to reduce redundancy
        for frame_idx, frame in extract_frames(video_path, frame_skip=config.TRAIN_FRAME_SKIP):
            # Detect persons in this frame
            detections = detector.detect(frame)

            for det in detections:
                x1, y1, x2, y2, conf = det

                # Crop the person region from the frame
                crop_rgb = crop_person(frame, (x1, y1, x2, y2))
                if crop_rgb is None:
                    continue

                # Preprocess the crop: resize → tensor → normalize
                crop_tensor = prepare_crop(crop_rgb)
                all_crops.append(crop_tensor)
                crop_count += 1

        print(f"[Trainer]   → Collected {crop_count} person crops from {video_name}")

    if not all_crops:
        raise ValueError(
            "No person crops collected from normal videos. "
            "Ensure the videos contain visible people."
        )

    # Stack all crop tensors into a single tensor: (N, C, H, W)
    crops_tensor = torch.stack(all_crops)
    print(f"[Trainer] Total training crops: {crops_tensor.shape[0]}")
    return crops_tensor


def train_model(crops_tensor):
    """
    Train the anomaly autoencoder on the collected person crops.

    Uses MSE loss to train the autoencoder to reconstruct normal person crops.
    The model learns the distribution of normal appearances.

    Args:
        crops_tensor (torch.Tensor): Training data of shape (N, C, H, W).

    Returns:
        AnomalyAutoencoder: The trained model.
    """
    # Create dataset and dataloader for batch training
    dataset = TensorDataset(crops_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.DEVICE.type == "cuda" else False,
    )

    # Initialize the autoencoder model and move to device (GPU/CPU)
    model = AnomalyAutoencoder().to(config.DEVICE)

    # MSE loss measures reconstruction quality
    criterion = nn.MSELoss()

    # Adam optimizer with configured learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"\n[Trainer] Starting training for {config.NUM_EPOCHS} epochs...")
    print(f"[Trainer] Device: {config.DEVICE}")
    print(f"[Trainer] Batch size: {config.BATCH_SIZE}")
    print(f"[Trainer] Training samples: {len(dataset)}")
    print("-" * 60)

    model.train()
    for epoch in range(1, config.NUM_EPOCHS + 1):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (batch_crops,) in enumerate(dataloader):
            # Move batch to device
            batch_crops = batch_crops.to(config.DEVICE)

            # Forward pass: reconstruct the input
            reconstructed = model(batch_crops)

            # Compute MSE loss between original and reconstruction
            loss = criterion(reconstructed, batch_crops)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Log average loss for this epoch
        avg_loss = total_loss / num_batches
        print(f"  Epoch [{epoch:3d}/{config.NUM_EPOCHS}] — Avg Loss: {avg_loss:.6f}")

    print("-" * 60)
    print("[Trainer] Training complete!")

    return model


def save_model(model):
    """
    Save the trained model weights to disk.

    Args:
        model (AnomalyAutoencoder): The trained model to save.
    """
    config.ensure_dirs()
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"[Trainer] Model saved to: {config.MODEL_SAVE_PATH}")


def train():
    """
    Full training pipeline:
    1. Collect person crops from normal videos.
    2. Train the autoencoder on those crops.
    3. Save the trained model weights.
    """
    print("=" * 60)
    print("  TRAINING PHASE — Self-supervised learning on normal videos")
    print("=" * 60)

    # Step 1: Collect training data from normal showroom videos
    crops_tensor = collect_training_crops()

    # Step 2: Train the autoencoder
    model = train_model(crops_tensor)

    # Step 3: Save the trained model
    save_model(model)

    return model


if __name__ == "__main__":
    train()
