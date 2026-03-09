"""
trainer.py — Training of the OpenVAD anomaly detection model on normal videos.

Training pipeline:
1. Scan all videos in the NORMAL_VIDEO_DIR folder.
2. For each video:
   a. Extract frames (with frame skipping to reduce redundancy).
   b. Run YOLOv8 person detection on each frame.
   c. Crop all detected person regions and preprocess them.
3. Train the OpenVAD model using a combination of:
   - EDL loss (Dirichlet-based classification with uncertainty)
   - MIL ranking loss (top-k instance selection)
   - Triplet loss (discriminative feature learning)
   - Normalizing Flow loss (density estimation on normal features)
4. Pseudo-anomalies are generated via the Normalizing Flow module.
5. Save the trained model weights to RESULTS_DIR.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from detector import PersonDetector
from preprocessing import extract_frames, crop_person, prepare_crop
from anomaly_model import (
    AnomalyAutoencoder,  # alias for OpenVADModel
    edl_mse_loss,
    mil_ranking_loss,
    triplet_loss,
)


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
    Train the OpenVAD model on the collected person crops.

    Uses a combination of EDL loss, MIL ranking loss, triplet loss,
    and normalizing flow loss. Pseudo-anomalies are generated via the
    NF module to provide negative examples.

    Args:
        crops_tensor (torch.Tensor): Training data of shape (N, C, H, W).

    Returns:
        OpenVADModel: The trained model.
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

    # Initialize the OpenVAD model and move to device (GPU/CPU)
    model = AnomalyAutoencoder().to(config.DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"\n[Trainer] Starting OpenVAD training for {config.NUM_EPOCHS} epochs...")
    print(f"[Trainer] Device: {config.DEVICE}")
    print(f"[Trainer] Batch size: {config.BATCH_SIZE}")
    print(f"[Trainer] Training samples: {len(dataset)}")
    print(f"[Trainer] Architecture: OpenVAD (GCN + EDL + NF + MIL)")
    print("-" * 60)

    num_classes = config.EDL_NUM_CLASSES

    model.train()
    for epoch in range(1, config.NUM_EPOCHS + 1):
        total_loss = 0.0
        total_edl = 0.0
        total_nf = 0.0
        total_mil = 0.0
        num_batches = 0

        for batch_idx, (batch_crops,) in enumerate(dataloader):
            # Move batch to device
            batch_crops = batch_crops.to(config.DEVICE)
            batch_size = batch_crops.size(0)

            # ============================================================
            # Step 1: Forward pass on normal crops → label = 0 (normal)
            # ============================================================
            evidence, alpha, uncertainty = model(batch_crops)
            normal_target = torch.zeros(batch_size, num_classes, device=config.DEVICE)
            normal_target[:, 0] = 1.0  # One-hot: class 0 = normal

            # EDL loss for normal samples
            loss_edl_normal = edl_mse_loss(alpha, normal_target, epoch)

            # ============================================================
            # Step 2: Generate pseudo-anomalies via Normalizing Flow
            # ============================================================
            pseudo_anomaly_features = model.generate_pseudo_anomalies(
                batch_crops, scale=3.0
            )

            # Classify the pseudo-anomaly features directly through EDL
            pseudo_evidence, pseudo_alpha, pseudo_uncertainty = (
                model.edl_classifier(pseudo_anomaly_features)
            )
            anomaly_target = torch.zeros(batch_size, num_classes, device=config.DEVICE)
            anomaly_target[:, 1] = 1.0  # One-hot: class 1 = anomalous

            # EDL loss for pseudo-anomaly samples
            loss_edl_anomaly = edl_mse_loss(pseudo_alpha, anomaly_target, epoch)

            # Combined EDL loss
            loss_edl = (loss_edl_normal + loss_edl_anomaly) / 2.0

            # ============================================================
            # Step 3: Normalizing Flow log-probability loss
            # Normal features should have HIGH log-probability
            # ============================================================
            nf_log_prob = model.compute_nf_log_prob(batch_crops)
            loss_nf = -nf_log_prob.mean()  # Maximize log-prob → minimize negative

            # ============================================================
            # Step 4: MIL ranking loss
            # Anomaly scores should be higher than normal scores
            # ============================================================
            loss_mil = mil_ranking_loss(uncertainty, pseudo_uncertainty)

            # ============================================================
            # Step 5: Triplet loss for feature discrimination
            # ============================================================
            normal_features = model.extract_features(batch_crops)
            # Use first half as anchor, second half as positive (same class)
            if batch_size > 2:
                half = batch_size // 2
                anchor = normal_features[:half]
                positive = normal_features[half:2 * half]
                negative = pseudo_anomaly_features[:half]
                loss_triplet = triplet_loss(anchor, positive, negative)
            else:
                loss_triplet = torch.tensor(0.0, device=config.DEVICE)

            # ============================================================
            # Total loss: weighted combination
            # ============================================================
            loss = loss_edl + 0.1 * loss_nf + loss_mil + 0.5 * loss_triplet

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_edl += loss_edl.item()
            total_nf += loss_nf.item()
            total_mil += loss_mil.item()
            num_batches += 1

        # Log average losses for this epoch
        avg_loss = total_loss / num_batches
        avg_edl = total_edl / num_batches
        avg_nf = total_nf / num_batches
        avg_mil = total_mil / num_batches
        print(
            f"  Epoch [{epoch:3d}/{config.NUM_EPOCHS}] — "
            f"Total: {avg_loss:.4f}  EDL: {avg_edl:.4f}  "
            f"NF: {avg_nf:.4f}  MIL: {avg_mil:.4f}"
        )

    print("-" * 60)
    print("[Trainer] OpenVAD training complete!")

    return model


def save_model(model):
    """
    Save the trained model weights to disk.

    Args:
        model (OpenVADModel): The trained model to save.
    """
    config.ensure_dirs()
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"[Trainer] Model saved to: {config.MODEL_SAVE_PATH}")


def train():
    """
    Full training pipeline:
    1. Collect person crops from normal videos.
    2. Train the OpenVAD model on those crops.
    3. Save the trained model weights.
    """
    print("=" * 60)
    print("  TRAINING PHASE — OpenVAD self-supervised learning on normal videos")
    print("=" * 60)

    # Step 1: Collect training data from normal showroom videos
    crops_tensor = collect_training_crops()

    # Step 2: Train the OpenVAD model
    model = train_model(crops_tensor)

    # Step 3: Save the trained model
    save_model(model)

    return model


if __name__ == "__main__":
    train()
