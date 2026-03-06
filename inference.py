"""
inference.py — Anomaly detection inference on abnormal showroom videos.

Inference pipeline:
1. Load the trained anomaly autoencoder model.
2. For each video in ABNORMAL_VIDEO_DIR:
   a. Extract all frames.
   b. Detect and track persons using YOLOv8 + ByteTrack.
   c. Crop each tracked person, preprocess, and compute anomaly score.
   d. Annotate the frame with bounding boxes, track IDs, and anomaly scores.
   e. Write annotated frames to an output video.
   f. Log per-frame, per-person anomaly scores to a CSV file.
"""

import os
import csv

import cv2
import pandas as pd
from tqdm import tqdm

import config
from detector import PersonDetector
from preprocessing import (
    extract_frames,
    get_video_properties,
    crop_person,
    prepare_crop,
)
from anomaly_model import AnomalyAutoencoder, compute_anomaly_score, load_model
from visualizer import Visualizer


def run_inference_on_video(video_path, model, detector, visualizer):
    """
    Run the full inference pipeline on a single video.

    For each frame:
    - Detect and track persons with YOLOv8 + ByteTrack.
    - Compute anomaly scores for each tracked person.
    - Annotate and save the output video.
    - Collect anomaly data for CSV export.

    Args:
        video_path (str): Path to the input video.
        model (AnomalyAutoencoder): Trained anomaly model.
        detector (PersonDetector): YOLOv8 person detector (with ByteTrack tracking).
        visualizer (Visualizer): Visualization utility.

    Returns:
        list: List of dicts with keys: frame_idx, track_id, x1, y1, x2, y2,
              anomaly_score, is_anomalous.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n[Inference] Processing video: {video_name}")

    # Get video properties for creating the output writer
    props = get_video_properties(video_path)
    fps = props["fps"]
    width = props["width"]
    height = props["height"]
    total_frames = props["total_frames"]

    # Create output video writer
    output_video_path = os.path.join(config.RESULTS_DIR, f"{video_name}_annotated.mp4")
    writer = visualizer.create_video_writer(output_video_path, fps, width, height)

    # Collect anomaly records for CSV export
    anomaly_records = []

    # Process each frame
    frame_count = 0
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    pbar = tqdm(total=total_frames, desc=f"  {video_name}", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = frame_count
        frame_count += 1
        pbar.update(1)

        # Skip frames based on inference frame skip setting
        if frame_idx % config.INFER_FRAME_SKIP != 0:
            writer.write(frame)  # Write original frame without annotations
            continue

        # ------------------------------------------------------------------
        # Step 1: Detect and track persons using YOLOv8 + ByteTrack
        # ------------------------------------------------------------------
        tracked_persons = detector.track(frame)
        total_persons = len(tracked_persons)
        anomalous_count = 0

        # ------------------------------------------------------------------
        # Step 2: For each tracked person, compute anomaly score
        # ------------------------------------------------------------------
        for person in tracked_persons:
            bbox = person["bbox"]
            track_id = person["track_id"]

            # Crop the person region from the frame
            crop_rgb = crop_person(frame, bbox)

            if crop_rgb is None:
                # Invalid crop — skip but still draw the box with score 0
                anomaly_score = 0.0
            else:
                # Preprocess the crop and compute anomaly score
                crop_tensor = prepare_crop(crop_rgb)
                anomaly_score = compute_anomaly_score(model, crop_tensor)

            # Determine if this person is anomalous
            is_anomalous = anomaly_score >= config.ANOMALY_THRESHOLD
            if is_anomalous:
                anomalous_count += 1

            # ----------------------------------------------------------
            # Step 3: Annotate the frame with bbox, ID, and score
            # ----------------------------------------------------------
            visualizer.draw_tracked_person(frame, bbox, track_id, anomaly_score)

            # ----------------------------------------------------------
            # Step 4: Record anomaly data for CSV export
            # ----------------------------------------------------------
            x1, y1, x2, y2 = bbox
            anomaly_records.append({
                "frame_idx": frame_idx,
                "track_id": track_id,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "anomaly_score": round(anomaly_score, 6),
                "is_anomalous": is_anomalous,
            })

        # Draw frame-level info overlay
        visualizer.draw_frame_info(frame, frame_idx, total_persons, anomalous_count)

        # Write the annotated frame to the output video
        writer.write(frame)

    pbar.close()
    cap.release()
    writer.release()

    print(f"[Inference] Output video saved: {output_video_path}")
    return anomaly_records


def save_anomaly_csv(records, video_name):
    """
    Save anomaly records to a CSV file.

    The CSV contains per-frame, per-person anomaly scores with columns:
    frame_idx, track_id, x1, y1, x2, y2, anomaly_score, is_anomalous.

    Args:
        records (list): List of anomaly record dicts.
        video_name (str): Base name of the video (used for the CSV filename).
    """
    if not records:
        print(f"[Inference] No detections to save for {video_name}.")
        return

    csv_path = os.path.join(config.RESULTS_DIR, f"{video_name}_anomaly_scores.csv")
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

    # Print summary statistics
    total_detections = len(df)
    anomalous = df["is_anomalous"].sum()
    mean_score = df["anomaly_score"].mean()
    max_score = df["anomaly_score"].max()

    print(f"[Inference] CSV saved: {csv_path}")
    print(f"  → Total detections: {total_detections}")
    print(f"  → Anomalous detections: {anomalous} ({100*anomalous/total_detections:.1f}%)")
    print(f"  → Mean anomaly score: {mean_score:.6f}")
    print(f"  → Max anomaly score:  {max_score:.6f}")


def run():
    """
    Full inference pipeline:
    1. Load trained model.
    2. Iterate all videos in ABNORMAL_VIDEO_DIR.
    3. For each video: detect, track, score, annotate, and save results.
    """
    print("=" * 60)
    print("  INFERENCE PHASE — Anomaly detection on abnormal videos")
    print("=" * 60)

    config.ensure_dirs()

    # Load the trained anomaly model
    model = load_model()

    # Initialize detector and visualizer
    detector = PersonDetector()
    visualizer = Visualizer()

    # Find all videos in the abnormal video directory
    video_dir = config.ABNORMAL_VIDEO_DIR
    video_files = [
        os.path.join(video_dir, f)
        for f in sorted(os.listdir(video_dir))
        if f.lower().endswith(config.VIDEO_EXTENSIONS)
    ]

    if not video_files:
        raise FileNotFoundError(
            f"No videos found in {video_dir}. "
            f"Please place abnormal showroom videos in this folder."
        )

    print(f"[Inference] Found {len(video_files)} abnormal video(s) for inference.\n")

    # Process each video
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Run inference on this video
        records = run_inference_on_video(video_path, model, detector, visualizer)

        # Save anomaly scores to CSV
        save_anomaly_csv(records, video_name)

    print("\n" + "=" * 60)
    print("  INFERENCE COMPLETE — Check results in:", config.RESULTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    run()
