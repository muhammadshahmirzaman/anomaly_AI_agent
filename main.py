"""
main.py — Entry point for the Showroom Anomaly Detection Pipeline.

Usage:
    python main.py                  # Run both training and inference (default)
    python main.py --mode train     # Run training only
    python main.py --mode infer     # Run inference only (requires a trained model)
    python main.py --mode both      # Run training followed by inference

The pipeline:
    1. TRAINING: Learns normal person appearances from showroom videos.
    2. INFERENCE: Detects anomalous person behavior in new videos.
"""

import argparse
import sys
import os

import config
from trainer import train
from inference import run


def print_banner():
    """Print a startup banner with configuration summary."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║   SHOWROOM ANOMALY DETECTION PIPELINE                    ║")
    print("║   YOLOv8 + ByteTrack + Autoencoder                      ║")
    print("╚" + "═" * 58 + "╝")
    print()
    print(f"  Device:            {config.DEVICE}")
    print(f"  Normal videos:     {os.path.abspath(config.NORMAL_VIDEO_DIR)}")
    print(f"  Abnormal videos:   {os.path.abspath(config.ABNORMAL_VIDEO_DIR)}")
    print(f"  Results output:    {os.path.abspath(config.RESULTS_DIR)}")
    print(f"  Anomaly threshold: {config.ANOMALY_THRESHOLD}")
    print(f"  Model save path:   {os.path.abspath(config.MODEL_SAVE_PATH)}")
    print()


def main():
    """Main function: parse arguments and run the selected pipeline mode."""
    parser = argparse.ArgumentParser(
        description="Showroom Anomaly Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  Run both training and inference
  python main.py --mode train     Train the anomaly model on normal videos
  python main.py --mode infer     Run inference on abnormal videos
  python main.py --mode both      Full pipeline (train + infer)

Input Folders:
  Place normal showroom videos in:   ./data/showroom_normal/
  Place abnormal showroom videos in: ./data/showroom_abnormal/

Output:
  Annotated videos and CSV files are saved to: ./results/
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer", "both"],
        default="both",
        help="Pipeline mode: 'train' (training only), 'infer' (inference only), "
             "'both' (train then infer). Default: 'both'.",
    )

    args = parser.parse_args()

    # Print configuration banner
    print_banner()

    # Ensure required directories exist
    config.ensure_dirs()

    # ---- TRAINING PHASE ----
    if args.mode in ("train", "both"):
        # Check that normal videos exist
        normal_videos = [
            f for f in os.listdir(config.NORMAL_VIDEO_DIR)
            if f.lower().endswith(config.VIDEO_EXTENSIONS)
        ]
        if not normal_videos:
            print(f"[ERROR] No videos found in: {os.path.abspath(config.NORMAL_VIDEO_DIR)}")
            print("        Please place your normal showroom videos there first.")
            sys.exit(1)

        train()

    # ---- INFERENCE PHASE ----
    if args.mode in ("infer", "both"):
        # Check that the trained model exists
        if not os.path.exists(config.MODEL_SAVE_PATH):
            print(f"[ERROR] Trained model not found at: {config.MODEL_SAVE_PATH}")
            print("        Run training first: python main.py --mode train")
            sys.exit(1)

        # Check that abnormal videos exist
        abnormal_videos = [
            f for f in os.listdir(config.ABNORMAL_VIDEO_DIR)
            if f.lower().endswith(config.VIDEO_EXTENSIONS)
        ]
        if not abnormal_videos:
            print(f"[ERROR] No videos found in: {os.path.abspath(config.ABNORMAL_VIDEO_DIR)}")
            print("        Please place your abnormal showroom videos there first.")
            sys.exit(1)

        run()

    print("\n[Done] Pipeline finished successfully!")


if __name__ == "__main__":
    main()
