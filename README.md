# 🏬 Showroom Anomaly Detection Pipeline

A modular Python pipeline for detecting anomalous person behavior in showroom surveillance videos using **YOLOv8** (person detection), **ByteTrack** (multi-object tracking), and a **self-supervised convolutional autoencoder** (anomaly scoring).

---

## 📁 Project Structure

```
anomaly_AI_agent/
│
├── config.py              # Centralized configuration (paths, hyperparameters, thresholds)
├── preprocessing.py       # Frame extraction, cropping, resizing, normalization
├── detector.py            # YOLOv8 person detection + ByteTrack tracking
├── anomaly_model.py       # Convolutional autoencoder for anomaly detection
├── trainer.py             # Self-supervised training on normal videos
├── inference.py           # Inference pipeline on abnormal videos
├── visualizer.py          # Bounding box drawing, annotation, video output
├── main.py                # Entry point (CLI interface)
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── data/
│   ├── showroom_normal/   # ← PUT YOUR NORMAL TRAINING VIDEOS HERE
│   └── showroom_abnormal/ # ← PUT YOUR ABNORMAL TEST VIDEOS HERE
│
└── results/               # ← OUTPUT: annotated videos, CSVs, model weights
    ├── anomaly_model.pth          # Trained model weights
    ├── <video>_annotated.mp4      # Annotated output videos
    └── <video>_anomaly_scores.csv # Per-frame per-person anomaly scores
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch with CUDA support is recommended for GPU acceleration. Install from [pytorch.org](https://pytorch.org/) for your specific CUDA version.

### 2. Prepare Your Data

Place your videos in the appropriate folders:

| Folder | What to put here |
|--------|-----------------|
| `./data/showroom_normal/` | Normal showroom videos (for training) — footage with **no anomalies** |
| `./data/showroom_abnormal/` | Abnormal showroom videos (for inference) — footage where you want to **detect anomalies** |

**Supported formats:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`

### 3. Run the Pipeline

```bash
# Full pipeline: train on normal videos, then infer on abnormal videos
python main.py

# Training only
python main.py --mode train

# Inference only (requires a trained model in ./results/)
python main.py --mode infer

# Show help
python main.py --help
```

### 4. View Results

After running, check the `./results/` folder:

- **`<video>_annotated.mp4`** — Output video with bounding boxes, track IDs, and anomaly scores
- **`<video>_anomaly_scores.csv`** — CSV with per-frame, per-person anomaly data
- **`anomaly_model.pth`** — Saved model weights

---

## 📥 Where to Provide Inputs

| Input | Location | Description |
|-------|----------|-------------|
| **Normal training videos** | `./data/showroom_normal/` | Place video files of normal showroom footage here. These are used to train the model. |
| **Abnormal test videos** | `./data/showroom_abnormal/` | Place video files to test for anomalies here. The trained model will score each person. |
| **Configuration** | `config.py` | Modify hyperparameters, paths, thresholds, and other settings. |
| **Anomaly threshold** | `config.py` → `ANOMALY_THRESHOLD` | Adjust this value to change sensitivity (lower = more sensitive). |
| **Training epochs** | `config.py` → `NUM_EPOCHS` | Increase for better model quality, decrease for faster training. |
| **Frame skip (training)** | `config.py` → `TRAIN_FRAME_SKIP` | Higher value = fewer frames processed = faster training. |
| **YOLO confidence** | `config.py` → `YOLO_CONFIDENCE` | Minimum detection confidence for person detection. |

---

## 🔧 Configuration Reference

All configuration is in **`config.py`**. Key parameters:

### Paths
| Parameter | Default | Description |
|-----------|---------|-------------|
| `NORMAL_VIDEO_DIR` | `./data/showroom_normal/` | Training videos directory |
| `ABNORMAL_VIDEO_DIR` | `./data/showroom_abnormal/` | Inference videos directory |
| `RESULTS_DIR` | `./results/` | Output directory |
| `MODEL_SAVE_PATH` | `./results/anomaly_model.pth` | Trained model location |

### Detection & Tracking
| Parameter | Default | Description |
|-----------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | YOLOv8 model variant |
| `YOLO_CONFIDENCE` | `0.5` | Detection confidence threshold |
| `TRACKER_TYPE` | `bytetrack.yaml` | Tracker configuration |

### Anomaly Model
| Parameter | Default | Description |
|-----------|---------|-------------|
| `CROP_SIZE` | `(64, 128)` | Size to resize person crops (W, H) |
| `ANOMALY_THRESHOLD` | `0.02` | Score above which a person is flagged |
| `NUM_EPOCHS` | `50` | Training epochs |
| `BATCH_SIZE` | `64` | Training batch size |
| `LEARNING_RATE` | `1e-3` | Adam optimizer learning rate |

---

## 🧠 How It Works

### Training Phase (Self-Supervised)
1. **Frame Extraction**: Extracts frames from normal showroom videos.
2. **Person Detection**: YOLOv8 detects all persons in each frame.
3. **Crop Collection**: Person regions are cropped, resized to 64×128, and normalized.
4. **Autoencoder Training**: A convolutional autoencoder learns to reconstruct normal person appearances using MSE loss. The model captures the distribution of "normal" person looks.

### Inference Phase
1. **Frame Processing**: Extracts frames from abnormal/test videos.
2. **Person Detection + Tracking**: YOLOv8 detects persons; ByteTrack assigns persistent IDs.
3. **Anomaly Scoring**: Each person crop is fed through the trained autoencoder. The **reconstruction error (MSE)** is the anomaly score:
   - **Low score** → Person looks normal (similar to training data).
   - **High score** → Person looks anomalous (different from training data).
4. **Annotation**: Frames are annotated with bounding boxes (green=normal, red=anomalous), track IDs, and scores.
5. **Output**: Annotated videos and CSV score files are saved.

---

## 📊 CSV Output Format

Each CSV file contains:

| Column | Description |
|--------|-------------|
| `frame_idx` | Frame number in the video |
| `track_id` | Unique person tracking ID |
| `x1, y1, x2, y2` | Bounding box coordinates |
| `anomaly_score` | Reconstruction error (higher = more anomalous) |
| `is_anomalous` | `True` if score ≥ threshold |

---

## 🖥️ GPU Support

The pipeline automatically uses GPU (CUDA) if available. To verify:

```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should show your GPU
```

If CUDA is not available, the pipeline falls back to CPU (slower but functional).

---

## 📌 Module Descriptions

| Module | Purpose |
|--------|---------|
| `config.py` | All paths, hyperparameters, device config, and thresholds |
| `preprocessing.py` | Frame extraction from videos, person crop resizing/normalization |
| `detector.py` | YOLOv8 person detection + ByteTrack tracking wrapper |
| `anomaly_model.py` | Conv autoencoder architecture + anomaly score computation |
| `trainer.py` | Self-supervised training loop on normal video crops |
| `inference.py` | Full inference pipeline: detect → track → score → annotate → save |
| `visualizer.py` | Drawing bboxes, labels, and writing output video files |
| `main.py` | CLI entry point orchestrating train/infer modes |

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| No videos found | Ensure videos are in the correct folders with supported extensions |
| CUDA out of memory | Reduce `BATCH_SIZE` in `config.py` or use a smaller YOLO model |
| Poor anomaly detection | Increase `NUM_EPOCHS`, add more normal training videos, or adjust `ANOMALY_THRESHOLD` |
| Model not found (inference) | Run training first: `python main.py --mode train` |
| Low FPS during inference | Increase `INFER_FRAME_SKIP` in `config.py` to skip frames |

---

## 📜 License

This project is for educational and research purposes.
