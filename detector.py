"""
detector.py — YOLOv8-based person detection module.

Uses the ultralytics YOLOv8 model to detect people in video frames.
Only detections of the 'person' class (COCO class 0) are returned,
filtered by a confidence threshold.
"""

from ultralytics import YOLO
import config


class PersonDetector:
    """
    Wrapper around YOLOv8 for detecting persons in video frames.

    Attributes:
        model (YOLO): The loaded YOLOv8 model instance.
        confidence (float): Minimum confidence threshold for detections.
    """

    def __init__(self):
        """
        Initialize the YOLOv8 model.
        The model weights are automatically downloaded on first use.
        """
        print(f"[Detector] Loading YOLOv8 model: {config.YOLO_MODEL}")
        self.model = YOLO(config.YOLO_MODEL)
        self.confidence = config.YOLO_CONFIDENCE
        print(f"[Detector] Model loaded. Using device: {config.DEVICE}")

    def detect(self, frame):
        """
        Run person detection on a single video frame.

        Args:
            frame (np.ndarray): Input frame in BGR format (H, W, 3).

        Returns:
            list: List of detections, each as [x1, y1, x2, y2, confidence].
                  Coordinates are in pixel values. Only 'person' class detections
                  are returned.
        """
        # Run YOLOv8 inference on the frame
        # verbose=False suppresses per-frame console output
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # Get the class ID for this detection
                cls_id = int(boxes.cls[i].item())

                # Only keep 'person' class detections (COCO class 0)
                if cls_id != config.PERSON_CLASS_ID:
                    continue

                # Extract bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = boxes.conf[i].item()

                detections.append([x1, y1, x2, y2, conf])

        return detections

    def track(self, frame):
        """
        Run person detection + ByteTrack tracking on a single video frame.

        ByteTrack is integrated into ultralytics and provides multi-object
        tracking with persistent IDs across frames.

        Args:
            frame (np.ndarray): Input frame in BGR format (H, W, 3).

        Returns:
            list: List of tracked detections, each as a dict with keys:
                  'bbox' (x1, y1, x2, y2), 'track_id' (int), 'confidence' (float).
                  Returns empty list if no persons are tracked.
        """
        # Run YOLOv8 with ByteTrack tracking enabled
        # persist=True maintains track IDs across consecutive frames
        results = self.model.track(
            frame,
            conf=self.confidence,
            tracker=config.TRACKER_TYPE,
            persist=True,
            verbose=False,
        )

        tracked = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())

                # Only keep person detections
                if cls_id != config.PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = boxes.conf[i].item()
                track_id = int(boxes.id[i].item())

                tracked.append({
                    "bbox": (x1, y1, x2, y2),
                    "track_id": track_id,
                    "confidence": conf,
                })

        return tracked
