"""
visualizer.py — Visualization utilities for annotated output videos.

Draws bounding boxes, track IDs, and anomaly scores on video frames.
Persons with anomaly scores above the threshold are highlighted in red;
others are drawn in green.
"""

import cv2
import config


class Visualizer:
    """
    Handles drawing annotations on video frames and writing output videos.
    """

    def __init__(self):
        self.threshold = config.ANOMALY_THRESHOLD

    def draw_tracked_person(self, frame, bbox, track_id, anomaly_score):
        """
        Draw a bounding box, track ID, and anomaly score on the frame.

        Args:
            frame (np.ndarray): Video frame (H, W, 3) in BGR format.
            bbox (tuple): Bounding box as (x1, y1, x2, y2).
            track_id (int): Unique tracking ID for this person.
            anomaly_score (float): Anomaly score for this person.

        Returns:
            np.ndarray: The annotated frame (modified in-place).
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Choose color based on anomaly threshold
        if anomaly_score >= self.threshold:
            color = config.COLOR_ANOMALY   # Red — anomalous
            label_prefix = "ANOMALY"
        else:
            color = config.COLOR_NORMAL    # Green — normal
            label_prefix = "Normal"

        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

        # Prepare the label text: "ID:X | Normal/ANOMALY: 0.0123"
        label = f"ID:{track_id} | {label_prefix}: {anomaly_score:.4f}"

        # Calculate text size for the background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1
        )

        # Draw a filled rectangle behind the text for readability
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            color,
            cv2.FILLED,
        )

        # Draw the text label
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            (255, 255, 255),   # White text
            1,
            cv2.LINE_AA,
        )

        return frame

    def draw_frame_info(self, frame, frame_idx, total_persons, anomalous_count):
        """
        Draw frame-level information at the top of the frame.

        Args:
            frame (np.ndarray): Video frame.
            frame_idx (int): Current frame index.
            total_persons (int): Total persons detected in this frame.
            anomalous_count (int): Number of anomalous persons in this frame.

        Returns:
            np.ndarray: Annotated frame.
        """
        info_text = (
            f"Frame: {frame_idx} | "
            f"Persons: {total_persons} | "
            f"Anomalous: {anomalous_count}"
        )
        cv2.putText(
            frame,
            info_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    @staticmethod
    def create_video_writer(output_path, fps, width, height):
        """
        Create an OpenCV VideoWriter for saving annotated output videos.

        Args:
            output_path (str): Path to the output video file.
            fps (int): Frames per second.
            width (int): Frame width.
            height (int): Frame height.

        Returns:
            cv2.VideoWriter: Initialized video writer.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {output_path}")
        print(f"[Visualizer] Video writer created: {output_path}")
        return writer
