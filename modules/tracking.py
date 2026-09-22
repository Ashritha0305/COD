"""
modules/tracking.py
===================
DeepSORT Multi-Object Wildlife Tracking Module.

Architecture Position:
    Input -> U-Net -> Optical Flow -> Fusion -> YOLOv8 -> [DeepSORT Tracking] -> ZoeDepth -> Threat Analysis

Contract:
    - Uses DeepSORT (deep_sort_realtime) for persistent track IDs across frames
    - Binds with YOLOv8 detection outputs
    - Tracks animal identity, persistence (frames tracked), and velocity/motion
    - Draws persistent track IDs on video output frames
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class WildlifeTracker:
    """
    DeepSORT wrapper for tracking detected wildlife across consecutive video frames.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 2,
        nms_max_overlap: float = 1.0,
        max_cosine_distance: float = 0.3,
        nn_budget: Optional[int] = None,
    ) -> None:
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            override_track_class=None,
        )
        self.track_history: Dict[int, int] = {}  # track_id -> total frames tracked

    def update(
        self,
        detections: List[Tuple[List[float], float, str]],
        frame_bgr: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Update tracker with new frame detections.

        Parameters
        ----------
        detections : list of ([left, top, w, h], confidence, class_name)
        frame_bgr : current video frame (H, W, 3) uint8 BGR

        Returns
        -------
        tracked_objects : list of dicts with:
            - track_id: int
            - bbox_ltrb: [left, top, right, bottom]
            - class_name: str
            - confidence: float
            - persistence: int (consecutive frames tracked)
        """
        tracks = self.tracker.update_tracks(detections, frame=frame_bgr)
        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            ltrb = [int(v) for v in track.to_ltrb()]
            class_name = track.get_det_class() or "Animal"
            conf = float(track.get_det_conf() or 0.80)

            # Update persistence count
            self.track_history[track_id] = self.track_history.get(track_id, 0) + 1
            persistence = self.track_history[track_id]

            results.append({
                "track_id": track_id,
                "bbox_ltrb": ltrb,
                "class_name": class_name,
                "confidence": conf,
                "persistence": persistence,
            })

        return results

    def reset(self) -> None:
        """Reset internal tracker state between video clips."""
        self.tracker.delete_all_tracks()
        self.track_history.clear()

    def draw_tracks(
        self,
        frame_bgr: np.ndarray,
        tracks: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 220, 255),
    ) -> np.ndarray:
        """Draw tracked bounding boxes with persistent Track IDs."""
        out = frame_bgr.copy()
        H, W = out.shape[:2]

        for t in tracks:
            left, top, right, bottom = t["bbox_ltrb"]
            left = max(0, min(W - 1, left))
            top = max(0, min(H - 1, top))
            right = max(0, min(W - 1, right))
            bottom = max(0, min(H - 1, bottom))

            track_id = t["track_id"]
            cls_name = t["class_name"]
            conf = t["confidence"]

            # Box
            cv2.rectangle(out, (left, top), (right, bottom), color, 2)

            # Track ID label badge
            label = f"ID: {track_id} | {cls_name.capitalize()} ({conf * 100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(
                out,
                (left, max(0, top - th - 8)),
                (left + tw + 6, max(0, top)),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                out,
                label,
                (left + 3, max(th + 2, top - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return out
