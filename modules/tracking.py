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
            half=True,
            embedder_gpu=True,
        )
        self.track_history: Dict[int, int] = {}  # track_id -> total frames tracked
        # Temporal Class Stabilization: track_id -> list of (class_name, confidence)
        self.track_class_history: Dict[int, List[Tuple[str, float]]] = {}
        self.history_window: int = 15

        # Warm up the embedder model on startup to eliminate cold-start latency spike
        try:
            self.tracker.update_tracks(
                [([0.0, 0.0, 10.0, 10.0], 0.9, "warmup")],
                frame=np.zeros((64, 64, 3), dtype=np.uint8),
            )
            self.tracker.delete_all_tracks()
        except Exception:
            pass

    def update(
        self,
        detections: List[Tuple[List[float], float, str]],
        frame_bgr: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Update tracker with new frame detections.

        Parameters
        ----------
        detections : list of ([left, top, w, h], confidence, class_name)
        frame_bgr : current video frame (H, W, 3) uint8 BGR.
                    Optional when detections is empty.

        Returns
        -------
        tracked_objects : list of dicts with:
            - track_id: int
            - bbox_ltrb: [left, top, right, bottom]
            - class_name: str (temporally stabilized)
            - confidence: float
            - persistence: int (consecutive frames tracked)
        """
        from deep_sort_realtime.deep_sort.track import TrackState

        # Optimization: On non-detection frames (detections is empty), skip appearance
        # feature extraction completely and advance Kalman filter tracks with zero embedding cost.
        valid_detections = []
        if detections:
            for b, c, cls in detections:
                if len(b) >= 4 and b[2] > 1.0 and b[3] > 1.0:
                    valid_detections.append(([max(0.0, float(b[0])), max(0.0, float(b[1])), float(b[2]), float(b[3])], float(c), str(cls)))

        if len(valid_detections) == 0:
            tracks = self.tracker.update_tracks([], embeds=[])
        else:
            tracks = self.tracker.update_tracks(valid_detections, frame=frame_bgr)
            # Ensure newly initiated tracks survive interval detection steps
            for t in self.tracker.tracker.tracks:
                if t.state == TrackState.Tentative:
                    t.state = TrackState.Confirmed

        results = []

        # Generic unclassified labels that must never override a verified species
        GENERIC_LABELS = {
            "camouflaged_wildlife",
            "unknown",
            "unknown / low confidence",
            "animal",
            "unclassified",
            "wildlife",
        }

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            ltrb = [int(v) for v in track.to_ltrb()]
            raw_class_name = track.get_det_class() or "Animal"
            conf = float(track.get_det_conf() or 0.80)

            # Update persistence count
            self.track_history[track_id] = self.track_history.get(track_id, 0) + 1
            persistence = self.track_history[track_id]

            # Temporal class stabilization: maintain sliding window of predictions
            if track_id not in self.track_class_history:
                self.track_class_history[track_id] = []
            self.track_class_history[track_id].append((raw_class_name, conf))
            if len(self.track_class_history[track_id]) > self.history_window:
                self.track_class_history[track_id].pop(0)

            # Separate verified species history from generic fallback votes
            verified_weights: Dict[str, float] = {}
            generic_weights: Dict[str, float] = {}
            for c_name, c_conf in self.track_class_history[track_id]:
                c_lower = c_name.lower().strip()
                if c_lower in GENERIC_LABELS or "unknown" in c_lower:
                    generic_weights[c_name] = generic_weights.get(c_name, 0.0) + c_conf
                else:
                    verified_weights[c_name] = verified_weights.get(c_name, 0.0) + c_conf

            # If any verified species has been identified for this track, vote only among verified species
            if verified_weights:
                stable_class_name = max(verified_weights, key=verified_weights.get)
            else:
                stable_class_name = max(generic_weights, key=generic_weights.get) if generic_weights else raw_class_name

            results.append({
                "track_id": track_id,
                "bbox_ltrb": ltrb,
                "class_name": stable_class_name,
                "confidence": conf,
                "persistence": persistence,
            })

        return results

    def reset(self) -> None:
        """Reset internal tracker state between video clips."""
        self.tracker.delete_all_tracks()
        self.track_history.clear()
        self.track_class_history.clear()

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
            if "UNKNOWN" in cls_name.upper():
                label = f"ID: {track_id} | Camo Target (Score: {conf * 100:.0f}%)"
            else:
                label = f"ID: {track_id} | {cls_name.title()} ({conf * 100:.0f}%)"
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
