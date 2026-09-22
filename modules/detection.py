"""
modules/detection.py
====================
YOLOv8 Wildlife Detection Module.

Architecture Position:
    Input -> U-Net -> Optical Flow -> Fusion -> [YOLOv8 Detection] -> DeepSORT -> ZoeDepth -> Threat Analysis

Contract:
    - Loads official YOLOv8 model (yolov8n.pt or user-configured weight)
    - Automatically targets NVIDIA RTX 4050 (CUDA) if available
    - Returns structured bounding boxes, confidence scores, and class labels
    - Provides standardized format for DeepSORT tracker: [([left, top, w, h], confidence, class_name), ...]
"""

import os
import sys
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


class WildlifeDetector:
    """
    YOLOv8 wrapper for wildlife and camouflaged object detection.
    """

    # COCO animal class indices and names
    ANIMAL_CLASSES = {
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
    }

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        device: Optional[str] = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ) -> None:
        from ultralytics import YOLO

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[WildlifeDetector] Loading YOLOv8 from '{weights_path}' on {self.device}...")
        self.model = YOLO(weights_path)
        print(f"[WildlifeDetector] YOLOv8 successfully initialized on {self.device}.")

    def detect(
        self,
        frame_bgr: np.ndarray,
        filter_animals_only: bool = False,
    ) -> List[Tuple[List[float], float, str]]:
        """
        Run YOLOv8 inference on a single BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray (H, W, 3) uint8 BGR
        filter_animals_only : bool, if True only keeps COCO animal classes

        Returns
        -------
        detections : list of ([left, top, width, height], confidence, class_name)
                     Format matching deep_sort_realtime expectation.
        """
        results = self.model(
            frame_bgr,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names = self.model.names  # class id -> name mapping

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = names.get(cls_id, f"class_{cls_id}")

            if filter_animals_only and cls_id not in self.ANIMAL_CLASSES:
                continue

            # box.xywh is [x_center, y_center, width, height]
            # convert to [left, top, width, height] for DeepSORT
            x_c, y_c, w, h = box.xywh[0].tolist()
            left = max(0.0, x_c - w / 2.0)
            top = max(0.0, y_c - h / 2.0)

            detections.append(([left, top, w, h], conf, class_name))

        return detections

    def draw_detections(
        self,
        frame_bgr: np.ndarray,
        detections: List[Tuple[List[float], float, str]],
        color: Tuple[int, int, int] = (0, 255, 128),
    ) -> np.ndarray:
        """Draw bounding boxes and class labels onto a copy of the frame."""
        out = frame_bgr.copy()
        H, W = out.shape[:2]

        for (box, conf, cls_name) in detections:
            left, top, w, h = [int(v) for v in box]
            right = min(W - 1, left + w)
            bottom = min(H - 1, top + h)

            cv2.rectangle(out, (left, top), (right, bottom), color, 2)
            label = f"{cls_name.capitalize()} {conf * 100:.1f}%"
            cv2.putText(
                out,
                label,
                (left, max(18, top - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        return out
