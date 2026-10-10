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
    YOLO wrapper for wildlife and camouflaged object detection.
    Supports:
      1. Open-Vocabulary YOLO-World (detects and names any animal dynamically)
      2. Custom MoCA wildlife detector
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        conf_thresh: float = config.YOLO_CONF_THRESH,
        iou_thresh: float = config.YOLO_IOU_THRESH,
        unknown_conf_thresh: float = config.YOLO_UNKNOWN_THRESH,
        classes: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> None:
        from ultralytics import YOLO

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.unknown_conf_thresh = unknown_conf_thresh

        if mode is None:
            mode = getattr(config, "YOLO_MODE", "world")
        self.mode = mode

        if weights_path is None:
            if self.mode == "world":
                weights_path = getattr(config, "YOLO_WORLD_WEIGHTS", "yolov8s-worldv2.pt")
            else:
                weights_path = config.YOLO_WILDLIFE_WEIGHTS

        self.weights_path = weights_path

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("=" * 60)
        print("YOLO MODEL:")
        print(f"Weights: {weights_path} | Mode: {self.mode}")
        print("=" * 60)

        # Initialize Ultralytics model
        self.model = YOLO(weights_path)

        # If using YOLO-World open-vocabulary model, configure custom wildlife classes
        is_world_model = "world" in str(weights_path).lower() or self.mode == "world"
        if is_world_model:
            target_classes = classes or getattr(config, "OPEN_WILDLIFE_CLASSES", None)
            if target_classes:
                try:
                    clean_classes = [c.replace("_", " ").strip() for c in target_classes]
                    self.model.set_classes(clean_classes)
                    print(f"[WildlifeDetector] YOLO-World open-vocabulary active with {len(clean_classes)} wildlife species.")
                except Exception as e:
                    print(f"[WildlifeDetector] Warning setting YOLO-World classes: {e}")

        self.names = self.model.names
        print("CUSTOM YOLO CLASSES:")
        print(f"Total active classes: {len(self.names) if self.names else 0}")
        print("=" * 60)

        # Initialize vision classifier only if explicitly in world/auxiliary mode
        # In custom YOLOv8 mode (the active architecture), ResNet-50 is bypassed to eliminate overhead and prevent hallucinations.
        self.classifier = None
        self.classifier_categories = []
        self.classifier_transform = None

        if self.mode == "world":
            try:
                from torchvision.models import resnet50, ResNet50_Weights
                weights = ResNet50_Weights.DEFAULT
                self.classifier = resnet50(weights=weights).to(self.device).eval()
                self.classifier_categories = weights.meta["categories"]
                self.classifier_transform = weights.transforms()
                print(f"[WildlifeDetector] Auxiliary vision classifier loaded ({len(self.classifier_categories)} categories).")
            except Exception as clf_err:
                print(f"[WildlifeDetector] Vision classifier not available: {clf_err}")
                self.classifier = None
        else:
            print("[WildlifeDetector] Active mode 'custom': Auxiliary ResNet-50 classifier disabled per fixed architecture.")

        self.is_custom_wildlife = (self.mode == "custom")
        print(f"[WildlifeDetector] Detector initialized successfully on {self.device}.")

    def classify_crop(self, crop_bgr: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Classify a cropped animal region using the fine-grained vision model.
        Restricted specifically to animal classes (ImageNet indices 0..397).
        Returns (predicted_animal_class, confidence).
        """
        if self.classifier is None or crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            from PIL import Image
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            inp = self.classifier_transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.classifier(inp)
                probs = torch.softmax(out, dim=1)[0]
                # Filter specifically to animal kingdom classes (indices 0..397)
                animal_probs = probs[:398]
                sum_animal = float(animal_probs.sum().item())
                if sum_animal > 0.005:
                    rel_probs = animal_probs / animal_probs.sum()
                else:
                    rel_probs = animal_probs

                top5_prob, top5_catid = torch.topk(rel_probs, 5)

            for i in range(5):
                idx = top5_catid[i].item()
                cat_name = self.classifier_categories[idx].lower().strip()
                prob = float(top5_prob[i].item())
                clean_name = cat_name.replace(" ", "_")
                if prob >= 0.15:
                    return clean_name, prob

            top_idx = top5_catid[0].item()
            top_name = self.classifier_categories[top_idx].lower().strip().replace(" ", "_")
            return top_name, float(top5_prob[0].item())
        except Exception:
            return None

    def detect(
        self,
        frame_bgr: np.ndarray,
        filter_animals_only: bool = False,
    ) -> List[Tuple[List[float], float, str]]:
        """
        Run YOLO inference on a single BGR frame.

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
        print("\nYOLO DETECTIONS")
        print("-" * 35)

        if not results or len(results) == 0:
            print("No detections found.")
            print("-" * 35)
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            print("No detections found above threshold.")
            print("-" * 35)
            return detections

        names = self.model.names

        H, W = frame_bgr.shape[:2]
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            raw_name = names.get(cls_id, f"class_{cls_id}")
            
            # If confidence is below unknown threshold, do not invent/fabricate species
            if conf < self.unknown_conf_thresh:
                class_name = "UNKNOWN / LOW CONFIDENCE"
            else:
                class_name = str(raw_name).lower().strip().replace(" ", "_")

            # box.xywh is [x_center, y_center, width, height]
            x_c, y_c, w, h = box.xywh[0].tolist()
            left = max(0.0, min(float(W - 1), x_c - w / 2.0))
            top = max(0.0, min(float(H - 1), y_c - h / 2.0))
            w = max(1.0, min(float(W - left), w))
            h = max(1.0, min(float(H - top), h))

            # Bounding box xyxy format for logging
            x1 = round(left, 1)
            y1 = round(top, 1)
            x2 = round(left + w, 1)
            y2 = round(top + h, 1)

            print(f"Class ID:     {cls_id}")
            print(f"Class Name:   {class_name}")
            print(f"Confidence:   {conf:.4f} ({conf * 100:.1f}%)")
            print(f"Bounding Box: [{x1}, {y1}, {x2}, {y2}]")
            print("-" * 35)

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
            display_name = cls_name.replace("_", " ").title()
            label = f"{display_name} {conf * 100:.1f}%"
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

