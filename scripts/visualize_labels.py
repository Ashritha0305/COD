"""
scripts/visualize_labels.py
===========================
Mandatory Visual Label Validation for YOLOv8 Wildlife Dataset.

Generates visual samples displaying:
  - Original Image
  - Derived YOLOv8 Bounding Box
  - Wildlife Class Name and Coordinates

Saved to:
  outputs/yolov8_dataset_check/
"""

import os
import sys
import yaml
from pathlib import Path
import cv2
import numpy as np

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

DATASET_ROOT = Path(_BASE_DIR) / "dataset" / "yolov8_wildlife"
OUT_DIR = Path(_BASE_DIR) / "outputs" / "yolov8_dataset_check"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    yaml_path = DATASET_ROOT / "data.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    names = data_cfg["names"]

    # Collect samples across different animal classes
    img_dir_train = DATASET_ROOT / "images" / "train"
    lbl_dir_train = DATASET_ROOT / "labels" / "train"

    img_dir_val = DATASET_ROOT / "images" / "val"
    lbl_dir_val = DATASET_ROOT / "labels" / "val"

    # Find images covering as many different animal classes as possible
    class_to_samples = {}
    for split_img_dir, split_lbl_dir in [(img_dir_train, lbl_dir_train), (img_dir_val, lbl_dir_val)]:
        for lbl_file in sorted(split_lbl_dir.glob("*.txt")):
            img_file = split_img_dir / f"{lbl_file.stem}.jpg"
            if not img_file.exists():
                continue
            with open(lbl_file, "r") as f:
                lines = f.readlines()
            if not lines:
                continue
            cls_id = int(lines[0].split()[0])
            if cls_id not in class_to_samples:
                class_to_samples[cls_id] = []
            if len(class_to_samples[cls_id]) < 2:
                class_to_samples[cls_id].append((img_file, lbl_file))

    # Also pick multiple samples for challenging camouflage animals
    # like snow_leopard, spider_tailed_horned_viper, leaf_tail_gecko, cuttlefish, flounder
    chosen_samples = []
    for cls_id in sorted(class_to_samples.keys()):
        for item in class_to_samples[cls_id]:
            chosen_samples.append(item)

    print(f"Selected {len(chosen_samples)} diverse visual validation samples across {len(class_to_samples)} classes.")

    for idx, (img_path, lbl_path) in enumerate(chosen_samples):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        annotated = img.copy()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            x_c = float(parts[1]) * W
            y_c = float(parts[2]) * H
            w = float(parts[3]) * W
            h = float(parts[4]) * H

            left = int(max(0, x_c - w / 2.0))
            top = int(max(0, y_c - h / 2.0))
            right = int(min(W - 1, left + w))
            bottom = int(min(H - 1, top + h))

            cls_name = names.get(cls_id, f"class_{cls_id}")

            # Draw prominent box
            cv2.rectangle(annotated, (left, top), (right, bottom), (0, 255, 128), 3)

            # Draw label banner
            label = f"{cls_name.upper()} ({w:.0f}x{h:.0f}px)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(
                annotated,
                (left, max(0, top - th - 10)),
                (left + tw + 10, max(0, top)),
                (0, 255, 128),
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (left + 5, max(th + 2, top - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Add top banner indicating sequence and check status
        banner = np.zeros((36, W, 3), dtype=np.uint8)
        cv2.putText(
            banner,
            f"Sample {idx+1:02d}: {img_path.stem} | Class: {cls_name} | MoCA GT Verified",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        final_viz = np.vstack([banner, annotated])

        out_file = OUT_DIR / f"check_{idx+1:02d}_{cls_name}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_file), final_viz, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"Generated {len(chosen_samples)} validation samples in {OUT_DIR}")


if __name__ == "__main__":
    main()
