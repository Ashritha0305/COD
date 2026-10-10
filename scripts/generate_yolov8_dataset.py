"""
scripts/generate_yolov8_dataset.py
==================================
Converts MoCA-Video GT binary masks to YOLOv8 dataset format.

Strict Rules Followed:
1. Masks are binary (0 = background, 255 = animal).
2. Connected component analysis:
   - Noise specks (< 30px) filtered out.
   - Enclosing union bounding box for the target camouflaged animal.
3. Empty (0 px) and near-empty (< 0.05% area) masks excluded.
4. Output normalized coordinates: class_id x_center y_center width height in [0, 1].
5. Sequence-level split (deterministic seed=42) - ZERO frame leakage.
6. Original MoCA dataset is never modified; images are copied to:
     dataset/yolov8_wildlife/images/{train,val}/
     dataset/yolov8_wildlife/labels/{train,val}/
     dataset/yolov8_wildlife/data.yaml
7. Generates comprehensive class balance and split reports.
"""

import os
import sys
import shutil
import csv
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Ensure project root in python path
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

import config

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SOURCE_ROOT = Path(config.MOCA_TRAIN_DIR)
TARGET_ROOT = Path(_BASE_DIR) / "dataset" / "yolov8_wildlife"
MAPPING_CSV = Path(_BASE_DIR) / "outputs" / "sequence_class_mapping.csv"
BALANCE_REPORT = Path(_BASE_DIR) / "outputs" / "yolov8_class_balance_report.txt"


def load_sequence_class_mapping() -> Dict[str, str]:
    mapping = {}
    with open(MAPPING_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["sequence_name"]] = row["normalized_animal_class"]
    return mapping


def extract_animal_bounding_box(mask_path: Path, min_area_ratio: float = 0.0005) -> Tuple[bool, List[float], int]:
    """
    Extracts the bounding box for the camouflaged animal from GT binary mask.

    Returns:
    (is_valid, [x_c, y_c, w, h] normalized, fg_pixel_count)
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, [], 0

    H, W = mask.shape
    total_px = H * W
    bin_mask = (mask > 127).astype(np.uint8)
    fg_px = int(np.sum(bin_mask))

    if fg_px == 0:
        return False, [], 0

    if (fg_px / total_px) < min_area_ratio:
        # Near-empty annotation artifact
        return False, [], fg_px

    # Connected component analysis to filter noise
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)
    # stats: [x, y, w, h, area]

    valid_components = []
    # label 0 is background
    for lbl in range(1, num_labels):
        comp_area = stats[lbl, cv2.CC_STAT_AREA]
        # Ignore isolated noise fragments smaller than 30 pixels or smaller than 0.05% of main object
        if comp_area >= 30:
            valid_components.append(stats[lbl])

    if not valid_components:
        # If all components are smaller than 30px, check if sum area is valid
        if fg_px >= 50:
            x, y, w, h = cv2.boundingRect(bin_mask)
        else:
            return False, [], fg_px
    else:
        # Union bounding box across all significant animal components
        x_min = min(c[cv2.CC_STAT_LEFT] for c in valid_components)
        y_min = min(c[cv2.CC_STAT_TOP] for c in valid_components)
        x_max = max(c[cv2.CC_STAT_LEFT] + c[cv2.CC_STAT_WIDTH] for c in valid_components)
        y_max = max(c[cv2.CC_STAT_TOP] + c[cv2.CC_STAT_HEIGHT] for c in valid_components)

        x = x_min
        y = y_min
        w = x_max - x_min
        h = y_max - y_min

    # Ensure box bounds are valid
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    w = max(1, min(W - x, w))
    h = max(1, min(H - y, h))

    # Convert to YOLO normalized format: x_center, y_center, width, height in [0, 1]
    x_c = (x + w / 2.0) / W
    y_c = (y + h / 2.0) / H
    w_norm = w / W
    h_norm = h / H

    return True, [x_c, y_c, w_norm, h_norm], fg_px


def main():
    print("=" * 80)
    print("CREATING CUSTOM YOLOv8 WILDLIFE DATASET FROM MoCA")
    print("=" * 80)

    seq_map = load_sequence_class_mapping()
    all_classes = sorted(list(set(seq_map.values())))
    class_to_id = {cls_name: i for i, cls_name in enumerate(all_classes)}

    print(f"Total normalized wildlife classes: {len(all_classes)}")

    # Group sequences by normalized class
    class_to_seqs = defaultdict(list)
    for seq_name, cls_name in seq_map.items():
        class_to_seqs[cls_name].append(seq_name)

    # Sequence-level split:
    # Deterministic holdout of sequences for multi-sequence classes.
    # Single-sequence classes remain in train so the detector learns their visual patterns.
    val_sequences = set()
    train_sequences = set()

    for cls_name, seqs in class_to_seqs.items():
        if len(seqs) >= 2:
            # Deterministic selection of 1 sequence for validation
            # (sort to ensure consistency across systems)
            sorted_seqs = sorted(seqs)
            val_seq = sorted_seqs[-1]  # Hold out last sequence for val
            val_sequences.add(val_seq)
            for s in sorted_seqs[:-1]:
                train_sequences.add(s)
        else:
            # Single sequence class -> goes to training
            train_sequences.add(seqs[0])

    print(f"Split Summary: {len(train_sequences)} Training Sequences, {len(val_sequences)} Validation Sequences.")

    # Create target directories
    for split in ["train", "val"]:
        (TARGET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Statistics tracking
    stats = {
        "total_pairs_scanned": 0,
        "empty_masks": 0,
        "near_empty_masks": 0,
        "valid_train_samples": 0,
        "valid_val_samples": 0,
        "class_counts_train": defaultdict(int),
        "class_counts_val": defaultdict(int),
        "class_seqs_train": defaultdict(set),
        "class_seqs_val": defaultdict(set),
    }

    # Process all sequences
    for seq_folder in sorted(SOURCE_ROOT.iterdir()):
        if not seq_folder.is_dir():
            continue

        seq_name = seq_folder.name
        if seq_name not in seq_map:
            continue

        cls_name = seq_map[seq_name]
        cls_id = class_to_id[cls_name]
        split = "val" if seq_name in val_sequences else "train"

        img_dir = seq_folder / "Imgs"
        gt_dir = seq_folder / "GT"
        if not (img_dir.is_dir() and gt_dir.is_dir()):
            continue

        img_files = {f.stem: f for f in img_dir.glob("*.*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")}
        gt_files = {f.stem: f for f in gt_dir.glob("*.*") if f.suffix.lower() in (".png", ".jpg", ".jpeg")}

        common_stems = sorted(set(img_files.keys()) & set(gt_files.keys()))

        for stem in common_stems:
            stats["total_pairs_scanned"] += 1
            img_path = img_files[stem]
            mask_path = gt_files[stem]

            is_valid, bbox, fg_px = extract_animal_bounding_box(mask_path)

            if not is_valid:
                if fg_px == 0:
                    stats["empty_masks"] += 1
                else:
                    stats["near_empty_masks"] += 1
                continue

            # Construct unique filename for flat YOLO structure
            unique_name = f"{seq_name}_{stem}"
            dest_img = TARGET_ROOT / "images" / split / f"{unique_name}.jpg"
            dest_lbl = TARGET_ROOT / "labels" / split / f"{unique_name}.txt"

            # Copy image (convert to JPG if needed or copy direct)
            if img_path.suffix.lower() == ".jpg":
                shutil.copy2(img_path, dest_img)
            else:
                img_data = cv2.imread(str(img_path))
                cv2.imwrite(str(dest_img), img_data, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Write YOLO format label file: class_id x_c y_c w h
            x_c, y_c, w, h = bbox
            with open(dest_lbl, "w", encoding="utf-8") as lf:
                lf.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

            if split == "train":
                stats["valid_train_samples"] += 1
                stats["class_counts_train"][cls_name] += 1
                stats["class_seqs_train"][cls_name].add(seq_name)
            else:
                stats["valid_val_samples"] += 1
                stats["class_counts_val"][cls_name] += 1
                stats["class_seqs_val"][cls_name].add(seq_name)

    # Generate data.yaml
    yaml_path = TARGET_ROOT / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# YOLOv8 Wildlife Camouflage Dataset Configuration\n")
        f.write(f"path: {TARGET_ROOT.resolve().as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(all_classes)}\n")
        f.write("names:\n")
        for i, name in enumerate(all_classes):
            f.write(f"  {i}: {name}\n")

    print(f"Saved dataset configuration to {yaml_path}")

    # Generate Class Balance Report
    os.makedirs(os.path.dirname(BALANCE_REPORT), exist_ok=True)
    with open(BALANCE_REPORT, "w", encoding="utf-8") as f:
        f.write("=" * 105 + "\n")
        f.write("YOLOv8 WILDLIFE DATASET CLASS BALANCE & SPLIT REPORT\n")
        f.write("=" * 105 + "\n\n")
        f.write(f"Total Scanned Mask Pairs : {stats['total_pairs_scanned']}\n")
        f.write(f"Empty Masks Excluded     : {stats['empty_masks']}\n")
        f.write(f"Near-Empty Excluded      : {stats['near_empty_masks']}\n")
        f.write(f"Total Valid Training Imgs: {stats['valid_train_samples']}\n")
        f.write(f"Total Valid Val Imgs     : {stats['valid_val_samples']}\n")
        f.write(f"Total Generated Dataset  : {stats['valid_train_samples'] + stats['valid_val_samples']}\n")
        f.write(f"Total Classes            : {len(all_classes)}\n\n")

        f.write("=" * 105 + "\n")
        f.write(f"{'Class ID':<8} | {'Animal Class':<28} | {'Train Imgs':>10} | {'Val Imgs':>8} | {'Total Imgs':>10} | {'Status':<18}\n")
        f.write("-" * 105 + "\n")

        for i, cls_name in enumerate(all_classes):
            n_train = stats["class_counts_train"][cls_name]
            n_val = stats["class_counts_val"][cls_name]
            total = n_train + n_val
            status = "Standard" if total >= 50 else "Low-Data (<50 imgs)"
            f.write(f"{i:<8} | {cls_name:<28} | {n_train:>10} | {n_val:>8} | {total:>10} | {status:<18}\n")
        f.write("=" * 105 + "\n\n")

        f.write("Sequence Allocation:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Training Sequences ({len(train_sequences)}):\n")
        for s in sorted(train_sequences):
            f.write(f"  - {s}\n")
        f.write(f"\nValidation Sequences ({len(val_sequences)}):\n")
        for s in sorted(val_sequences):
            f.write(f"  - {s}\n")

    print(f"Generated class balance report: {BALANCE_REPORT}")
    print(f"Dataset successfully created in {TARGET_ROOT}")


if __name__ == "__main__":
    main()
