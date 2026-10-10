"""
scripts/add_leopard_data.py
===========================
Generates camouflaged leopard training and validation samples with accurate bounding boxes,
adds class 33 ('leopard') to dataset/yolov8_wildlife/data.yaml.
"""

import os
import sys
import random
import cv2
import numpy as np

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_IMG_PATH = os.environ.get("LEOPARD_SRC_IMG", os.path.join(_BASE_DIR, "data", "leopard_sample.jpg"))
DATASET_DIR = os.path.join(_BASE_DIR, "dataset", "yolov8_wildlife")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

CLASS_ID = 33
CLASS_NAME = "leopard"

# Ground truth box for the base image: [x1, y1, x2, y2]
BASE_BOX = [215, 150, 435, 275]


def augment_and_save(img, box, out_img_path, out_lbl_path, aug_type="orig"):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    aug_img = img.copy()

    if aug_type == "orig":
        pass
    elif aug_type == "hflip":
        aug_img = cv2.flip(img, 1)
        x1_new = W - x2
        x2_new = W - x1
        x1, x2 = x1_new, x2_new
    elif aug_type == "bright_up":
        aug_img = np.clip(img.astype(np.float32) * 1.25 + 15, 0, 255).astype(np.uint8)
    elif aug_type == "bright_down":
        aug_img = np.clip(img.astype(np.float32) * 0.75 - 10, 0, 255).astype(np.uint8)
    elif aug_type == "contrast_up":
        aug_img = cv2.convertScaleAbs(img, alpha=1.3, beta=0)
    elif aug_type == "contrast_down":
        aug_img = cv2.convertScaleAbs(img, alpha=0.7, beta=10)
    elif aug_type == "blur":
        aug_img = cv2.GaussianBlur(img, (5, 5), 1.5)
    elif aug_type == "hsv_sat":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif aug_type == "hsv_desat":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.7, 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif aug_type == "hsv_warm":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + 5, 0, 179)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif aug_type.startswith("shift_"):
        parts = aug_type.split("_")
        dx, dy = int(parts[1]), int(parts[2])
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aug_img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        x1 = max(0, min(W, x1 + dx))
        x2 = max(0, min(W, x2 + dx))
        y1 = max(0, min(H, y1 + dy))
        y2 = max(0, min(H, y2 + dy))
    elif aug_type.startswith("scale_"):
        s = float(aug_type.split("_")[1])
        new_w, new_h = int(W * s), int(H * s)
        scaled = cv2.resize(img, (new_w, new_h))
        if s > 1.0:
            # Crop center
            cx, cy = new_w // 2, new_h // 2
            crop_x1 = max(0, cx - W // 2)
            crop_y1 = max(0, cy - H // 2)
            aug_img = scaled[crop_y1:crop_y1 + H, crop_x1:crop_x1 + W]
            x1 = max(0, min(W, int(x1 * s) - crop_x1))
            x2 = max(0, min(W, int(x2 * s) - crop_x1))
            y1 = max(0, min(H, int(y1 * s) - crop_y1))
            y2 = max(0, min(H, int(y2 * s) - crop_y1))
        else:
            # Pad
            aug_img = np.zeros_like(img)
            pad_x = (W - new_w) // 2
            pad_y = (H - new_h) // 2
            aug_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = scaled
            x1 = pad_x + int(x1 * s)
            x2 = pad_x + int(x2 * s)
            y1 = pad_y + int(y1 * s)
            y2 = pad_y + int(y2 * s)
    elif aug_type == "flip_bright":
        flipped = cv2.flip(img, 1)
        aug_img = np.clip(flipped.astype(np.float32) * 1.15 + 10, 0, 255).astype(np.uint8)
        x1_new = W - x2
        x2_new = W - x1
        x1, x2 = x1_new, x2_new
    elif aug_type == "flip_sat":
        flipped = cv2.flip(img, 1)
        hsv = cv2.cvtColor(flipped, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        x1_new = W - x2
        x2_new = W - x1
        x1, x2 = x1_new, x2_new

    # Clamp bbox
    x1 = max(0, min(W - 1, x1))
    x2 = max(1, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(1, min(H, y2))
    bw = x2 - x1
    bh = y2 - y1

    if bw < 10 or bh < 10:
        return False

    # Normalized YOLO format: class_id x_c y_c w h
    x_c = (x1 + x2) / (2.0 * W)
    y_c = (y1 + y2) / (2.0 * H)
    nw = bw / float(W)
    nh = bh / float(H)

    cv2.imwrite(out_img_path, aug_img)
    with open(out_lbl_path, "w") as f:
        f.write(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}\n")

    return True


def main():
    print(f"Loading source image from {SRC_IMG_PATH}...")
    img = cv2.imread(SRC_IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not read source image: {SRC_IMG_PATH}")

    train_img_dir = os.path.join(DATASET_DIR, "images", "train")
    train_lbl_dir = os.path.join(DATASET_DIR, "labels", "train")
    val_img_dir = os.path.join(DATASET_DIR, "images", "val")
    val_lbl_dir = os.path.join(DATASET_DIR, "labels", "val")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Define augmentation list
    train_augs = [
        "orig", "hflip", "bright_up", "bright_down", "contrast_up", "contrast_down",
        "blur", "hsv_sat", "hsv_desat", "hsv_warm", "flip_bright", "flip_sat",
        "shift_-25_-15", "shift_25_-15", "shift_-25_15", "shift_25_15",
        "shift_-40_0", "shift_40_0", "shift_0_-25", "shift_0_25",
        "scale_1.1", "scale_0.9", "scale_1.15", "scale_0.85",
        "shift_-15_-10", "shift_15_-10", "shift_-15_10", "shift_15_10",
        "shift_-30_-20", "shift_30_-20", "shift_-30_20", "shift_30_20"
    ]

    val_augs = [
        "orig_val", "hflip_val", "bright_up_val", "contrast_up_val",
        "shift_20_10_val", "scale_1.05_val", "hsv_warm_val", "flip_bright_val"
    ]

    print("Generating train samples...")
    train_count = 0
    for idx, aug in enumerate(train_augs):
        img_name = f"leopard_train_{idx:03d}.jpg"
        lbl_name = f"leopard_train_{idx:03d}.txt"
        aug_clean = aug.replace("_val", "")
        success = augment_and_save(
            img, BASE_BOX,
            os.path.join(train_img_dir, img_name),
            os.path.join(train_lbl_dir, lbl_name),
            aug_type=aug_clean
        )
        if success:
            train_count += 1

    print(f"Generated {train_count} training samples for class {CLASS_ID} ({CLASS_NAME}).")

    print("Generating val samples...")
    val_count = 0
    for idx, aug in enumerate(val_augs):
        img_name = f"leopard_val_{idx:03d}.jpg"
        lbl_name = f"leopard_val_{idx:03d}.txt"
        aug_clean = aug.replace("_val", "")
        success = augment_and_save(
            img, BASE_BOX,
            os.path.join(val_img_dir, img_name),
            os.path.join(val_lbl_dir, lbl_name),
            aug_type=aug_clean
        )
        if success:
            val_count += 1

    print(f"Generated {val_count} validation samples for class {CLASS_ID} ({CLASS_NAME}).")

    # Update data.yaml to 34 classes
    print("Updating data.yaml...")
    with open(DATA_YAML, "r") as f:
        lines = f.readlines()

    # Reconstruct data.yaml
    names_idx = -1
    for i, l in enumerate(lines):
        if l.strip().startswith("nc:"):
            lines[i] = "nc: 34\n"
        if l.strip().startswith("names:"):
            names_idx = i

    has_leopard = any("leopard" in l and "33:" in l for l in lines)
    if not has_leopard:
        lines.append("  33: leopard\n")

    with open(DATA_YAML, "w") as f:
        f.writelines(lines)

    print("data.yaml successfully updated with nc: 34 and 33: leopard.")


if __name__ == "__main__":
    main()
