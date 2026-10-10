"""
scripts/train_wildlife_yolo_34class.py
======================================
Fine-tunes the Wildlife YOLOv8 model to include Class 33 ('leopard')
while preserving all 33 MoCA benchmark classes.

Starts from: runs/detect/wildlife_yolov8/weights/best_33class_backup.pt
Target output: runs/detect/wildlife_yolov8/weights/best.pt
"""

import os
import sys
import time
import shutil
import torch
from ultralytics import YOLO

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

DATA_YAML = os.path.join(_BASE_DIR, "dataset", "yolov8_wildlife", "data.yaml")
BASE_WEIGHTS = os.path.join(_BASE_DIR, "runs", "detect", "wildlife_yolov8", "weights", "best_33class_backup.pt")
RUNS_DIR = os.path.join(_BASE_DIR, "runs", "detect")
EXP_NAME = "wildlife_yolov8"


def main():
    print("=" * 80)
    print("FINE-TUNING CUSTOM YOLOv8 WITH LEOPARD (34 CLASSES)")
    print("=" * 80)

    device_arg = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_arg} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Starting Weights: {BASE_WEIGHTS}")
    print(f"Dataset: {DATA_YAML}")

    model = YOLO(BASE_WEIGHTS)

    epochs = 12
    batch_size = 16
    imgsz = 512
    lr0 = 0.001

    print(f"Hyperparameters: Epochs={epochs}, Batch={batch_size}, ImgSz={imgsz}, LR0={lr0}")

    t0 = time.time()
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device_arg,
        project=RUNS_DIR,
        name=EXP_NAME,
        exist_ok=True,
        save=True,
        plots=True,
        workers=2,
        lr0=lr0,
        verbose=True,
    )

    elapsed = (time.time() - t0) / 60.0
    print(f"\n[Training Complete] Elapsed: {elapsed:.2f} minutes")

    best_pt = os.path.join(RUNS_DIR, EXP_NAME, "weights", "best.pt")
    if os.path.exists(best_pt):
        print(f"Verified new weights at: {best_pt}")
        v_model = YOLO(best_pt)
        print("Trained Model Classes:")
        print(v_model.names)
    else:
        raise FileNotFoundError(f"Expected {best_pt} not found!")


if __name__ == "__main__":
    main()
