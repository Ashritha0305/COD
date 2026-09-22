# wildlife-COD

**Real-Time Camouflage-Aware Wildlife Detection with Distance-Based Threat Analysis**

Final-Year Engineering Project — Member 1 Repository (Camouflage Preprocessing Module)

---

## Project Overview

This repository contains **Member 1's contribution**: the camouflage-aware preprocessing module that forms the first stage of a four-member real-time wildlife detection pipeline.

### Full System Pipeline

```
Input: Live Camera / Video
        ↓
[THIS MODULE] U-Net + Optical Flow  ←── Member 1
(Camouflage-aware preprocessing)
        ↓
YOLOv8                              ←── Member 2
(Animal Detection)
        ↓
DeepSORT                            ←── Member 3
(Multi-Object Tracking)
        ↓
ZoeDepth + Threat Analysis          ←── Member 4
(Distance Estimation + SAFE/WARNING/CRITICAL)
        ↓
Real-Time Alert
```

### Member 1 Responsibility

- Train a **U-Net** for camouflaged animal segmentation on the **CamoVid60K** dataset
- Implement **OpenCV Farneback Dense Optical Flow** for inter-frame motion analysis
- Implement a configurable **fusion / enhancement** strategy combining both outputs
- Expose a clean **`CamouflageProcessor`** interface for Member 2's YOLOv8 module

This module does **not** implement YOLOv8, DeepSORT, ZoeDepth, or threat analysis.

---

## Project Structure

```
wildlife-COD/
│
├── modules/
│   └── camouflage.py            # CamouflageProcessor class (Phase 6)
│
├── models/
│   └── unet/                    # U-Net architecture definition
│
├── utils/
│   ├── preprocessing.py         # Frame preprocessing utilities
│   ├── visualization.py         # Mask / flow / enhanced frame visualisation
│   └── metrics.py               # IoU, Dice, Precision, Recall
│
├── scripts/
│   ├── verify_environment.py    # Phase 1 — environment check
│   ├── inspect_dataset.py       # Phase 2 — CamoVid60K inspection
│   ├── train_unet.py            # Phase 3 — U-Net training
│   ├── evaluate_unet.py         # Phase 3 — U-Net evaluation
│   ├── test_unet.py             # Phase 3 — U-Net qualitative test
│   ├── test_optical_flow.py     # Phase 4 — Optical Flow test
│   └── test_camouflage_pipeline.py  # Phase 5/6 — Full module test
│
├── data/
│   ├── CamoVid60K/              # Raw dataset (not committed to git)
│   ├── train/
│   ├── val/
│   └── test/
│
├── outputs/
│   ├── masks/
│   ├── optical_flow/
│   ├── enhanced/
│   └── metrics/
│
├── checkpoints/                 # Saved U-Net weights (not committed to git)
│
├── config.py                    # All configurable parameters
├── requirements.txt             # Python dependencies
├── README.md
└── .gitignore
```

---

## Setup Instructions

### 1. Clone / create the environment

```bash
# Windows PowerShell / Command Prompt
python -m venv venv
venv\Scripts\activate
```

### 2. Install PyTorch (GPU — choose the correct CUDA version)

```bash
# CUDA 12.1 (most current NVIDIA GPUs):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU-only (slow, for development without GPU):
pip install torch torchvision torchaudio
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the environment

```bash
python scripts/verify_environment.py
```

All checks must **PASS** before proceeding.

---

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | Environment setup |
| 2 | ⬜ Pending | CamoVid60K dataset inspection |
| 3 | ⬜ Pending | U-Net implementation & training |
| 4 | ⬜ Pending | Farneback Optical Flow |
| 5 | ⬜ Pending | Fusion / enhancement |
| 6 | ⬜ Pending | `CamouflageProcessor` class |
| 7 | ⬜ Pending | YOLOv8 handoff interface |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| U-Net weights | Trained from scratch on CamoVid60K | No verified camouflage-segmentation checkpoint available; generic ImageNet weights are insufficient for this task |
| Optical Flow | OpenCV Farneback (classical, no training) | Real-time compatible, no additional model required, interpretable |
| Fusion | Configurable weighted combination (baseline) | Simple, explainable baseline; tunable via `config.py` |

---

## Research Honesty

- This project does **not** claim to invent U-Net, Optical Flow, YOLOv8, DeepSORT, or ZoeDepth.
- The contribution is the **integration** of these components into a unified camouflage-aware wildlife monitoring pipeline.
- No FPS or accuracy numbers are promised before experimental measurement.
- No pretrained weights are fabricated or invented.

---

## Dataset

**CamoVid60K** — Moving Camouflaged Animal Video Dataset
- 218 videos, ~62,774 annotated frames, 70 animal categories
- Provides segmentation masks and bounding-box annotations
- Video-disjoint train/test split (prevents temporal leakage)

> Dataset paths and exact directory structure will be confirmed during Phase 2 (dataset inspection).

---

## Interface to Member 2 (YOLOv8)

```python
from modules.camouflage import CamouflageProcessor

processor = CamouflageProcessor(config_path="config.py")

result = processor.process(current_frame, previous_frame)

# Pass to Member 2:
enhanced_frame = result["enhanced_frame"]  # Standard BGR/RGB NumPy array
```

---

## Performance Measurement

After implementation, the following will be measured experimentally:
- U-Net inference time (ms)
- Optical Flow processing time (ms)
- Fusion time (ms)
- Total module latency (ms)
- Achievable FPS

No values are fabricated or promised before measurement.
