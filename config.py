"""
config.py
=========
Central configuration for the wildlife-COD camouflage preprocessing module.

ALL tuneable parameters live here.
No absolute paths should be hard-coded anywhere else in the project.

Sections:
  - Paths
  - Reproducibility
  - U-Net Architecture
  - Training
  - Loss
  - Optical Flow (Farneback)
  - Fusion / Enhancement
  - Inference / CamouflageProcessor
  - Visualization
  - Logging
"""

import os

# ============================================================
# PATHS
# ============================================================

# Root of the project — all other paths are derived from this.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------- Dataset ---------------
# ---- DATASET DECISION UPDATE ----
# Dataset switched to: MoCA-Mask (Moving Camouflaged Animals + Pixel Masks)
#
# Why MoCA-Mask:
#   - Exclusively wildlife / camouflaged animals (67 categories)
#   - Video dataset — frames from 87 real wildlife video sequences
#   - Pixel-level binary masks (needed for U-Net training)
#   - Motion is inherent — ideal pairing with Optical Flow branch
#   - 87 sequences, ~22,939 annotated frames (every 5th frame labelled)
#
# MoCA (original, bounding boxes only):
#   https://www.robots.ox.ac.uk/~vgg/data/MoCA/
#
# MoCA-Mask (pixel masks — what we use):
#   https://drive.google.com/file/d/1fb_0LGLL_4IiNFnFpkqEKVGzpFBRsT9K/view
#
# Expected directory structure after extraction:
#   dataset/MoCA_Mask/
#   ├── TrainDataset/
#   │   ├── Imgs/   ← JPEG frames per video sequence
#   │   └── GT/     ← binary PNG masks (every 5th frame)
#   └── TestDataset/
#       ├── Imgs/
#       └── GT/
#
# Mask convention: grayscale PNG, 0 = background, 255 = camouflaged animal.
#
# NOTE: Update this path after downloading and extracting MoCA-Mask.
# Updated to match the actual downloaded structure:
#   dataset/MoCA_Video/TrainDataset_per_sq/<sequence>/{Imgs,GT}/
DATASET_ROOT   = os.path.join(BASE_DIR, "dataset", "MoCA_Video")
MOCA_TRAIN_DIR = os.path.join(DATASET_ROOT, "TrainDataset_per_sq")
MOCA_TEST_DIR  = os.path.join(DATASET_ROOT, "TestDataset_per_sq")  # placeholder — not yet used
MOCA_IMG_SUBDIR  = "Imgs"   # subfolder inside each sequence folder containing frames
MOCA_MASK_SUBDIR = "GT"     # subfolder inside each sequence folder containing PNG masks

# Processed / split data directories.
# Actual sub-structure will be determined during dataset inspection (Phase 2).
DATA_DIR  = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

# --------------- Outputs ---------------
OUTPUTS_DIR      = os.path.join(BASE_DIR, "outputs")
MASKS_OUT_DIR    = os.path.join(OUTPUTS_DIR, "masks")
FLOW_OUT_DIR     = os.path.join(OUTPUTS_DIR, "optical_flow")
ENHANCED_OUT_DIR = os.path.join(OUTPUTS_DIR, "enhanced")
METRICS_OUT_DIR  = os.path.join(OUTPUTS_DIR, "metrics")

# --------------- Checkpoints ---------------
CHECKPOINTS_DIR  = os.path.join(BASE_DIR, "checkpoints")
BEST_MODEL_PATH  = os.path.join(CHECKPOINTS_DIR, "unet_best.pth")
LAST_MODEL_PATH  = os.path.join(CHECKPOINTS_DIR, "unet_last.pth")

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

# Fraction of SEQUENCES held out for validation during training.
# Split is done at sequence level — no sequence appears in both splits.
# Adjust if you want more / fewer validation sequences.
VAL_FRACTION = 0.15  # ~10 of 67 sequences → ~10% of annotated frames

# ============================================================
# U-NET ARCHITECTURE
# ============================================================

UNET_IN_CHANNELS  = 3   # RGB input frame
UNET_OUT_CHANNELS = 1   # Binary segmentation probability map

# DIAGNOSIS FIX (RC-1): Reduce from 64 → 32 features.
# The F=64 model has ~31M parameters trained on ~3,053 samples — a ratio that
# guarantees memorisation. F=32 gives ~4M parameters, far more appropriate
# for this dataset size while preserving representational capacity.
# Architecture: 32 → 64 → 128 → 256 (bottleneck) → 128 → 64 → 32 → 1
UNET_INIT_FEATURES = 32

# Spatial dropout rate applied after each encoder DoubleConv block.
# nn.Dropout2d drops entire feature-map channels, providing stronger
# regularisation than per-pixel dropout for convolutional features.
# 0.2 = 20% channel dropout during training; disabled at eval time.
DROPOUT_RATE = 0.2

# ============================================================
# INPUT RESOLUTION
# ============================================================

# Resolution at which frames are fed into U-Net during training and inference.
# Must be divisible by 32 to align with U-Net's max-pool/up-sample stages.
# Adjust based on dataset native resolution and available GPU memory.
INPUT_HEIGHT = 256
INPUT_WIDTH  = 256

# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE    = 8      # Reduce if GPU OOM occurs.
LEARNING_RATE = 1e-4   # Initial LR for Adam optimizer.
# DIAGNOSIS FIX (RC-3, RC-4): Increased from 60→80 because we now stop much
# earlier via ReduceLROnPlateau + tighter early stopping. Extra headroom costs
# nothing when early stopping fires at the right epoch.
NUM_EPOCHS    = 80

VAL_INTERVAL  = 1      # Run validation every N epochs.
SAVE_INTERVAL = 5      # Save a periodic checkpoint every N epochs.

# Number of DataLoader worker processes.
# Set to 0 on Windows if multiprocessing errors occur.
NUM_WORKERS = 0

# L2 regularization via Adam weight_decay.
# With F=32 the model is much smaller; keep weight decay for additional stability.
WEIGHT_DECAY = 1e-4

# DIAGNOSIS FIX (RC-4): Reduced patience from 8 → 5.
# Val IoU drops immediately after epoch 12 in the training history (0.218→0.144).
# Patience=8 let the model train until epoch 50 making things consistently worse.
# Patience=5 fires within 5 epochs of peak val IoU.
EARLY_STOP_PATIENCE = 5

# Training samples with foreground fraction < this value are skipped.
# These are near-empty annotation errors (10 found in data inspection).
# Applied to TRAINING SPLIT ONLY -- val samples are never filtered.
MIN_FOREGROUND_FRAC = 0.001

# DIAGNOSIS FIX (RC-2): BCE pos_weight disabled.
# pos_weight=11.7 combined with Dice loss caused the model to predict >60% of
# pixels as foreground (seg_mean ≈ 0.631). The Tversky loss (below) already
# controls FP/FN balance via alpha/beta — adding pos_weight on top created
# conflicting and unstable gradients. Set to None to disable entirely.
BCE_POS_WEIGHT = None

# DIAGNOSIS FIX (RC-5): Tighter RandomResizedCrop scale range.
# The previous scale=(0.5, 1.0) could crop to 50% of the original area,
# frequently excluding tiny camouflaged animals from the crop entirely.
# scale=(0.7, 1.0) guarantees the crop covers ≥70% of the original area,
# dramatically reducing the probability of cropping out the animal.
RANDOM_CROP_SCALE_MIN = 0.7

# ============================================================
# LOSS FUNCTION
# ============================================================

# Options: "bce"        → Binary Cross-Entropy only
#          "dice"       → Dice Loss only
#          "bce_dice"   → Weighted sum of BCE + Dice
#          "tversky"    → Tversky Loss (recommended — controls FP/FN independently)
LOSS_TYPE    = "tversky"

# BCE / Dice weights (only used when LOSS_TYPE = "bce_dice")
BCE_WEIGHT   = 0.4
DICE_WEIGHT  = 0.6
DICE_SMOOTH  = 1.0     # Smoothing constant in Dice loss to avoid division by zero.

# DIAGNOSIS FIX (RC-2): Tversky loss parameters.
# Tversky loss = 1 - TP / (TP + alpha*FP + beta*FN)
# alpha=0.3 → relatively tolerant of false positives
# beta=0.7  → heavily penalises missing the animal (false negatives)
# This replaces the unstable BCE(pos_weight=11.7)+Dice combination that was
# driving seg_mean to 0.631. Tversky controls the FP/FN trade-off directly
# via interpretable parameters instead of through a brittle BCE weight.
TVERSKY_ALPHA  = 0.3   # FP weight  (lower = more tolerant of over-prediction)
TVERSKY_BETA   = 0.7   # FN weight  (higher = penalise missing the animal more)
TVERSKY_SMOOTH = 1.0   # Smoothing constant to prevent division by zero

# Phase 2: Size-Aware Loss Additions
LOSS_SIZE_AWARE = True
BOUNDARY_LOSS_WEIGHT = 0.1

# ============================================================
# OPTICAL FLOW — FARNEBACK (OpenCV)
# ============================================================
# Reference: cv2.calcOpticalFlowFarneback documentation.
# These are NOT neural-network weights — they are algorithm parameters.

OF_PYR_SCALE  = 0.5   # Pyramid scale between levels (< 1.0).
OF_LEVELS     = 3      # Number of pyramid levels.
OF_WINSIZE    = 15     # Averaging window size (larger → smoother, slower).
OF_ITERATIONS = 3      # Iterations at each pyramid level.
OF_POLY_N     = 5      # Size of the pixel neighbourhood for polynomial expansion.
OF_POLY_SIGMA = 1.2    # Std-dev of Gaussian used for polynomial expansion.
OF_FLAGS      = 0      # Additional flags (0 = default behaviour).

# ============================================================
# FUSION / ENHANCEMENT
# ============================================================
# Strategy: combine U-Net probability map and Farneback motion magnitude map
# into a single attention map, then use that map to enhance the original frame.
#
# These weights are STARTING POINTS for the baseline.
# They will be tuned after qualitative and quantitative evaluation.
# No performance claims are made before measurement.

FUSION_UNET_WEIGHT          = 0.7   # U-Net segmentation dominates attention.
FUSION_FLOW_WEIGHT          = 0.3   # Motion is a secondary cue (gated by U-Net).

# Gentle luminance lift — the PRIMARY brightness increase in the animal region.
# 0.15 → maximum 15% brightness increase.  Much more subtle than the old 0.3.
# Do NOT increase above 0.3 without visual inspection.
FUSION_ENHANCEMENT_STRENGTH = 0.15

# CLAHE local contrast enhancement.
# clipLimit: higher = more aggressive local contrast.  2.0 is a natural baseline.
# tileGridSize is fixed at 8x8 inside fusion.py.
FUSION_CLAHE_STRENGTH       = 2.0

# Maximum HSV saturation boost in the animal region [0, 1].
# 0.25 → 25% saturation increase where attention=1.0.
FUSION_SAT_BOOST            = 0.25

# Temporal EMA weight for the previous attention map.
# 0.7 → previous frame contributes 70%, current 30%.
# Higher = smoother highlight, less sensitive to single-frame noise.
# 0.0 = no temporal smoothing (independent frames).
FUSION_TEMPORAL_ALPHA       = 0.7

# ============================================================
# INFERENCE / CAMOUFLAGE PROCESSOR
# ============================================================

# Probability threshold above which a pixel is considered foreground.
SEG_THRESHOLD = 0.5

# Whether to run U-Net on every frame (True) or allow frame-skipping (False).
# Frame-skipping logic will be added only if performance testing requires it.
RUN_UNET_EVERY_FRAME = True

# Whether to run Optical Flow on every consecutive frame pair.
RUN_FLOW_EVERY_FRAME = True

# Processing mode for CamouflageProcessor and run_pipeline.py.
# "video" : full pipeline — U-Net + Optical Flow + Fusion with temporal smoothing.
#           Use for video sequences or camera streams.
# "image" : U-Net only — no optical flow (no previous frame to compare against),
#           no temporal EMA smoothing. Fusion falls back to seg-only attention.
#           Use for single still images.
# This flag sets the DEFAULT mode; it can be overridden per-call or via CLI.
PROCESSING_MODE = "video"  # "video" | "image"

# ============================================================
# VISUALIZATION
# ============================================================

VIZ_OVERLAY_ALPHA = 0.4   # Alpha for mask overlay drawn on original frame.
VIZ_FLOW_COLORMAP = "jet"  # Matplotlib colormap name for flow magnitude images.

# ============================================================
# LOGGING
# ============================================================

LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")
