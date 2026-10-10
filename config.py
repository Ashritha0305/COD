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

# Optical Flow target working resolution for video pipeline speedup
OPTICAL_FLOW_WIDTH  = 640
OPTICAL_FLOW_HEIGHT = 360

# Synchronous debug PNG writes (cv2.imwrite of debug_unet_probability.png and mask.png)
# Set to False during video inference to avoid synchronous disk locks.
SAVE_DEBUG_MASKS    = False

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
RUN_UNET_EVERY_FRAME = True

# Whether to run Optical Flow on every consecutive frame pair.
RUN_FLOW_EVERY_FRAME = True

# Processing mode for CamouflageProcessor and run_pipeline.py.
PROCESSING_MODE = "video"  # "video" | "image"

# ============================================================
# VIDEO PROCESSING OPTIMIZATIONS (PHASE 1 LATENCY REDUCTION)
# ============================================================
VIDEO_DETECTION_INTERVAL = 3   # Run YOLOv8 detection every N frames (DeepSORT tracks in-between)
VIDEO_DEPTH_INTERVAL     = 5   # Run ZoeDepth estimation every N frames (reused in-between)
VIDEO_UNET_INTERVAL      = 2   # Run U-Net segmentation every N frames (reused in-between)
VIDEO_FLOW_INTERVAL      = 1   # Run Optical Flow every N frames

VIDEO_PROCESS_WIDTH      = 960 # Intermediate processing resolution width for video
VIDEO_PROCESS_HEIGHT     = 540 # Intermediate processing resolution height for video

# ============================================================
# VISUALIZATION
# ============================================================

VIZ_OVERLAY_ALPHA = 0.4   # Alpha for mask overlay drawn on original frame.
VIZ_FLOW_COLORMAP = "jet"  # Matplotlib colormap name for flow magnitude images.

# ============================================================
# LOGGING
# ============================================================

LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")

# ============================================================
# YOLOV8 / YOLO-WORLD WILDLIFE DETECTION
# ============================================================
YOLO_BASE_WEIGHTS = os.path.join(BASE_DIR, "yolov8n.pt")
YOLO_WILDLIFE_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "wildlife_yolov8", "weights", "best.pt")

# Detection Mode:
#   "world"  : Open-Vocabulary YOLO-World (detects and names any animal beyond the 34 MoCA classes)
#   "custom" : Closed 34-class MoCA wildlife model (intended project detector)
YOLO_MODE = "custom"
YOLO_WORLD_WEIGHTS = "yolov8s-worldv2.pt"

YOLO_CONF_THRESH = 0.15       # Camouflage inherently reduces raw confidence
YOLO_IOU_THRESH = 0.45
YOLO_UNKNOWN_THRESH = 0.20   # Detections below this are marked as Unknown/Low Confidence

# Open-Vocabulary Wildlife Dictionary for YOLO-World
# Includes all 34 MoCA camouflage animals + global predators, mammals, reptiles, birds, marine, and insects
OPEN_WILDLIFE_CLASSES = [
    # --- MoCA Camouflage Species ---
    "crab", "cuttlefish", "devil_scorpionfish", "eastern_screech_owl", "egyptian_nightjar",
    "elephant", "flatfish", "flounder", "grasshopper", "hedgehog", "hermit_crab",
    "jerboa", "leaf_tail_gecko", "lichen_katydid", "markhor", "mountain_goat",
    "nile_monitor", "octopus", "orchid_mantis", "pallas_cat", "peacock_flounder",
    "plaice", "polar_bear", "potoo", "pygmy_seahorse", "rusty_spotted_cat",
    "scorpionfish", "seal", "shrimp", "smallfish", "snow_leopard", "sole",
    "spider_tailed_horned_viper", "leopard",
    # --- Apex Predators & Dangerous Wildlife ---
    "lion", "tiger", "cheetah", "jaguar", "panther", "cougar", "puma", "wolf", "hyena",
    "grizzly_bear", "brown_bear", "black_bear", "bear", "crocodile", "alligator", "caiman",
    "coyote", "wolverine", "rhino", "hippopotamus", "buffalo", "bison", "wild_boar", "boar",
    # --- Venomous & Camouflaged Reptiles / Amphibians ---
    "snake", "viper", "pit_viper", "rattlesnake", "cobra", "python", "mamba", "anaconda",
    "chameleon", "lizard", "gecko", "iguana", "monitor_lizard", "komodo_dragon", "horned_lizard",
    "toad", "frog", "bullfrog", "tree_frog", "poison_dart_frog", "salamander", "tortoise", "turtle",
    # --- Birds of Prey & Camouflaged Birds ---
    "owl", "barn_owl", "screech_owl", "snowy_owl", "great_horned_owl", "eagle", "hawk", "falcon",
    "vulture", "nightjar", "bittern", "heron", "egret", "ptarmigan", "partridge", "grouse", "pheasant", "bird",
    # --- Mammals (Herbivores, Ungulates & Small Mammals) ---
    "deer", "elk", "moose", "antelope", "gazelle", "impala", "zebra", "giraffe", "monkey", "baboon",
    "fox", "arctic_fox", "red_fox", "fennec_fox", "rabbit", "hare", "arctic_hare", "squirrel",
    "badger", "otter", "meerkat", "mongoose", "sloth", "armadillo", "pangolin", "porcupine", "bat",
    # --- Aquatic & Marine Camouflage ---
    "shark", "stingray", "manta_ray", "squid", "stonefish", "lionfish", "frogfish", "seahorse",
    "pipefish", "lobster", "sea_lion", "walrus",
    # --- Camouflaged Insects & Arthropods ---
    "praying_mantis", "mantis", "stick_insect", "leaf_insect", "katydid", "cicada", "moth",
    "peppered_moth", "caterpillar", "beetle", "spider", "tarantula", "scorpion",
    # --- General Fallbacks ---
    "wildlife", "animal", "predator", "reptile"
]

# ============================================================
# RULE-BASED THREAT ANALYSIS (CENTRALIZED CONFIG)
# ============================================================
# Apex predators and dangerous large wildlife
HIGH_THREAT_ANIMALS = {
    "snow_leopard", "spider_tailed_horned_viper", "polar_bear",
    "elephant", "nile_monitor", "devil_scorpionfish", "scorpionfish",
    "bear", "grizzly_bear", "brown_bear", "black_bear", "leopard",
    "jaguar", "lion", "tiger", "wolf", "cheetah", "panther", "cougar", "puma",
    "hyena", "crocodile", "alligator", "caiman", "snake", "viper", "pit_viper",
    "cobra", "mamba", "rattlesnake", "python", "anaconda", "rhino", "hippopotamus",
    "shark", "stonefish", "lionfish", "komodo_dragon", "wolverine"
}

# Medium-risk wildlife
MEDIUM_THREAT_ANIMALS = {
    "markhor", "mountain_goat", "seal", "pallas_cat", "rusty_spotted_cat",
    "eastern_screech_owl", "cuttlefish", "octopus", "hedgehog",
    "deer", "elk", "moose", "zebra", "boar", "wild_boar", "bison", "buffalo",
    "monkey", "baboon", "fox", "arctic_fox", "red_fox", "coyote", "sea_lion", "walrus",
    "monitor_lizard", "badger", "porcupine", "eagle", "vulture"
}

# Low-threat wildlife
LOW_THREAT_ANIMALS = {
    "crab", "hermit_crab", "shrimp", "smallfish", "flatfish", "flounder",
    "peacock_flounder", "plaice", "sole", "grasshopper", "lichen_katydid",
    "orchid_mantis", "praying_mantis", "mantis", "stick_insect", "leaf_insect",
    "katydid", "jerboa", "leaf_tail_gecko", "chameleon", "gecko", "lizard",
    "pygmy_seahorse", "seahorse", "pipefish", "egyptian_nightjar", "nightjar",
    "potoo", "owl", "bird", "ptarmigan", "grouse", "partridge", "pheasant",
    "rabbit", "hare", "squirrel", "sloth", "armadillo", "turtle", "tortoise",
    "frog", "toad", "salamander", "moth", "caterpillar", "sheep", "horse", "dog", "cat",
    "animal", "wildlife"
}

THREAT_MOTION_LOW = 0.15
THREAT_MOTION_HIGH = 0.40
THREAT_CONF_CRITICAL = 0.50
THREAT_PERSISTENCE_CRITICAL = 5


