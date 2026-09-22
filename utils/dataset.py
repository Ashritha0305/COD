"""
utils/dataset.py
=================
PyTorch Dataset for MoCA-Video (Moving Camouflaged Animals with pixel masks).

Actual dataset structure
------------------------
    dataset/MoCA_Video/TrainDataset_per_sq/
    ├── crab/
    │   ├── Imgs/
    │   │   ├── 00000.jpg
    │   │   ├── 00005.jpg
    │   │   └── ...
    │   └── GT/
    │       ├── 00000.png
    │       ├── 00005.png
    │       └── ...
    ├── elephant/
    │   ├── Imgs/
    │   └── GT/
    └── ...   (67 sequences total)

Pairing logic (stem-based, NOT positional)
------------------------------------------
For each sequence folder <seq>:
  - Collect image files from <seq>/Imgs/  (extensions: .jpg, .jpeg, .png)
  - Collect mask files  from <seq>/GT/    (extensions: .png, .jpg, .jpeg)
  - Build a stem->path map for each side
  - Valid pair: stem present in BOTH maps  ->  kept
  - Stem only in images                   ->  unmatched image (skipped)
  - Stem only in masks                    ->  unmatched mask  (skipped)

Known example: spider_tailed_horned_viper_2
  Imgs: 21 files  |  GT: 35 files  ->  21 valid pairs, 14 unmatched GT masks

Train / Validation split (SEQUENCE-LEVEL)
-----------------------------------------
  - Sequences are shuffled with a fixed random seed (config.SEED = 42).
  - The bottom VAL_FRACTION fraction of sequences becomes validation.
  - A sequence NEVER appears in both splits.
  - All frames of a sequence go to the same split.
  - Split fraction is configurable via config.VAL_FRACTION.

Empty-mask filtering
---------------------
  Training samples with foreground fraction < config.MIN_FOREGROUND_FRAC
  are excluded. These are annotation errors (10 found in data inspection,
  contributing noise gradients). Validation samples are NEVER filtered.

Augmentation (training only)
-----------------------------
  Applied jointly to image and mask:
    1. RandomResizedCrop(scale=0.5..1.0, ratio=0.8..1.2) — scale invariance.
       Camouflaged animals appear at vastly different sizes; the model must
       learn to find them regardless of scale.  This REPLACES the fixed resize
       in training (val still uses fixed resize).
    2. Horizontal flip (50%) — left-right symmetry.
    3. Random rotation +-10 degrees (40%) — mild orientation augmentation.
  Applied to image only:
    4. ColorJitter (brightness, contrast, saturation, hue) — appearance variation.
    5. GaussianBlur (30%, sigma 0.1..1.5) — blur robustness.
    6. RandomGrayscale (5%) — prevents over-reliance on colour cues.
  Removed: vertical flip (inappropriate for natural wildlife scenes).

Usage
-----
    from utils.dataset import get_loaders, discover_sequences
    train_loader, val_loader = get_loaders()
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


# ---------------------------------------------------------------------------
# Data-structure types
# ---------------------------------------------------------------------------

class SampleInfo(NamedTuple):
    """One image-mask pair, tagged with its source sequence."""
    img_path:      Path
    mask_path:     Path
    sequence_name: str


class SequenceInfo(NamedTuple):
    """Aggregated information for one video sequence."""
    name:            str
    pairs:           List[Tuple[Path, Path]]  # (img_path, mask_path) in stem order
    unmatched_imgs:  List[Path]               # images with no matching mask
    unmatched_masks: List[Path]               # masks  with no matching image


# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------

_IMG_EXTENSIONS  = frozenset({".jpg", ".jpeg", ".png"})
_MASK_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------

def discover_sequences(train_dir: Path) -> Tuple[List[SequenceInfo], int]:
    """
    Scan *train_dir* for per-sequence folders and build stem-paired sample lists.

    Parameters
    ----------
    train_dir : Path
        Path to the per-sequence root, e.g.
        ``dataset/MoCA_Video/TrainDataset_per_sq/``.

    Returns
    -------
    sequences : list[SequenceInfo]
        Only sequences that have *both* ``Imgs/`` and ``GT/`` directories
        **and** at least one valid image-mask pair.
    n_folders : int
        Total number of immediate subdirectories found (including invalid ones),
        useful for reporting how many folders were examined.
    """
    if not train_dir.exists():
        raise FileNotFoundError(
            f"[discover_sequences] Training directory not found:\n  {train_dir}\n\n"
            f"Expected structure:\n"
            f"  dataset/MoCA_Video/TrainDataset_per_sq/<sequence>/{{Imgs,GT}}/\n"
        )

    all_subdirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    n_folders   = len(all_subdirs)
    sequences: List[SequenceInfo] = []

    for seq_dir in all_subdirs:
        imgs_dir = seq_dir / config.MOCA_IMG_SUBDIR   # e.g. <seq>/Imgs
        gt_dir   = seq_dir / config.MOCA_MASK_SUBDIR  # e.g. <seq>/GT

        # A valid sequence must have both subdirectories.
        if not imgs_dir.is_dir() or not gt_dir.is_dir():
            continue

        # Build stem -> Path maps (case-insensitive extension check)
        img_map: Dict[str, Path] = {}
        for f in sorted(imgs_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in _IMG_EXTENSIONS:
                img_map[f.stem] = f

        mask_map: Dict[str, Path] = {}
        for f in sorted(gt_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in _MASK_EXTENSIONS:
                mask_map[f.stem] = f

        img_stems  = set(img_map.keys())
        mask_stems = set(mask_map.keys())

        common_stems    = sorted(img_stems & mask_stems)
        unmatched_img_s = sorted(img_stems  - mask_stems)
        unmatched_msk_s = sorted(mask_stems - img_stems)

        pairs           = [(img_map[s], mask_map[s]) for s in common_stems]
        unmatched_imgs  = [img_map[s]  for s in unmatched_img_s]
        unmatched_masks = [mask_map[s] for s in unmatched_msk_s]

        # Only keep sequences with at least one valid pair.
        if not pairs:
            continue

        sequences.append(SequenceInfo(
            name=seq_dir.name,
            pairs=pairs,
            unmatched_imgs=unmatched_imgs,
            unmatched_masks=unmatched_masks,
        ))

    return sequences, n_folders


# ---------------------------------------------------------------------------
# Sequence-level train / validation split
# ---------------------------------------------------------------------------

def sequence_level_split(
    sequences:    List[SequenceInfo],
    val_fraction: float,
    seed:         int,
) -> Tuple[List[SequenceInfo], List[SequenceInfo]]:
    """
    Shuffle sequences then deterministically split into train and validation.

    The same sequence NEVER appears in both outputs.

    Parameters
    ----------
    sequences : list[SequenceInfo]
        All valid sequences returned by :func:`discover_sequences`.
    val_fraction : float
        Fraction of sequences to hold out for validation
        (e.g. 0.15 -> 10 of 67 sequences).
    seed : int
        Fixed seed for reproducibility; use ``config.SEED`` (= 42).

    Returns
    -------
    train_seqs, val_seqs : list[SequenceInfo], list[SequenceInfo]
    """
    rng = random.Random(seed)
    shuffled = list(sequences)
    rng.shuffle(shuffled)

    n_val      = max(1, round(len(shuffled) * val_fraction))
    val_seqs   = shuffled[:n_val]
    train_seqs = shuffled[n_val:]
    return train_seqs, val_seqs


# ---------------------------------------------------------------------------
# Empty-mask filter
# ---------------------------------------------------------------------------

def _has_sufficient_foreground(
    mask_path: Path,
    min_frac:  float = config.MIN_FOREGROUND_FRAC,
) -> bool:
    """
    Return True if the mask has at least *min_frac* foreground pixels.

    Used to discard near-empty annotation errors from the TRAINING split only.
    Loading a grayscale PNG and calling np.mean is fast (~0.1ms per mask).
    """
    arr = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
    return float((arr > 127).mean()) >= min_frac


# ---------------------------------------------------------------------------
# Joint augmentation helpers  (image + mask, same spatial transform)
# ---------------------------------------------------------------------------

def _get_color_jitter() -> T.ColorJitter:
    return T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.07)


def _joint_augment(
    img:      Image.Image,
    mask:     Image.Image,
    img_size: Tuple[int, int],
) -> Tuple[Image.Image, Image.Image]:
    """
    Apply identical spatial augmentation to image and mask.

    Changes vs v1:
      - RandomResizedCrop REPLACES the fixed resize for training.
        The crop is sampled once and applied to both image (BILINEAR)
        and mask (NEAREST) at the target img_size.  This is the single
        most impactful augmentation for scale invariance.
      - Removed vertical flip — animals are almost never upside-down in
        wildlife footage; this augmentation was adding noise without signal.
      - Rotation angle reduced from +-15 to +-10 degrees and probability
        reduced from 50% to 40%.  Large rotations can crop out the animal.
    """
    H, W = img_size  # target output resolution

    # ---- RandomResizedCrop (replaces fixed resize for training) ----
    # DIAGNOSIS FIX (RC-5): scale min raised from 0.5 -> config.RANDOM_CROP_SCALE_MIN (0.7).
    # scale=0.5 could crop to 50% of the original area, frequently excluding tiny
    # camouflaged animals from the crop, producing noise gradients with no animal
    # in the crop. scale=0.7 guarantees >=70% of the image is covered, dramatically
    # reducing the probability of losing the animal in the crop window.
    # Ratio range 0.85..1.15 (tightened from 0.8..1.2) for mild aspect variation.
    crop_params = T.RandomResizedCrop.get_params(
        img, scale=(config.RANDOM_CROP_SCALE_MIN, 1.0), ratio=(0.85, 1.15)
    )
    i, j, h, w = crop_params
    img  = TF.resized_crop(img,  i, j, h, w, (H, W), interpolation=Image.BILINEAR)
    mask = TF.resized_crop(mask, i, j, h, w, (H, W), interpolation=Image.NEAREST)

    # ---- Horizontal flip ----
    if random.random() > 0.5:
        img  = TF.hflip(img)
        mask = TF.hflip(mask)

    # ---- Mild rotation (reduced from +-15° to +-10°, prob 0.4) ----
    if random.random() > 0.6:
        angle = random.uniform(-10, 10)
        img  = TF.rotate(img,  angle, fill=0)
        mask = TF.rotate(mask, angle, fill=0)

    return img, mask


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MoCAVideoDataset(Dataset):
    """
    PyTorch Dataset for one split (train or val) of MoCA-Video.

    Takes a pre-built flat list of :class:`SampleInfo` so that callers
    control the sequence-level split before constructing the dataset.
    Sequence identity is preserved on every sample and can be queried via
    :meth:`get_sequence_name`.

    Parameters
    ----------
    samples : list[SampleInfo]
        Flat list of (img_path, mask_path, sequence_name) named-tuples.
        For the TRAINING split, near-empty mask samples should already be
        filtered out by the caller (see :func:`get_loaders`).
    augment : bool
        Whether to apply joint spatial augmentation + colour jitter.
        Set ``True`` for training, ``False`` for validation / test.
    img_size : Tuple[int, int]
        Target ``(H, W)`` for all images and masks.
        Must be divisible by 32 for U-Net compatibility.
    """

    def __init__(
        self,
        samples:  List[SampleInfo],
        augment:  bool = False,
        img_size: Tuple[int, int] = (config.INPUT_HEIGHT, config.INPUT_WIDTH),
    ) -> None:
        super().__init__()
        if not samples:
            raise ValueError(
                "[MoCAVideoDataset] Empty sample list -- no image-mask pairs to load."
            )
        self.samples  = samples
        self.augment  = augment
        self.img_size = img_size

        self._color_jitter   = _get_color_jitter() if augment else None
        self._gaussian_blur  = T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)) if augment else None
        self._random_gray    = T.RandomGrayscale(p=0.05) if augment else None
        self._normalize      = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        )

    # ------------------------------------------------------------------
    # Sequence-level accessors
    # ------------------------------------------------------------------

    def get_sequence_name(self, idx: int) -> str:
        """Return the video sequence name for the sample at *idx*."""
        return self.samples[idx].sequence_name

    def get_sequence_names(self) -> List[str]:
        """Return a sorted list of unique sequence names in this dataset."""
        return sorted({s.sequence_name for s in self.samples})

    # ------------------------------------------------------------------
    # Standard Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # --- Load -----------------------------------------------------------
        img  = Image.open(sample.img_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")   # grayscale

        # --- Resize / Augment -----------------------------------------------
        if self.augment:
            # RandomResizedCrop replaces the fixed resize for training.
            # This teaches scale invariance — animals appear at all sizes.
            img, mask = _joint_augment(img, mask, self.img_size)
        else:
            # Validation: fixed deterministic resize, no augmentation.
            img  = img.resize( (self.img_size[1], self.img_size[0]), Image.BILINEAR)
            mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        # --- Colour augmentation (image only) -------------------------------
        if self.augment:
            # ColorJitter — brightness, contrast, saturation, hue variation
            img = self._color_jitter(img)

            # GaussianBlur — 30% probability; mild sigma range
            if random.random() < 0.3:
                img = self._gaussian_blur(img)

            # RandomGrayscale — 5% probability; prevents colour-only memorisation
            img = self._random_gray(img)

        # --- PIL -> tensor --------------------------------------------------
        img_t  = TF.to_tensor(img)                              # (3, H, W) float32 [0,1]
        mask_t = torch.from_numpy(
            np.array(mask, dtype=np.float32) / 255.0
        ).unsqueeze(0)                                          # (1, H, W) float32 [0,1]

        # Hard binarise mask (handles any resize / JPEG artefacts in masks)
        mask_t = (mask_t > 0.5).float()

        # ImageNet normalisation for the image
        img_t = self._normalize(img_t)

        return img_t, mask_t


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_loaders(
    val_fraction: float          = config.VAL_FRACTION,
    seed:         int            = config.SEED,
    train_dir:    Optional[Path] = None,
    filter_empty: bool           = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from ``TrainDataset_per_sq/``.

    The split is performed at **sequence** level: every frame of a given
    sequence goes entirely to train *or* validation -- never both.

    Parameters
    ----------
    val_fraction : float
        Fraction of sequences used for validation (default: ``config.VAL_FRACTION``).
    seed : int
        Random seed for reproducible sequence shuffle (default: ``config.SEED``).
    train_dir : Path, optional
        Override the per-sequence root directory.
        Defaults to ``config.MOCA_TRAIN_DIR``.
    filter_empty : bool
        If True (default), remove training samples where the foreground mask
        covers less than ``config.MIN_FOREGROUND_FRAC`` of pixels.
        These are near-empty annotation errors that corrupt gradients.
        Validation samples are NEVER filtered regardless of this flag.

    Returns
    -------
    train_loader, val_loader : DataLoader, DataLoader
    """
    if train_dir is None:
        train_dir = Path(config.MOCA_TRAIN_DIR)

    sequences, _ = discover_sequences(train_dir)
    train_seqs, val_seqs = sequence_level_split(sequences, val_fraction, seed)

    # Flatten sequences -> sample lists (sequence name preserved on every sample)
    train_samples_raw: List[SampleInfo] = [
        SampleInfo(img, mask, seq.name)
        for seq in train_seqs
        for img, mask in seq.pairs
    ]
    val_samples: List[SampleInfo] = [
        SampleInfo(img, mask, seq.name)
        for seq in val_seqs
        for img, mask in seq.pairs
    ]

    # Filter near-empty training masks (annotation errors, not valid data).
    # Val set is NEVER filtered — all val frames must be evaluated.
    if filter_empty:
        n_before = len(train_samples_raw)
        train_samples = [
            s for s in train_samples_raw
            if _has_sufficient_foreground(s.mask_path)
        ]
        n_removed = n_before - len(train_samples)
        if n_removed > 0:
            print(
                f"[get_loaders] Filtered {n_removed} near-empty training masks "
                f"(fg < {config.MIN_FOREGROUND_FRAC * 100:.1f}%)  "
                f"{n_before} -> {len(train_samples)} training samples."
            )
    else:
        train_samples = train_samples_raw

    train_ds = MoCAVideoDataset(train_samples, augment=True)
    val_ds   = MoCAVideoDataset(val_samples,   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        num_workers = config.NUM_WORKERS,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = True,
        drop_last   = False,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Dataset statistics helper (quick summary, no split)
# ---------------------------------------------------------------------------

def print_dataset_stats(train_dir: Optional[Path] = None) -> None:
    """Print a one-line summary of MoCA-Video sequences and pairs."""
    if train_dir is None:
        train_dir = Path(config.MOCA_TRAIN_DIR)
    print("=== MoCA-Video Dataset Statistics ===")
    try:
        sequences, n_folders = discover_sequences(train_dir)
        print(f"  Sequence folders found : {n_folders}")
        print(f"  Valid sequences        : {len(sequences)}")
        print(f"  Total image-mask pairs : {sum(len(s.pairs) for s in sequences)}")
        print(f"  Total unmatched images : {sum(len(s.unmatched_imgs) for s in sequences)}")
        print(f"  Total unmatched masks  : {sum(len(s.unmatched_masks) for s in sequences)}")
    except FileNotFoundError as e:
        print(f"  [NOT FOUND] {e}")


# ---------------------------------------------------------------------------
# Comprehensive self-test  (python utils/dataset.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    SEP = "=" * 65

    print(f"\n{SEP}")
    print("  MoCA-Video Dataset  --  Self-Test")
    print(SEP)

    train_dir = Path(config.MOCA_TRAIN_DIR)
    print(f"\n  Training dir : {train_dir}")
    print(f"  Exists       : {train_dir.exists()}\n")

    # ------------------------------------------------------------------
    # 1. Discover sequences
    # ------------------------------------------------------------------
    try:
        sequences, n_folders = discover_sequences(train_dir)
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)

    total_pairs           = sum(len(s.pairs)           for s in sequences)
    total_unmatched_imgs  = sum(len(s.unmatched_imgs)  for s in sequences)
    total_unmatched_masks = sum(len(s.unmatched_masks) for s in sequences)

    print(f"  Sequence folders found  : {n_folders}")
    print(f"  Valid sequences         : {len(sequences)}")
    print(f"  Total image-mask pairs  : {total_pairs}")
    print(f"  Total unmatched images  : {total_unmatched_imgs}")
    print(f"  Total unmatched masks   : {total_unmatched_masks}")

    # ------------------------------------------------------------------
    # 2. Known-problematic sequence check
    # ------------------------------------------------------------------
    TARGET_SEQ = "spider_tailed_horned_viper_2"
    target = next((s for s in sequences if s.name == TARGET_SEQ), None)
    print(f"\n  [Check] {TARGET_SEQ}:")
    if target is None:
        print(f"    WARNING -- sequence not found in discovered list.")
    else:
        print(f"    Valid pairs     : {len(target.pairs)}  (expected 21)")
        print(f"    Unmatched masks : {len(target.unmatched_masks)}  (expected 14)")
        assert len(target.pairs) == 21, (
            f"Expected 21 valid pairs, got {len(target.pairs)}"
        )
        assert len(target.unmatched_masks) == 14, (
            f"Expected 14 unmatched masks, got {len(target.unmatched_masks)}"
        )
        print(f"    PASS -- stem-based pairing correct.")

    # ------------------------------------------------------------------
    # 3. Sequence-level train / val split
    # ------------------------------------------------------------------
    train_seqs, val_seqs = sequence_level_split(
        sequences, config.VAL_FRACTION, config.SEED
    )

    train_names = {s.name for s in train_seqs}
    val_names   = {s.name for s in val_seqs}
    overlap     = train_names & val_names

    n_train_samples = sum(len(s.pairs) for s in train_seqs)
    n_val_samples   = sum(len(s.pairs) for s in val_seqs)

    print(f"\n  VAL_FRACTION            : {config.VAL_FRACTION}  (SEED={config.SEED})")
    print(f"  Training sequences      : {len(train_seqs)}")
    print(f"  Validation sequences    : {len(val_seqs)}")
    print(f"  Training samples (raw)  : {n_train_samples}")
    print(f"  Validation samples      : {n_val_samples}")
    print(f"  Sequence overlap count  : {len(overlap)}  (must be 0)")

    assert len(overlap) == 0, (
        f"[FAIL] Sequences in BOTH splits: {overlap}"
    )
    print(f"  PASS -- no sequence appears in both splits.")

    # ------------------------------------------------------------------
    # 4. Empty mask filter check
    # ------------------------------------------------------------------
    train_flat = [
        SampleInfo(img, mask, seq.name)
        for seq in train_seqs
        for img, mask in seq.pairs
    ]
    n_before = len(train_flat)
    train_filtered = [s for s in train_flat if _has_sufficient_foreground(s.mask_path)]
    n_removed = n_before - len(train_filtered)
    print(f"\n  Empty mask filter (fg < {config.MIN_FOREGROUND_FRAC * 100:.1f}%):")
    print(f"    Before : {n_before}")
    print(f"    After  : {len(train_filtered)}")
    print(f"    Removed: {n_removed}  (annotation errors)")

    # ------------------------------------------------------------------
    # 5. DataLoader smoke-test (one batch)
    # ------------------------------------------------------------------
    print(f"\n  Building DataLoaders ...")
    train_loader, val_loader = get_loaders()
    print(f"  Train batches           : {len(train_loader)}")
    print(f"  Val   batches           : {len(val_loader)}")

    imgs, masks = next(iter(train_loader))
    print(f"\n  Batch image shape       : {tuple(imgs.shape)}   dtype={imgs.dtype}")
    print(f"  Batch mask  shape       : {tuple(masks.shape)}  dtype={masks.dtype}")
    print(f"  Mask unique values      : {sorted(masks.unique().tolist())}")

    assert masks.shape[1] == 1, "Mask should have 1 channel"
    assert set(masks.unique().tolist()).issubset({0.0, 1.0}), (
        "Mask should be binary {0, 1}"
    )

    print(f"\n  PASS -- all assertions passed.")
    print(f"\n{SEP}")
    print("  Dataset self-test  COMPLETE")
    print(f"{SEP}\n")
