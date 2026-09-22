"""
scripts/inspect_dataset.py
===========================
MoCA-Video Dataset Inspection (per-sequence layout).

Walks ``dataset/MoCA_Video/TrainDataset_per_sq/`` and reports:
  1. Number of sequence folders found
  2. Number of valid sequences (have both Imgs/ and GT/)
  3. Total image-mask pairs (stem-matched)
  4. Total unmatched images (image with no matching mask)
  5. Total unmatched masks  (mask with no matching image)
  6. Per-sequence breakdown (flag anomalies)
  7. Sample inspection: image/mask sizes, mask pixel values

Run AFTER downloading MoCA-Video to dataset/MoCA_Video/TrainDataset_per_sq/.

Usage::

    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --root "dataset/MoCA_Video/TrainDataset_per_sq"
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from utils.dataset import discover_sequences, sequence_level_split


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _bar(char: str = "-", width: int = 62) -> str:
    return char * width


def print_sequence_table(sequences, max_anomalies: int = 10) -> None:
    """Print a per-sequence breakdown, flagging mismatched counts."""
    anomalies: List[str] = []
    print(f"\n  {'Sequence':<40}  {'Pairs':>6}  {'!Imgs':>5}  {'!Masks':>6}")
    print(f"  {_bar('-', 62)}")
    for s in sequences:
        flag = ""
        if s.unmatched_imgs or s.unmatched_masks:
            flag = "  <-- MISMATCH"
            anomalies.append(
                f"    {s.name}: "
                f"{len(s.pairs)} pairs, "
                f"{len(s.unmatched_imgs)} unmatched imgs, "
                f"{len(s.unmatched_masks)} unmatched masks"
            )
        print(
            f"  {s.name:<40}  {len(s.pairs):>6}"
            f"  {len(s.unmatched_imgs):>5}  {len(s.unmatched_masks):>6}{flag}"
        )

    if anomalies:
        print(f"\n  Anomalies ({len(anomalies)}):")
        for line in anomalies[:max_anomalies]:
            print(line)
        if len(anomalies) > max_anomalies:
            print(f"    ... and {len(anomalies) - max_anomalies} more")


def inspect_samples(sequences, n_samples: int = 3) -> None:
    """Open and inspect the first *n_samples* valid pairs."""
    print(f"\n  Sample pairs (first {n_samples}):")
    count = 0
    for seq in sequences:
        for img_path, mask_path in seq.pairs:
            if count >= n_samples:
                return
            try:
                img_pil  = Image.open(img_path).convert("RGB")
                mask_pil = Image.open(mask_path).convert("L")
                mask_arr = np.array(mask_pil)
                unique   = sorted(np.unique(mask_arr).tolist())
                is_bin   = set(unique).issubset({0, 255})
                print(
                    f"    [{seq.name}] {img_path.stem}"
                    f"  img={img_pil.size}px ({img_pil.mode})"
                    f"  mask={mask_pil.size}px  unique={unique}"
                    f"  binary={'OK' if is_bin else 'WARN'}"
                )
            except Exception as exc:
                print(f"    [ERROR] {img_path}: {exc}")
            count += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inspect the MoCA-Video per-sequence dataset directory."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=config.MOCA_TRAIN_DIR,
        help=f"Path to TrainDataset_per_sq (default: {config.MOCA_TRAIN_DIR})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample pairs to inspect in detail (default: 3)",
    )
    args = parser.parse_args()

    train_dir = Path(args.root)

    print(f"\n{_bar('=')}")
    print("  MoCA-Video Dataset Inspection  (per-sequence layout)")
    print(_bar("="))
    print(f"  Root   : {train_dir}")
    print(f"  Exists : {train_dir.exists()}")

    if not train_dir.exists():
        print(
            f"\n  [ERROR] Directory not found: {train_dir}\n\n"
            "  Expected structure:\n"
            "    dataset/MoCA_Video/TrainDataset_per_sq/<sequence>/{Imgs,GT}/\n"
        )
        return

    # ------------------------------------------------------------------
    # Discover sequences (stem-based pairing)
    # ------------------------------------------------------------------
    sequences, n_folders = discover_sequences(train_dir)

    total_pairs           = sum(len(s.pairs)           for s in sequences)
    total_unmatched_imgs  = sum(len(s.unmatched_imgs)  for s in sequences)
    total_unmatched_masks = sum(len(s.unmatched_masks) for s in sequences)

    print(f"\n  {_bar()}")
    print(f"  Summary")
    print(f"  {_bar()}")
    print(f"  Sequence folders found  : {n_folders}")
    print(f"  Valid sequences         : {len(sequences)}")
    print(f"  Total image-mask pairs  : {total_pairs}")
    print(f"  Total unmatched images  : {total_unmatched_imgs}")
    print(f"  Total unmatched masks   : {total_unmatched_masks}")

    # ------------------------------------------------------------------
    # Per-sequence table
    # ------------------------------------------------------------------
    print(f"\n  {_bar()}")
    print(f"  Per-sequence breakdown  (!Imgs = unmatched images, !Masks = unmatched masks)")
    print(f"  {_bar()}")
    print_sequence_table(sequences)

    # ------------------------------------------------------------------
    # Train / val split summary
    # ------------------------------------------------------------------
    train_seqs, val_seqs = sequence_level_split(
        sequences, config.VAL_FRACTION, config.SEED
    )
    n_train_samples = sum(len(s.pairs) for s in train_seqs)
    n_val_samples   = sum(len(s.pairs) for s in val_seqs)
    overlap         = {s.name for s in train_seqs} & {s.name for s in val_seqs}

    print(f"\n  {_bar()}")
    print(f"  Train / Validation split  "
          f"(VAL_FRACTION={config.VAL_FRACTION}, SEED={config.SEED})")
    print(f"  {_bar()}")
    print(f"  Training sequences      : {len(train_seqs)}")
    print(f"  Validation sequences    : {len(val_seqs)}")
    print(f"  Training samples        : {n_train_samples}")
    print(f"  Validation samples      : {n_val_samples}")
    print(f"  Sequence overlap        : {len(overlap)}  ({'OK -- no leakage' if not overlap else 'FAIL -- LEAKAGE DETECTED'})")

    # ------------------------------------------------------------------
    # Sample inspection
    # ------------------------------------------------------------------
    print(f"\n  {_bar()}")
    print(f"  Sample Inspection")
    print(f"  {_bar()}")
    inspect_samples(sequences, n_samples=args.samples)

    print(f"\n{_bar('=')}")
    print("  Inspection complete.")
    print("  Next step:")
    print("    python utils/dataset.py   (full self-test with assertions)")
    print(f"{_bar('=')}\n")


if __name__ == "__main__":
    main()
