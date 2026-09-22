"""
scripts/diagnose_masks.py
==========================
Diagnostic script: audit all training masks in MoCA-Mask for common issues.

WHAT THIS SCRIPT CHECKS
------------------------
For every mask in the training split it measures:
  1. Foreground fraction (fraction of pixels > 0)
  2. Whether the mask is completely empty (fg_fraction = 0.0)
  3. Whether the mask has suspiciously HIGH foreground (> 50% -- may be inverted)
  4. Per-sequence statistics (mean fg, min fg, max fg, empty count)
  5. Overall dataset statistics

The dataset.py loader already filters out masks with fg < MIN_FOREGROUND_FRAC
at training time. This script lets you see WHAT is being filtered and whether
the filter threshold is appropriate.

OUTPUT
------
  outputs/member1_eval/mask_audit.csv    : one row per mask
  outputs/member1_eval/sequence_stats.csv : one row per sequence
  stdout                                 : summary statistics

USAGE
-----
    python scripts/diagnose_masks.py

    # Or with explicit dataset root:
    python scripts/diagnose_masks.py --data-root /path/to/MoCA-VideoSegment

    # To also check the val split:
    python scripts/diagnose_masks.py --split both
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_fg_fraction(mask_path: str) -> float:
    """Return fraction of pixels that are foreground (> 0) in a mask."""
    try:
        img = Image.open(mask_path).convert("L")
        arr = np.array(img, dtype=np.float32)
        return float((arr > 0).sum()) / max(arr.size, 1)
    except Exception as e:
        return -1.0   # signal load error


def scan_sequence(seq_dir: Path) -> list:
    """
    Scan one sequence directory for mask files.
    Returns a list of dicts: {seq, mask_path, fg_fraction}.
    """
    results = []
    mask_exts = {".png", ".jpg", ".jpeg", ".bmp"}

    # Common MoCA-Mask layout: <seq>/SegmentationClassPNG/ or <seq>/masks/
    for sub in ("SegmentationClassPNG", "masks", "gt", "GT"):
        mask_subdir = seq_dir / sub
        if mask_subdir.is_dir():
            mask_files = sorted(
                p for p in mask_subdir.iterdir()
                if p.suffix.lower() in mask_exts
            )
            for mp in mask_files:
                fg = get_fg_fraction(str(mp))
                results.append({
                    "sequence": seq_dir.name,
                    "mask_path": str(mp),
                    "fg_fraction": fg,
                    "is_empty":    fg == 0.0,
                    "is_inverted": fg > 0.5,
                    "load_error":  fg < 0.0,
                })
            return results

    # Fallback: mask files directly in seq_dir (no subdirectory)
    mask_files = sorted(
        p for p in seq_dir.iterdir()
        if p.is_file() and p.suffix.lower() in mask_exts
    )
    for mp in mask_files:
        fg = get_fg_fraction(str(mp))
        results.append({
            "sequence": seq_dir.name,
            "mask_path": str(mp),
            "fg_fraction": fg,
            "is_empty":    fg == 0.0,
            "is_inverted": fg > 0.5,
            "load_error":  fg < 0.0,
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Audit MoCA-Mask mask files.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=config.MOCA_TRAIN_DIR,
        help=f"Root directory of MoCA-VideoSegment (default: {config.MOCA_TRAIN_DIR})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "both"],
        help="Which split to audit (default: train)",
    )
    parser.add_argument(
        "--fg-threshold",
        type=float,
        default=config.MIN_FOREGROUND_FRAC,
        help=(
            f"Foreground fraction threshold (default: {config.MIN_FOREGROUND_FRAC}). "
            "Masks with fg_fraction < this are flagged as 'filtered'."
        ),
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[ERROR] Dataset root not found: {data_root}")
        print("        Set --data-root or config.MOCA_TRAIN_DIR to the correct path.")
        sys.exit(1)

    out_dir = Path(config.OUTPUTS_DIR) / "member1_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Discover sequences ----
    # MoCA-VideoSegment layout: <root>/<split>/<sequence_name>/
    splits_to_scan = []
    if args.split in ("train", "both"):
        train_dir = data_root
        if (data_root / "train").is_dir():
            train_dir = data_root / "train"
        splits_to_scan.append(("train", train_dir))
    if args.split in ("val", "both"):
        val_dir = data_root
        if (data_root / "val").is_dir():
            val_dir = data_root / "val"
        elif (data_root / "test").is_dir():
            val_dir = data_root / "test"
        splits_to_scan.append(("val", val_dir))

    all_records = []
    seq_dirs_found = 0

    for split_name, split_root in splits_to_scan:
        if not split_root.is_dir():
            print(f"[WARN] Split directory not found: {split_root}")
            continue
        seq_dirs = sorted(
            d for d in split_root.iterdir() if d.is_dir()
        )
        print(f"\n  {split_name} split: {len(seq_dirs)} sequences found in {split_root}")
        seq_dirs_found += len(seq_dirs)

        for seq_dir in tqdm(seq_dirs, desc=f"  Scanning {split_name}", unit="seq"):
            records = scan_sequence(seq_dir)
            for r in records:
                r["split"] = split_name
            all_records.extend(records)

    if not all_records:
        print("\n[ERROR] No mask files found. Check --data-root path.")
        sys.exit(1)

    # ---- Statistics ----
    fg_fracs = [r["fg_fraction"] for r in all_records if not r["load_error"]]

    n_total    = len(all_records)
    n_errors   = sum(r["load_error"]   for r in all_records)
    n_empty    = sum(r["is_empty"]     for r in all_records)
    n_inverted = sum(r["is_inverted"]  for r in all_records)
    n_filtered = sum(r["fg_fraction"] < args.fg_threshold
                     and not r["load_error"] for r in all_records)

    print(f"\n{'='*60}")
    print(f"  MASK AUDIT RESULTS")
    print(f"{'='*60}")
    print(f"  Sequences scanned  : {seq_dirs_found}")
    print(f"  Total masks        : {n_total:,}")
    print(f"  Load errors        : {n_errors}")
    print(f"  Empty masks (fg=0) : {n_empty}  ({100*n_empty/max(n_total,1):.1f}%)")
    print(f"  Filtered (fg<{args.fg_threshold:.3f}): {n_filtered}  ({100*n_filtered/max(n_total,1):.1f}%)")
    print(f"  Inverted? (fg>0.5) : {n_inverted}  ({100*n_inverted/max(n_total,1):.1f}%)")
    if fg_fracs:
        print(f"  Foreground fraction statistics:")
        print(f"    Mean   : {np.mean(fg_fracs):.4f}")
        print(f"    Median : {np.median(fg_fracs):.4f}")
        print(f"    Std    : {np.std(fg_fracs):.4f}")
        print(f"    Min    : {np.min(fg_fracs):.4f}")
        print(f"    Max    : {np.max(fg_fracs):.4f}")
        print(f"    90th%  : {np.percentile(fg_fracs, 90):.4f}")
        print(f"    99th%  : {np.percentile(fg_fracs, 99):.4f}")
        implied_pos_weight = (1.0 - np.mean(fg_fracs)) / max(np.mean(fg_fracs), 1e-6)
        print(f"  Implied BCE pos_weight (bg/fg ratio): {implied_pos_weight:.1f}")
        print(f"    NOTE: pos_weight is now DISABLED (set to None) -- Tversky handles imbalance.")
    print(f"{'='*60}")

    # ---- Per-sequence stats ----
    seq_stats = defaultdict(lambda: {"n": 0, "fg_sum": 0.0,
                                     "fg_min": 1.0, "fg_max": 0.0,
                                     "n_empty": 0, "n_inverted": 0})
    for r in all_records:
        if r["load_error"]:
            continue
        s = seq_stats[r["sequence"]]
        s["n"] += 1
        s["fg_sum"] += r["fg_fraction"]
        s["fg_min"]  = min(s["fg_min"], r["fg_fraction"])
        s["fg_max"]  = max(s["fg_max"], r["fg_fraction"])
        if r["is_empty"]:
            s["n_empty"] += 1
        if r["is_inverted"]:
            s["n_inverted"] += 1

    # Sort by mean fg fraction descending (most suspicious first)
    seq_sorted = sorted(
        seq_stats.items(),
        key=lambda x: x[1]["fg_sum"] / max(x[1]["n"], 1),
        reverse=True,
    )

    print(f"\n  Top 10 sequences by mean foreground fraction (potential mislabelling):")
    print(f"  {'Sequence':<30}  {'N':>5}  {'Mean FG':>8}  {'Max FG':>8}  {'Inverted':>8}")
    print(f"  {'-'*70}")
    for seq_name, s in seq_sorted[:10]:
        mean_fg = s["fg_sum"] / max(s["n"], 1)
        print(
            f"  {seq_name:<30}  {s['n']:>5}  {mean_fg:>8.4f}  "
            f"{s['fg_max']:>8.4f}  {s['n_inverted']:>8}"
        )

    print(f"\n  Bottom 10 sequences by mean foreground fraction (hardest / smallest targets):")
    print(f"  {'Sequence':<30}  {'N':>5}  {'Mean FG':>8}  {'Empty':>6}")
    print(f"  {'-'*60}")
    for seq_name, s in seq_sorted[-10:]:
        mean_fg = s["fg_sum"] / max(s["n"], 1)
        print(
            f"  {seq_name:<30}  {s['n']:>5}  {mean_fg:>8.4f}  {s['n_empty']:>6}"
        )

    # ---- Save CSV reports ----
    # Per-mask CSV
    mask_csv = out_dir / "mask_audit.csv"
    with open(mask_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "sequence", "mask_path", "fg_fraction",
                        "is_empty", "is_inverted", "load_error"],
        )
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n  Per-mask CSV  -> {mask_csv}")

    # Per-sequence CSV
    seq_csv = out_dir / "sequence_stats.csv"
    with open(seq_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "n_masks", "mean_fg", "min_fg", "max_fg",
                         "n_empty", "n_inverted"])
        for seq_name, s in sorted(seq_stats.items()):
            mean_fg = s["fg_sum"] / max(s["n"], 1)
            writer.writerow([
                seq_name, s["n"], f"{mean_fg:.6f}", f"{s['fg_min']:.6f}",
                f"{s['fg_max']:.6f}", s["n_empty"], s["n_inverted"],
            ])
    print(f"  Per-sequence CSV -> {seq_csv}")
    print(f"\n  Run complete.\n")


if __name__ == "__main__":
    main()
