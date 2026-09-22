"""
utils/metrics.py
=================
Evaluation metrics for binary segmentation.

IMPORTANT — input convention (after fix):
  pred   : (B, 1, H, W) — RAW LOGITS from U-Net.
           torch.sigmoid() is applied internally before thresholding.
  target : (B, 1, H, W) — Binary ground-truth mask, values in {0, 1}.

Metrics computed:
  - IoU (Intersection over Union) — also called Jaccard Index
  - Dice Coefficient (F1 score for segmentation)
  - Precision
  - Recall (Sensitivity)
  - Pixel Accuracy

All metrics operate on thresholded binary predictions.
Threshold is set by config.SEG_THRESHOLD (default 0.5).

Usage:
    from utils.metrics import compute_metrics, SegmentationMetrics
    m = compute_metrics(pred_batch, target_batch)   # pred_batch = raw logits
    print(m.iou, m.dice)
"""

import sys
import os
from dataclasses import dataclass
from typing import Optional

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SegmentationMetrics:
    """Container for one batch of segmentation metrics."""
    iou:       float
    dice:      float
    precision: float
    recall:    float
    accuracy:  float

    def __str__(self) -> str:
        return (
            f"IoU={self.iou:.4f}  Dice={self.dice:.4f}  "
            f"Prec={self.precision:.4f}  Rec={self.recall:.4f}  "
            f"Acc={self.accuracy:.4f}"
        )


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    threshold: float = config.SEG_THRESHOLD,
    smooth:    float = 1e-6,
) -> SegmentationMetrics:
    """
    Compute segmentation metrics for a batch of predictions.

    Parameters
    ----------
    pred : torch.Tensor, shape (B, 1, H, W)
        RAW LOGITS from U-Net (after fix). torch.sigmoid() is applied
        internally to convert to probabilities before thresholding.
    target : torch.Tensor, shape (B, 1, H, W)
        Binary ground-truth mask. Values in {0, 1}.
    threshold : float
        Probability threshold for converting soft predictions to binary.
        Applied to sigmoid(pred), not raw logits.
    smooth : float
        Small constant to prevent division by zero in precision/recall/IoU.

    Returns
    -------
    SegmentationMetrics
        Averaged over the batch.
    """
    # Convert raw logits to probabilities — applied exactly once here.
    # Do NOT pre-apply sigmoid before calling this function.
    pred_prob = torch.sigmoid(pred)

    # Threshold predictions to binary
    pred_bin = (pred_prob >= threshold).float()
    target   = target.float()

    # Flatten to (B, N)
    pred_flat   = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Element-wise statistics per sample in batch
    tp = (pred_flat * target_flat).sum(dim=1)                            # True Positives
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)                      # False Positives
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)                      # False Negatives
    tn = ((1 - pred_flat) * (1 - target_flat)).sum(dim=1)                # True Negatives

    iou       = (tp + smooth) / (tp + fp + fn + smooth)
    dice      = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    accuracy  = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)

    return SegmentationMetrics(
        iou       = iou.mean().item(),
        dice      = dice.mean().item(),
        precision = precision.mean().item(),
        recall    = recall.mean().item(),
        accuracy  = accuracy.mean().item(),
    )


# ---------------------------------------------------------------------------
# Running averager (used during training epoch loops)
# ---------------------------------------------------------------------------

class MetricAccumulator:
    """
    Accumulates metrics over multiple batches and returns epoch averages.

    Usage:
        acc = MetricAccumulator()
        for batch in loader:
            m = compute_metrics(pred, target)
            acc.update(m)
        epoch_avg = acc.average()
    """

    def __init__(self) -> None:
        self._totals = dict(iou=0.0, dice=0.0, precision=0.0,
                            recall=0.0, accuracy=0.0)
        self._count  = 0

    def update(self, m: SegmentationMetrics) -> None:
        self._totals["iou"]       += m.iou
        self._totals["dice"]      += m.dice
        self._totals["precision"] += m.precision
        self._totals["recall"]    += m.recall
        self._totals["accuracy"]  += m.accuracy
        self._count += 1

    def average(self) -> SegmentationMetrics:
        if self._count == 0:
            return SegmentationMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        n = self._count
        return SegmentationMetrics(
            iou       = self._totals["iou"]       / n,
            dice      = self._totals["dice"]      / n,
            precision = self._totals["precision"] / n,
            recall    = self._totals["recall"]    / n,
            accuracy  = self._totals["accuracy"]  / n,
        )

    def reset(self) -> None:
        for k in self._totals:
            self._totals[k] = 0.0
        self._count = 0


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Metrics self-test ===")
    B, H, W = 4, 64, 64

    # Perfect prediction — use logits that sigmoid-map to the correct binary values.
    # logit(1.0) → large positive; logit(0.0) → large negative
    target = (torch.rand(B, 1, H, W) > 0.5).float()
    # Construct logits that, after sigmoid, reproduce the target mask
    perfect_logits = target * 20.0 - (1 - target) * 20.0   # +20 for fg, -20 for bg
    m = compute_metrics(perfect_logits, target)
    print(f"  Perfect logits: {m}")
    assert abs(m.iou - 1.0) < 1e-3,  "IoU should be ~1.0 for perfect pred"
    assert abs(m.dice - 1.0) < 1e-3, "Dice should be ~1.0 for perfect pred"

    # Random logits prediction
    pred_logits = torch.randn(B, 1, H, W)   # raw logits
    m = compute_metrics(pred_logits, target)
    print(f"  Random logits: {m}")
    assert 0.0 <= m.iou  <= 1.0, "IoU out of [0,1]"
    assert 0.0 <= m.dice <= 1.0, "Dice out of [0,1]"

    # Accumulator
    acc = MetricAccumulator()
    for _ in range(5):
        acc.update(compute_metrics(torch.randn(B, 1, H, W), target))
    avg = acc.average()
    print(f"  Avg (5 batches): {avg}")

    print("PASS — metrics OK.")
