"""
utils/losses.py
================
Loss functions for U-Net training.

IMPORTANT — input convention:
  pred   : (B, 1, H, W) — RAW LOGITS from U-Net (no sigmoid applied)
  target : (B, 1, H, W) — Binary ground-truth mask, values in {0, 1}

Implemented losses:
  - BCELoss       : Wraps nn.BCEWithLogitsLoss — numerically stable,
                    applies sigmoid internally via log-sum-exp trick.
  - DiceLoss      : Overlap-based loss, robust to class imbalance.
                    Applies torch.sigmoid(pred) internally.
  - BCEDiceLoss   : Weighted combination of BCE + Dice
  - TverskyLoss   : Tversky Index loss (RECOMMENDED — replaces BCEDice).
                    TI = TP / (TP + alpha*FP + beta*FN)
                    Loss = 1 - TI
                    alpha=0.3, beta=0.7 (from config) gives ~2.3x FN penalty
                    over FP, correcting class imbalance without the instability
                    caused by combining BCE(pos_weight=11.7) with Dice.
                    Reference: Salehi et al., "Tversky Loss Function for Image
                    Segmentation Using 3D Fully Convolutional Deep Networks",
                    MLMI 2017.

All parameters come from config.py:
  LOSS_TYPE       : "bce" | "dice" | "bce_dice" | "tversky"
  BCE_WEIGHT      : float (default 0.4)
  DICE_WEIGHT     : float (default 0.6)
  DICE_SMOOTH     : float (default 1.0)
  BCE_POS_WEIGHT  : float | None  -- set to None after diagnosis
  TVERSKY_ALPHA   : float (default 0.3) -- FP weight
  TVERSKY_BETA    : float (default 0.7) -- FN weight
  TVERSKY_SMOOTH  : float (default 1.0)

Usage:
  from utils.losses import get_loss_fn
  criterion = get_loss_fn()
  loss = criterion(pred, target)   # pred: (B,1,H,W) raw logits; target: binary {0,1}
"""

import torch
import torch.nn as nn
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


# ---------------------------------------------------------------------------
# Binary Cross-Entropy Loss  (with optional imbalance correction)
# ---------------------------------------------------------------------------

class BCELoss(nn.Module):
    """
    Numerically stable Binary Cross-Entropy loss with class-imbalance correction.

    Uses nn.BCEWithLogitsLoss which fuses sigmoid + BCE via the
    log-sum-exp trick:  max(x,0) - x*y + log(1 + exp(-|x|))
    This avoids NaN/inf gradients that occur when sigmoid is applied
    before nn.BCELoss.

    pos_weight (from config.BCE_POS_WEIGHT):
        Scalar weight applied to ALL foreground (positive) pixels.
        NOTE: Set to None in config after diagnosis -- Tversky handles this better.

    Expects:
      pred   : (B, 1, H, W) -- RAW LOGITS from U-Net (no sigmoid pre-applied)
      target : (B, 1, H, W) -- Binary ground-truth mask, values in {0, 1}
    """
    def __init__(self, pos_weight_val: float = config.BCE_POS_WEIGHT) -> None:
        super().__init__()
        if pos_weight_val is not None and pos_weight_val > 0:
            # pos_weight must be a 1-element tensor; it is moved to the correct
            # device lazily in forward() so we don't need to know the device here.
            self._pos_weight_val = float(pos_weight_val)
        else:
            self._pos_weight_val = None
        self._bce = None   # built lazily on first forward call

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Build BCEWithLogitsLoss on the correct device (lazy init)
        if self._bce is None:
            if self._pos_weight_val is not None:
                pw = torch.tensor([self._pos_weight_val], device=pred.device)
                self._bce = nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                self._bce = nn.BCEWithLogitsLoss()
        # Move pos_weight to the same device as pred if device changed
        elif (
            self._pos_weight_val is not None
            and self._bce.pos_weight is not None
            and self._bce.pos_weight.device != pred.device
        ):
            self._bce.pos_weight = self._bce.pos_weight.to(pred.device)

        return self._bce(pred, target.float())


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Dice = 1 - (2 * |pred n target| + smooth) / (|pred| + |target| + smooth)

    Dice Loss handles class imbalance well because it measures the ratio of
    overlap to total area.

    Expects:
      pred   : (B, 1, H, W) -- RAW LOGITS from U-Net (sigmoid applied internally)
      target : (B, 1, H, W) -- Binary mask, values in {0, 1}
    """
    def __init__(self, smooth: float = config.DICE_SMOOTH) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()

        # Apply sigmoid to convert raw logits to probabilities [0, 1].
        # This is applied exactly once here; do NOT pre-apply sigmoid before calling.
        pred = torch.sigmoid(pred)

        # Flatten spatial dimensions to (B, N)
        pred_flat   = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)              # (B,)
        union        = pred_flat.sum(dim=1) + target_flat.sum(dim=1)     # (B,)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score.mean()


# ---------------------------------------------------------------------------
# Combined BCE + Dice Loss
# ---------------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    """
    Weighted combination of BCELoss and DiceLoss.

    loss = bce_weight * BCE(pred, target) + dice_weight * Dice(pred, target)

    NOTE: With BCE_POS_WEIGHT=None (post-diagnosis), prefer TverskyLoss instead.
    """
    def __init__(
        self,
        bce_weight:      float = config.BCE_WEIGHT,
        dice_weight:     float = config.DICE_WEIGHT,
        smooth:          float = config.DICE_SMOOTH,
        pos_weight_val:  float = config.BCE_POS_WEIGHT,
    ) -> None:
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self._bce  = BCELoss(pos_weight_val=pos_weight_val)
        self._dice = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss  = self._bce(pred, target)
        dice_loss = self._dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# Tversky Loss  (DIAGNOSIS FIX -- RC-2, replaces BCEDice as default)
# ---------------------------------------------------------------------------

class TverskyLoss(nn.Module):
    """
    Tversky Loss for binary segmentation with controllable FP/FN penalty.

    Tversky Index  TI = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    Tversky Loss      = 1 - TI

    When alpha = beta = 0.5  ->  equivalent to Dice Loss.
    When alpha < beta         ->  heavier penalty on FN (missed animals).
    When alpha > beta         ->  heavier penalty on FP (background activation).

    Default (alpha=0.3, beta=0.7):
      - FN penalised 0.7 / 0.3 ~= 2.3x more than FP.
      - Corrects for class imbalance without the unstable BCE pos_weight.
      - The model is allowed to slightly over-predict rather than completely
        missing the animal, but not as aggressively as BCE(pos_weight=11.7).

    DIAGNOSIS (RC-2): This replaces the BCE(pos_weight=11.7) + Dice combination
    that caused seg_mean ~= 0.631 (63% of pixels predicted as foreground).
    Tversky directly expresses the FP/FN trade-off via interpretable alpha/beta
    parameters, avoiding the gradient conflict between a high BCE pos_weight
    and Dice overlap maximisation.

    Reference:
      Salehi et al., "Tversky Loss Function for Image Segmentation Using 3D
      Fully Convolutional Deep Networks", MLMI 2017.

    Expects:
      pred   : (B, 1, H, W) -- RAW LOGITS from U-Net (sigmoid applied internally)
      target : (B, 1, H, W) -- Binary mask, values in {0, 1}
    """

    def __init__(
        self,
        alpha:  float = config.TVERSKY_ALPHA,
        beta:   float = config.TVERSKY_BETA,
        smooth: float = config.TVERSKY_SMOOTH,
    ) -> None:
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()

        # Apply sigmoid exactly once to convert raw logits -> probabilities [0,1].
        # Do NOT pre-apply sigmoid before calling this function.
        pred_prob = torch.sigmoid(pred)

        # Flatten spatial dimensions: (B, 1, H, W) -> (B, N)
        pred_flat   = pred_prob.view(pred_prob.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Per-sample confusion matrix components (soft / probabilistic)
        tp = (pred_flat * target_flat).sum(dim=1)                     # True Positives
        fp = (pred_flat * (1.0 - target_flat)).sum(dim=1)             # False Positives
        fn = ((1.0 - pred_flat) * target_flat).sum(dim=1)             # False Negatives

        # Tversky index per sample, then average over batch
        tversky_idx = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky_idx.mean()


# ---------------------------------------------------------------------------
# Size-Aware Tversky Loss (Phase 2)
# ---------------------------------------------------------------------------

class SizeAwareTverskyLoss(nn.Module):
    """
    Tversky Loss with dynamic boundary and size weighting.
    Provides extra penalty for missing small objects.
    """
    def __init__(
        self,
        alpha:  float = config.TVERSKY_ALPHA,
        beta:   float = config.TVERSKY_BETA,
        smooth: float = config.TVERSKY_SMOOTH,
        boundary_weight: float = getattr(config, 'BOUNDARY_LOSS_WEIGHT', 0.1),
    ) -> None:
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth
        self.boundary_weight = boundary_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        pred_prob = torch.sigmoid(pred)
        
        # Calculate object size (fraction of image area)
        batch_size = target.size(0)
        target_flat = target.view(batch_size, -1)
        pred_flat = pred_prob.view(batch_size, -1)
        
        area = target_flat.sum(dim=1)
        total_pixels = target_flat.size(1)
        size_fraction = area / total_pixels
        
        # Boost beta (FN penalty) for very small objects (< 2% of frame)
        size_boost = torch.ones(batch_size, device=target.device)
        small_mask = (size_fraction > 0) & (size_fraction < 0.02)
        # linear boost: up to 3x for smallest
        size_boost[small_mask] = 1.0 + 2.0 * (0.02 - size_fraction[small_mask]) / 0.02
        
        # Apply Tversky
        tp = (pred_flat * target_flat).sum(dim=1)
        fp = (pred_flat * (1.0 - target_flat)).sum(dim=1)
        fn = ((1.0 - pred_flat) * target_flat).sum(dim=1)
        
        # Apply boosted beta
        boosted_beta = self.beta * size_boost
        
        tversky_idx = (tp + self.smooth) / (
            tp + self.alpha * fp + boosted_beta * fn + self.smooth
        )
        
        tversky_loss = 1.0 - tversky_idx.mean()
        
        # Boundary loss
        if self.boundary_weight > 0:
            target_max = nn.functional.max_pool2d(target, kernel_size=3, stride=1, padding=1)
            target_min = -nn.functional.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
            target_boundary = target_max - target_min
            
            boundary_bce = nn.functional.binary_cross_entropy(pred_prob, target, weight=target_boundary, reduction='mean')
            tversky_loss = tversky_loss + self.boundary_weight * boundary_bce
            
        return tversky_loss


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_loss_fn() -> nn.Module:
    """
    Return the loss function specified in config.LOSS_TYPE.

    Returns:
      nn.Module -- one of BCELoss, DiceLoss, BCEDiceLoss, TverskyLoss
    """
    loss_type = config.LOSS_TYPE.lower().strip()

    if loss_type == "bce":
        return BCELoss()
    elif loss_type == "dice":
        return DiceLoss()
    elif loss_type == "bce_dice":
        return BCEDiceLoss()
    elif loss_type == "tversky":
        # DIAGNOSIS FIX (RC-2): Default loss after diagnosis.
        # Phase 2: Use SizeAwareTverskyLoss if LOSS_SIZE_AWARE is True
        if getattr(config, 'LOSS_SIZE_AWARE', False):
            return SizeAwareTverskyLoss()
        return TverskyLoss()
    else:
        raise ValueError(
            f"Unknown LOSS_TYPE='{config.LOSS_TYPE}' in config.py. "
            "Valid options: 'bce', 'dice', 'bce_dice', 'tversky'."
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    print("=== Loss functions self-test ===")
    B, H, W = 4, 64, 64

    pred   = torch.randn(B, 1, H, W)                   # raw logits
    target = (torch.rand(B, 1, H, W) > 0.5).float()   # random binary mask

    for name, fn in [("BCELoss",     BCELoss()),
                     ("DiceLoss",    DiceLoss()),
                     ("BCEDiceLoss", BCEDiceLoss()),
                     ("TverskyLoss", TverskyLoss()),
                     ("SizeAwareTverskyLoss", SizeAwareTverskyLoss())]:
        loss_val = fn(pred, target)
        print(f"  {name:20s}  loss = {loss_val.item():.6f}")
        assert loss_val.item() >= 0, f"{name} returned negative loss"

    # Tversky: verify that with beta>alpha, missing FG is penalised more
    # than over-predicting FG.
    all_fg_logits = torch.full((B, 1, H, W), 5.0)    # predict everything positive
    all_bg_logits = torch.full((B, 1, H, W), -5.0)   # predict everything negative
    sparse_target = (torch.rand(B, 1, H, W) > 0.9).float()  # 10% foreground

    tv = TverskyLoss(alpha=0.3, beta=0.7)
    loss_all_fg = tv(all_fg_logits, sparse_target).item()   # high FP, low FN
    loss_all_bg = tv(all_bg_logits, sparse_target).item()   # low FP, high FN
    print(f"\n  Tversky FP/FN penalty check (alpha=0.3, beta=0.7, sparse_target):")
    print(f"    All-FG prediction (high FP, low FN)  : {loss_all_fg:.4f}")
    print(f"    All-BG prediction (low FP, high FN)  : {loss_all_bg:.4f}")
    assert loss_all_bg > loss_all_fg, (
        f"With beta>alpha, missing FG (all-BG loss={loss_all_bg:.4f}) should "
        f"be penalised MORE than over-predicting (all-FG loss={loss_all_fg:.4f})"
    )
    print(f"    PASS -- FN penalised more than FP as expected.")

    # BCE pos_weight=None check (no crash)
    bce_no_pw  = BCELoss(pos_weight_val=None)
    loss_no_pw = bce_no_pw(pred, target).item()
    print(f"\n  BCE with pos_weight=None : {loss_no_pw:.4f}  (no crash)")

    # Factory test
    criterion = get_loss_fn()
    loss_val  = criterion(pred, target)
    print(f"\n  get_loss_fn() [{config.LOSS_TYPE}] = {loss_val.item():.6f}")
    print(f"  Tversky alpha={config.TVERSKY_ALPHA}, beta={config.TVERSKY_BETA}")
    print("PASS -- all loss functions OK.")
