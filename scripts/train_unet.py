"""
scripts/train_unet.py
======================
Full U-Net training on MoCA-Video (TrainDataset_per_sq).

DIAGNOSIS FIXES applied (v3):
-----------------------------------------------------------------------
  RC-1: UNet init_features 64 -> 32 (4M params vs 31M), Dropout2d in encoder
  RC-2: BCE pos_weight disabled; TverskyLoss (alpha=0.3, beta=0.7) as sole loss
  RC-3: CosineAnnealingLR replaced by ReduceLROnPlateau(mode='max', patience=3)
        keyed on val IoU -- LR only reduces when val IoU stops improving
  RC-4: Early stopping patience reduced 8 -> 5
  RC-5: RandomResizedCrop scale min 0.5 -> 0.7 (in dataset.py / config.py)
  RC-6: CBAM attention module added at bottleneck (in unet.py)

Baseline (epoch 12, the best the old model ever achieved):
  val IoU=0.2184  Dice=0.3248  Precision=0.4247  Recall=0.3494

What this script does
---------------------
  1.  Seeds all random sources for reproducibility (SEED=42).
  2.  Prints full configuration before training starts.
  3.  Builds train/val DataLoaders via the validated dataset loader
      (sequence-level split, ~57 train / ~10 val sequences, SEED=42).
  4.  Constructs UNet (v2: F=32, CBAM) from RANDOM initialisation.
  5.  Trains for up to NUM_EPOCHS epochs with:
        - Mixed-precision forward/backward (AMP, CUDA only)
        - Adam optimiser with weight_decay
        - ReduceLROnPlateau(mode='max', patience=3) keyed on val IoU
        - Tversky loss (alpha=0.3, beta=0.7)
        - Early stopping based on val IoU (patience=EARLY_STOP_PATIENCE)
  6.  After every epoch:
        - Logs train loss + metrics to console
        - Runs full validation pass (every VAL_INTERVAL epochs)
        - Saves best-checkpoint (best val IoU) to checkpoints/unet_best.pth
        - Saves last-checkpoint after every epoch to checkpoints/unet_last.pth
        - Saves periodic checkpoint every SAVE_INTERVAL epochs
        - Appends to train_vs_val_iou.json for plotting
  7.  After training completes:
        - Loads best checkpoint
        - Runs a final full validation pass and computes IoU/Dice/Prec/Recall
        - Saves qualitative examples (image, GT mask, predicted mask, overlay)
        - Writes training_history.json, final_metrics.json, train_vs_val_iou.json
        - Prints a concise results summary with comparison to baseline

Configuration source: config.py (no hard-coded values).

Usage::

    python scripts/train_unet.py

All hyperparameters are in config.py.  Do NOT modify any architecture file.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image, ImageDraw
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from models.unet.unet  import UNet
from utils.dataset     import get_loaders
from utils.losses      import get_loss_fn
from utils.metrics     import compute_metrics, MetricAccumulator

SEP = "=" * 65

# Baseline metrics from the original (v1) checkpoint at its best epoch (12).
# Used in the final summary to confirm improvement.
_BASELINE_VAL_IOU  = 0.2184
_BASELINE_VAL_DICE = 0.3248
_BASELINE_VAL_PREC = 0.4247
_BASELINE_VAL_REC  = 0.3494


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler) -> dict:
    """One full training epoch. Returns average loss + metrics."""
    model.train()
    loss_sum   = 0.0
    metric_acc = MetricAccumulator()

    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for imgs, masks in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(imgs)
        # TverskyLoss requires float32; cast back after autocast block
        preds_f32 = preds.float()
        loss = criterion(preds_f32, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        with torch.no_grad():
            metric_acc.update(compute_metrics(preds_f32.detach(), masks))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    m = metric_acc.average()
    return {
        "loss":      loss_sum / n,
        "iou":       m.iou,
        "dice":      m.dice,
        "precision": m.precision,
        "recall":    m.recall,
        "accuracy":  m.accuracy,
    }


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    """Full validation pass. Returns average loss + metrics."""
    model.eval()
    loss_sum   = 0.0
    metric_acc = MetricAccumulator()

    pbar = tqdm(loader, desc="  Val  ", leave=False, unit="batch")
    for imgs, masks in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            preds = model(imgs)
        # Cast back to float32 after autocast
        preds_f32 = preds.float()
        loss = criterion(preds_f32, masks)

        loss_sum += loss.item()
        metric_acc.update(compute_metrics(preds_f32, masks))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    m = metric_acc.average()
    return {
        "loss":      loss_sum / n,
        "iou":       m.iou,
        "dice":      m.dice,
        "precision": m.precision,
        "recall":    m.recall,
        "accuracy":  m.accuracy,
    }


# ---------------------------------------------------------------------------
# Qualitative output: save image / GT mask / predicted mask / overlay
# ---------------------------------------------------------------------------

# ImageNet mean/std used during normalisation -- needed to de-normalise
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm(img_t: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation and return uint8 HxWx3 array."""
    img = img_t.cpu().float() * _IMAGENET_STD + _IMAGENET_MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def save_qualitative_examples(
    model,
    val_loader,
    device,
    out_dir: Path,
    n_examples: int = 10,
) -> None:
    """
    Save side-by-side qualitative panels:
        [original image | GT mask | predicted mask | overlay]

    Saves n_examples panels as PNGs. Includes a probability-map panel
    (soft U-Net output before threshold) to reveal whether the model is
    confidently wrong or merely uncertain.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            if saved >= n_examples:
                break
            imgs  = imgs.to(device)
            masks = masks.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                preds = model(imgs)

            for i in range(imgs.size(0)):
                if saved >= n_examples:
                    break

                # De-normalise image
                img_np  = _denorm(imgs[i])                      # HxWx3  uint8

                # GT mask -> uint8 grayscale, then RGB
                gt_np   = (masks[i, 0].cpu().numpy() * 255).astype(np.uint8)
                gt_rgb  = np.stack([gt_np] * 3, axis=-1)

                # Soft probability map (not thresholded) -- blue=low, red=high
                prob_np  = torch.sigmoid(preds[i, 0]).cpu().float().numpy()
                prob_u8  = (np.clip(prob_np, 0, 1) * 255).astype(np.uint8)
                # Simple viridis-like: multiply blue channel down, red up
                prob_rgb = np.zeros((*prob_u8.shape, 3), dtype=np.uint8)
                prob_rgb[:, :, 0] = prob_u8                    # R = prob
                prob_rgb[:, :, 2] = 255 - prob_u8             # B = 1-prob

                # Predicted mask: sigmoid -> threshold at SEG_THRESHOLD
                pr_np   = ((prob_np >= config.SEG_THRESHOLD)
                           .astype(np.uint8) * 255)
                pr_rgb  = np.stack([pr_np] * 3, axis=-1)

                # Overlay: red channel = prediction on original image
                overlay = img_np.copy()
                mask_bool = pr_np > 127
                overlay[mask_bool, 0] = np.clip(
                    overlay[mask_bool, 0].astype(np.int32) + 80, 0, 255
                ).astype(np.uint8)
                overlay[mask_bool, 1] = (overlay[mask_bool, 1] * 0.5).astype(np.uint8)
                overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.5).astype(np.uint8)

                H, W = img_np.shape[:2]
                panel = np.zeros((H, W * 5, 3), dtype=np.uint8)
                panel[:, 0*W:1*W] = img_np
                panel[:, 1*W:2*W] = gt_rgb
                panel[:, 2*W:3*W] = prob_rgb      # soft probability map
                panel[:, 3*W:4*W] = pr_rgb        # binary threshold
                panel[:, 4*W:5*W] = overlay

                pil = Image.fromarray(panel)
                draw = ImageDraw.Draw(pil)
                for col_idx, label in enumerate(
                    ["Image", "GT Mask", "Prob Map", "Pred Mask", "Overlay"]
                ):
                    draw.text((col_idx * W + 4, 4), label, fill=(255, 255, 0))

                fname = out_dir / f"val_example_{saved+1:03d}.png"
                pil.save(fname)
                saved += 1

    print(f"    Saved {saved} qualitative examples -> {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ config banner
    print(f"\n{SEP}")
    print("  wildlife-COD  --  U-Net Training  (v3 -- diagnosis fixes applied)")
    print(SEP)
    print(f"  Device         : {device}")
    if device.type == "cuda":
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM           : {vram:.1f} GB")
    print(f"  Dataset root   : {config.MOCA_TRAIN_DIR}")
    print(f"  SEED           : {config.SEED}")
    print(f"  VAL_FRACTION   : {config.VAL_FRACTION}")
    print(f"  Max epochs     : {config.NUM_EPOCHS}  (early stopping patience={config.EARLY_STOP_PATIENCE})")
    print(f"  Batch size     : {config.BATCH_SIZE}")
    print(f"  Learning rate  : {config.LEARNING_RATE}")
    print(f"  Weight decay   : {config.WEIGHT_DECAY}   (L2 regularisation)")
    print(f"  LR schedule    : ReduceLROnPlateau(mode=max, patience=3, factor=0.5)")
    print(f"  Loss           : {config.LOSS_TYPE}  "
          f"(alpha={config.TVERSKY_ALPHA}, beta={config.TVERSKY_BETA})")
    print(f"  U-Net features : {config.UNET_INIT_FEATURES}  (RC-1: was 64, now 32 -> ~4M params)")
    print(f"  Dropout rate   : {config.DROPOUT_RATE}  (spatial dropout in encoder)")
    print(f"  CBAM           : bottleneck-only (channel + spatial attention)")
    print(f"  Crop scale min : {config.RANDOM_CROP_SCALE_MIN}  (RC-5: was 0.5)")
    print(f"  Input size     : {config.INPUT_HEIGHT}x{config.INPUT_WIDTH}")
    print(f"  Mixed precision: {device.type == 'cuda'}")
    print(f"  Val interval   : every {config.VAL_INTERVAL} epoch(s)")
    print(f"  Save interval  : every {config.SAVE_INTERVAL} epoch(s)")
    print(f"  Min fg filter  : {config.MIN_FOREGROUND_FRAC * 100:.1f}% (removes annotation errors)")
    print(f"  Best ckpt      : {config.BEST_MODEL_PATH}")
    print(f"  Last ckpt      : {config.LAST_MODEL_PATH}")
    print(f"  Pretrained     : NONE  (random init)")
    print(SEP)

    # ------------------------------------------------------------------ directories
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.METRICS_OUT_DIR, exist_ok=True)
    qual_dir = Path(config.OUTPUTS_DIR) / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("\n  Building DataLoaders ...")
    train_loader, val_loader = get_loaders()
    print(f"  Train: {len(train_loader)} batches  "
          f"({len(train_loader.dataset)} samples)")
    print(f"  Val  : {len(val_loader)} batches  "
          f"({len(val_loader.dataset)} samples)")

    # ------------------------------------------------------------------ model
    model     = UNet(
        in_channels   = config.UNET_IN_CHANNELS,
        out_channels  = config.UNET_OUT_CHANNELS,
        init_features = config.UNET_INIT_FEATURES,
        dropout_p     = config.DROPOUT_RATE,
    ).to(device)
    criterion = get_loss_fn()
    optimizer = optim.Adam(
        model.parameters(),
        lr           = config.LEARNING_RATE,
        weight_decay = config.WEIGHT_DECAY,
    )

    # DIAGNOSIS FIX (RC-3): Replace CosineAnnealingLR with ReduceLROnPlateau.
    # CosineAnnealingLR decayed LR to 1e-6 by epoch 50 regardless of val performance,
    # trapping the model in an overfit minimum once val IoU peaked at epoch 12.
    # ReduceLROnPlateau(mode='max') only reduces LR when val IoU stops improving,
    # naturally coordinating LR reduction with actual training progress.
    # patience=3: reduces LR after 3 epochs without val IoU improvement.
    # factor=0.5: halves the LR on each reduction.
    # min_lr=1e-6: prevents LR from becoming too small to update weights.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode      = "max",    # maximise val IoU
        patience  = 3,
        factor    = 0.5,
        min_lr    = 1e-6,
    )

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters : {total_params:,}  (random init)")
    print(f"  (Previous F=64 model had ~31M params; this F=32 model has ~{total_params//1_000_000}M)")

    # ------------------------------------------------------------------ training loop
    best_val_iou      = -1.0
    best_val_epoch    = -1
    patience_counter  = 0
    history           = []
    iou_curve         = []       # simple list of {epoch, train_iou, val_iou} for plotting
    total_t0          = time.time()
    stopped_early     = False

    print(f"\n  Starting training (max {config.NUM_EPOCHS} epochs, "
          f"early stop patience={config.EARLY_STOP_PATIENCE}) ...\n")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        val_stats = None
        if epoch % config.VAL_INTERVAL == 0:
            val_stats = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        # Console line with train/val IoU gap highlighted
        val_str = ""
        gap_str = ""
        if val_stats:
            gap = train_stats["iou"] - val_stats["iou"]
            val_str = (
                f"  val_loss={val_stats['loss']:.4f}"
                f"  val_IoU={val_stats['iou']:.4f}"
                f"  val_Dice={val_stats['dice']:.4f}"
            )
            # Highlight large gaps -- the key overfitting indicator
            gap_str = f"  [gap={gap:+.3f}]"
        patience_str = f"  patience={patience_counter}/{config.EARLY_STOP_PATIENCE}"
        print(
            f"  Epoch [{epoch:3d}/{config.NUM_EPOCHS}]"
            f"  train_loss={train_stats['loss']:.4f}"
            f"  train_IoU={train_stats['iou']:.4f}"
            f"  lr={lr_now:.2e}"
            f"  [{elapsed:.0f}s]{val_str}{gap_str}{patience_str}"
        )

        # History record (per epoch)
        record = {
            "epoch": epoch,
            "lr":    lr_now,
            **{f"train_{k}": v for k, v in train_stats.items()},
        }
        if val_stats:
            record.update({f"val_{k}": v for k, v in val_stats.items()})
        history.append(record)

        # IoU curve record (lightweight, used for plotting)
        iou_record = {"epoch": epoch, "train_iou": train_stats["iou"]}
        if val_stats:
            iou_record["val_iou"] = val_stats["iou"]
        iou_curve.append(iou_record)

        # ---- ReduceLROnPlateau step (DIAGNOSIS FIX RC-3) ----
        # Must be called with the metric we're monitoring (val IoU).
        # Only step when we have a fresh validation result.
        if val_stats:
            scheduler.step(val_stats["iou"])

        # ---- Best checkpoint (by val IoU) + patience tracking ----
        if val_stats:
            if val_stats["iou"] > best_val_iou:
                best_val_iou   = val_stats["iou"]
                best_val_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "epoch":       epoch,
                        "model_state": model.state_dict(),
                        "val_iou":     best_val_iou,
                        "val_stats":   val_stats,
                        "config": {
                            "in_channels":   config.UNET_IN_CHANNELS,
                            "out_channels":  config.UNET_OUT_CHANNELS,
                            "init_features": config.UNET_INIT_FEATURES,
                            "dropout_p":     config.DROPOUT_RATE,
                            "input_height":  config.INPUT_HEIGHT,
                            "input_width":   config.INPUT_WIDTH,
                        },
                    },
                    config.BEST_MODEL_PATH,
                )
                print(
                    f"    *** Best checkpoint saved "
                    f"(epoch {epoch}, val IoU={best_val_iou:.4f})"
                    f" -> {config.BEST_MODEL_PATH}"
                )
            else:
                patience_counter += 1

        # Periodic checkpoint
        if epoch % config.SAVE_INTERVAL == 0:
            ppath = os.path.join(
                config.CHECKPOINTS_DIR, f"unet_epoch_{epoch:03d}.pth"
            )
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                ppath,
            )

        # Last checkpoint (every epoch)
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict()},
            config.LAST_MODEL_PATH,
        )

        # ---- Early stopping (DIAGNOSIS FIX RC-4: patience=5 not 8) ----
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(
                f"\n  === Early stopping triggered at epoch {epoch} "
                f"(best val IoU={best_val_iou:.4f} at epoch {best_val_epoch}) ==="
            )
            stopped_early = True
            break

    total_time = time.time() - total_t0

    # ------------------------------------------------------------------ save history
    history_path = Path(config.METRICS_OUT_DIR) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save lightweight IoU curve for plotting
    iou_curve_path = Path(config.METRICS_OUT_DIR) / "train_vs_val_iou.json"
    with open(iou_curve_path, "w") as f:
        json.dump(iou_curve, f, indent=2)
    print(f"  IoU curve saved -> {iou_curve_path}")

    # ------------------------------------------------------------------ final metrics
    # Load best checkpoint and run a clean final validation pass
    if best_val_epoch < 0:
        print("\n  WARNING: No validation epoch completed. Using last model state.")
        best_val_epoch = epoch
    else:
        print(f"\n  Loading best checkpoint (epoch {best_val_epoch}) for final evaluation ...")
        ckpt = torch.load(config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    print("  Running final validation pass ...")
    final_val = validate(model, val_loader, criterion, device)

    # Save final metrics JSON
    actual_epochs = len(history)
    final_metrics = {
        "version":             "v3_diagnosis_fixes",
        "stopped_early":       stopped_early,
        "best_epoch":          best_val_epoch,
        "actual_epochs":       actual_epochs,
        "total_train_time_s":  round(total_time, 1),
        "final_val_loss":      final_val["loss"],
        "final_val_iou":       final_val["iou"],
        "final_val_dice":      final_val["dice"],
        "final_val_prec":      final_val["precision"],
        "final_val_rec":       final_val["recall"],
        "final_val_acc":       final_val["accuracy"],
        "final_train_loss":    history[-1]["train_loss"],
        # v3 settings (diagnosis fixes)
        "unet_init_features":  config.UNET_INIT_FEATURES,
        "dropout_rate":        config.DROPOUT_RATE,
        "loss_type":           config.LOSS_TYPE,
        "tversky_alpha":       config.TVERSKY_ALPHA,
        "tversky_beta":        config.TVERSKY_BETA,
        "bce_pos_weight":      config.BCE_POS_WEIGHT,
        "weight_decay":        config.WEIGHT_DECAY,
        "early_stop_patience": config.EARLY_STOP_PATIENCE,
        "crop_scale_min":      config.RANDOM_CROP_SCALE_MIN,
        # Baseline for comparison
        "baseline_val_iou":    _BASELINE_VAL_IOU,
        "baseline_val_dice":   _BASELINE_VAL_DICE,
        "improvement_iou":     round(final_val["iou"] - _BASELINE_VAL_IOU, 6),
    }
    metrics_path = Path(config.METRICS_OUT_DIR) / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    # ------------------------------------------------------------------ qualitative
    print("  Saving qualitative validation examples (10 panels with probability maps) ...")
    save_qualitative_examples(model, val_loader, device, qual_dir, n_examples=10)

    # ------------------------------------------------------------------ results banner
    last_train_loss = history[-1]["train_loss"]
    val_records = [r for r in history if "val_loss" in r]
    last_val_loss = val_records[-1]["val_loss"] if val_records else float("nan")
    last_train_iou = history[-1]["train_iou"]

    # Compute final train/val gap
    train_val_gap = last_train_iou - final_val["iou"]

    h  = int(total_time // 3600)
    m_ = int((total_time % 3600) // 60)
    s_ = int(total_time % 60)
    avg_epoch_time = total_time / max(actual_epochs, 1)

    print(f"\n{SEP}")
    print("  Training Complete -- Measured Results (v3 Diagnosis Fixes)")
    print(SEP)
    print(f"  Total training time   : {h}h {m_:02d}m {s_:02d}s")
    print(f"  Avg epoch time        : {avg_epoch_time:.1f}s")
    print(f"  Epochs completed      : {actual_epochs}  "
          f"{'(EARLY STOPPED)' if stopped_early else '(full run)'}")
    print(f"  Final train loss      : {last_train_loss:.6f}")
    print(f"  Final val   loss      : {last_val_loss:.6f}   (last val epoch)")
    print(f"  Best val IoU epoch    : {best_val_epoch}")
    print(f"  Best val IoU          : {best_val_iou:.6f}   (at checkpoint)")
    print(f"  Train/val IoU gap     : {train_val_gap:+.4f}  (was +0.73 in v1)")
    print(f"")
    print(f"  Final validation metrics (from best checkpoint):")
    print(f"    IoU       : {final_val['iou']:.6f}"
          f"    (baseline: {_BASELINE_VAL_IOU}  delta: {final_val['iou']-_BASELINE_VAL_IOU:+.4f})")
    print(f"    Dice      : {final_val['dice']:.6f}"
          f"    (baseline: {_BASELINE_VAL_DICE}  delta: {final_val['dice']-_BASELINE_VAL_DICE:+.4f})")
    print(f"    Precision : {final_val['precision']:.6f}"
          f"    (baseline: {_BASELINE_VAL_PREC})")
    print(f"    Recall    : {final_val['recall']:.6f}"
          f"    (baseline: {_BASELINE_VAL_REC})")
    print(f"    Accuracy  : {final_val['accuracy']:.6f}")
    print(f"")
    print(f"  v3 diagnosis fixes applied:")
    print(f"    RC-1  U-Net F=64->32     : ~31M -> ~4M params, Dropout2d in encoder")
    print(f"    RC-2  Tversky loss       : alpha={config.TVERSKY_ALPHA}, beta={config.TVERSKY_BETA}  (no BCE pos_weight)")
    print(f"    RC-3  ReduceLROnPlateau  : patience=3, factor=0.5, keyed on val IoU")
    print(f"    RC-4  Early stop         : patience={config.EARLY_STOP_PATIENCE} (was 8)")
    print(f"    RC-5  Crop scale         : min={config.RANDOM_CROP_SCALE_MIN} (was 0.5)")
    print(f"    RC-6  CBAM               : bottleneck-only, channel+spatial attention")
    print(f"")
    print(f"  Saved artefacts:")
    print(f"    Best checkpoint   : {config.BEST_MODEL_PATH}")
    print(f"    Last checkpoint   : {config.LAST_MODEL_PATH}")
    print(f"    Training history  : {history_path}")
    print(f"    IoU curve         : {iou_curve_path}")
    print(f"    Final metrics     : {metrics_path}")
    print(f"    Qualitative outs  : {qual_dir}")
    print(SEP)


if __name__ == "__main__":
    main()
