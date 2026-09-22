"""
scripts/sanity_train.py
========================
U-Net Training Sanity Test
---------------------------
Purpose
-------
Verify that the full training pipeline (data loading, forward pass, loss
calculation, backward pass, optimizer update, validation pass, and checkpoint
saving) runs without errors on the real MoCA-Video dataset.

This is NOT a performance benchmark. Numbers printed here (loss values, IoU)
carry no meaning about model quality -- the model is randomly initialised and
trained for only a handful of batches.

What is tested
--------------
  [1] Device detection (CUDA if available, else CPU)
  [2] Dataset loader (uses validated utils/dataset.py)
  [3] Model construction (UNet from models/unet/unet.py -- unmodified)
  [4] Forward pass  -- input flows through full encoder-bottleneck-decoder
  [5] Loss calculation -- BCE + Dice (utils/losses.py BCEDiceLoss)
  [6] Backward pass -- gradients computed for all parameters
  [7] Optimizer step -- Adam parameter update applied
  [8] Gradient check -- at least one parameter's gradient is non-zero
  [9] Validation pass -- no-grad forward pass on val loader
 [10] Checkpoint save / load -- state dict written then reloaded correctly

Run
---
    python scripts/sanity_train.py

Expected outcome
----------------
    All 10 checks print PASS and the script exits with code 0.
    A checkpoint file is written to checkpoints/sanity_unet.pth.
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from models.unet.unet import UNet
from utils.dataset import get_loaders
from utils.losses import BCEDiceLoss

# ---------------------------------------------------------------------------
# Sanity-test parameters  (deliberately tiny)
# ---------------------------------------------------------------------------
SANITY_TRAIN_BATCHES = 3   # number of train batches per "epoch"
SANITY_VAL_BATCHES   = 2   # number of val   batches to evaluate
SANITY_EPOCHS        = 2   # number of mini-epochs to run
CKPT_PATH = Path(config.CHECKPOINTS_DIR) / "sanity_unet.pth"

SEP  = "=" * 65
PASS = "  [PASS]"
FAIL = "  [FAIL]"


def step(n: int, desc: str) -> None:
    print(f"\n  [{n:02d}] {desc}")


def ok(msg: str = "") -> None:
    print(f"{PASS}  {msg}")


def fail(msg: str) -> None:
    print(f"{FAIL}  {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main sanity test
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{SEP}")
    print("  U-Net Training Sanity Test")
    print(SEP)

    # ------------------------------------------------------------------
    # [1] Device
    # ------------------------------------------------------------------
    step(1, "Device detection")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"       device = {device}")
    if device.type == "cuda":
        print(f"       GPU    = {torch.cuda.get_device_name(0)}")
        print(f"       VRAM   = {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    ok(f"using {device}")

    # ------------------------------------------------------------------
    # [2] Dataset / DataLoaders
    # ------------------------------------------------------------------
    step(2, "Dataset loader (validated utils/dataset.py)")
    t0 = time.time()
    train_loader, val_loader = get_loaders()
    elapsed = time.time() - t0
    print(f"       train batches total : {len(train_loader)}")
    print(f"       val   batches total : {len(val_loader)}")
    print(f"       loader build time   : {elapsed:.2f}s")
    if len(train_loader) == 0:
        fail("train_loader is empty")
    if len(val_loader) == 0:
        fail("val_loader is empty")
    ok("loaders built successfully")

    # ------------------------------------------------------------------
    # [3] Model construction
    # ------------------------------------------------------------------
    step(3, "UNet construction (no pretrained weights)")
    model = UNet(
        in_channels   = config.UNET_IN_CHANNELS,
        out_channels  = config.UNET_OUT_CHANNELS,
        init_features = config.UNET_INIT_FEATURES,
    ).to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       total parameters     : {total_params:,}")
    print(f"       trainable parameters : {trainable_params:,}")
    ok(f"UNet on {device}, {trainable_params:,} trainable params")

    # ------------------------------------------------------------------
    # Loss + optimizer
    # ------------------------------------------------------------------
    criterion = BCEDiceLoss(
        bce_weight  = config.BCE_WEIGHT,
        dice_weight = config.DICE_WEIGHT,
        smooth      = config.DICE_SMOOTH,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # ------------------------------------------------------------------
    # Mini-training loop
    # ------------------------------------------------------------------
    train_iter = iter(train_loader)

    for epoch in range(1, SANITY_EPOCHS + 1):
        print(f"\n  --- Epoch {epoch}/{SANITY_EPOCHS} ---")

        # ---- Training phase ------------------------------------------
        model.train()
        epoch_losses = []

        for batch_idx in range(SANITY_TRAIN_BATCHES):
            try:
                imgs, masks = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                imgs, masks = next(train_iter)

            imgs  = imgs.to(device)
            masks = masks.to(device)

            # [4] Forward pass
            if epoch == 1 and batch_idx == 0:
                step(4, "Forward pass")
            preds = model(imgs)
            if epoch == 1 and batch_idx == 0:
                print(f"       input  : {tuple(imgs.shape)}")
                print(f"       output : {tuple(preds.shape)}")
                assert preds.shape == masks.shape, (
                    f"Shape mismatch: pred {preds.shape} vs mask {masks.shape}"
                )
                probs = torch.sigmoid(preds)
                assert probs.min() >= 0.0 and probs.max() <= 1.0, (
                    "Sigmoid output out of [0,1]"
                )
                ok("forward pass shape and range correct")

            # [5] Loss calculation
            if epoch == 1 and batch_idx == 0:
                step(5, "Loss calculation (BCE + Dice)")
            loss = criterion(preds, masks)
            if epoch == 1 and batch_idx == 0:
                print(f"       loss value : {loss.item():.6f}")
                assert loss.item() > 0.0, "Loss is zero or negative before any training"
                assert not torch.isnan(loss), "Loss is NaN"
                assert not torch.isinf(loss), "Loss is Inf"
                ok("loss is finite and positive")

            # [6] Backward pass
            if epoch == 1 and batch_idx == 0:
                step(6, "Backward pass (gradient computation)")
            optimizer.zero_grad()
            loss.backward()
            if epoch == 1 and batch_idx == 0:
                ok("loss.backward() completed without error")

            # [8] Gradient check (first batch only)
            if epoch == 1 and batch_idx == 0:
                step(8, "Gradient check (at least one param has non-zero grad)")
                has_grad = any(
                    p.grad is not None and p.grad.abs().sum().item() > 0
                    for p in model.parameters()
                    if p.requires_grad
                )
                if not has_grad:
                    fail("all gradients are zero — backprop may be broken")
                ok("non-zero gradients confirmed")

            # [7] Optimizer step
            if epoch == 1 and batch_idx == 0:
                step(7, "Optimizer step (Adam parameter update)")
            optimizer.step()
            if epoch == 1 and batch_idx == 0:
                ok("optimizer.step() completed without error")

            epoch_losses.append(loss.item())
            print(
                f"       train  epoch={epoch}  batch={batch_idx+1}/{SANITY_TRAIN_BATCHES}"
                f"  loss={loss.item():.6f}"
            )

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"       avg train loss : {avg_train_loss:.6f}")

        # ---- Validation phase ----------------------------------------
        step(9, "Validation pass (no-grad forward)")  # only printed cleanly once
        model.eval()
        val_iter   = iter(val_loader)
        val_losses = []

        with torch.no_grad():
            for batch_idx in range(SANITY_VAL_BATCHES):
                try:
                    imgs, masks = next(val_iter)
                except StopIteration:
                    break
                imgs  = imgs.to(device)
                masks = masks.to(device)
                preds = model(imgs)
                vloss = criterion(preds, masks)
                val_losses.append(vloss.item())
                print(
                    f"       val    epoch={epoch}  batch={batch_idx+1}/{SANITY_VAL_BATCHES}"
                    f"  loss={vloss.item():.6f}"
                )

        if val_losses:
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"       avg val loss   : {avg_val_loss:.6f}")
            ok("validation pass completed without error")
        else:
            print("       WARNING: no val batches completed")

    # ------------------------------------------------------------------
    # [10] Checkpoint save + reload
    # ------------------------------------------------------------------
    step(10, "Checkpoint save and reload")
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch":       SANITY_EPOCHS,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": {
            "in_channels":   config.UNET_IN_CHANNELS,
            "out_channels":  config.UNET_OUT_CHANNELS,
            "init_features": config.UNET_INIT_FEATURES,
        },
    }
    torch.save(checkpoint, CKPT_PATH)
    print(f"       saved  : {CKPT_PATH}")
    print(f"       size   : {CKPT_PATH.stat().st_size / 1e6:.1f} MB")

    # Reload and verify
    loaded = torch.load(CKPT_PATH, map_location=device)
    model2 = UNet(
        in_channels   = loaded["config"]["in_channels"],
        out_channels  = loaded["config"]["out_channels"],
        init_features = loaded["config"]["init_features"],
    ).to(device)
    model2.load_state_dict(loaded["model_state"])

    # Both models must produce identical output on the same input
    model.eval()
    model2.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, config.INPUT_HEIGHT, config.INPUT_WIDTH).to(device)
        out1  = model(dummy)
        out2  = model2(dummy)
    max_diff = (out1 - out2).abs().max().item()
    print(f"       max output diff after reload : {max_diff:.2e}  (must be ~0)")
    if max_diff > 1e-5:
        fail(f"reloaded model output differs by {max_diff:.2e}")
    ok(f"checkpoint saved ({CKPT_PATH.stat().st_size/1e6:.1f} MB) and reloaded correctly")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  All 10 checks PASSED.")
    print()
    print("  NOTE: Loss values and predictions from this test are")
    print("  meaningless -- the model is randomly initialised and")
    print("  trained for only a few batches. No accuracy claims.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
