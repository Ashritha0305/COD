"""
scripts/verify_fixes.py
========================
Verification tests for the logits-based U-Net fix.

Tests:
  A. U-Net forward: shape check, output is raw logits (NOT clamped to [0,1])
  B. Loss test:    BCEWithLogitsLoss + Dice work with raw logits; loss is finite + positive
  C. Backward:     Gradients are non-zero after backward()
  D. Metric test:  compute_metrics correctly applies sigmoid internally
  E. Dataset test: MoCA-Mask loader -- 67 sequences, 3650 pairs, split counts
  F. One-batch:    Full forward->loss->backward->step on real data, no errors

DO NOT run full training -- this script stops after tests.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim

SEP  = "=" * 65
PASS = "PASS"
FAIL = "FAIL"

results = {}


# ---------------------------------------------------------------------------
# A. U-Net forward test
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  TEST A: U-Net forward pass")
print(SEP)

try:
    from models.unet.unet import UNet
    import config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = UNet(
        in_channels   = config.UNET_IN_CHANNELS,
        out_channels  = config.UNET_OUT_CHANNELS,
        init_features = config.UNET_INIT_FEATURES,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    dummy = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        out = model(dummy)

    # Shape check
    assert out.shape == (2, 1, 256, 256), f"Wrong shape: {out.shape}"
    print(f"  Output shape: {out.shape}  [expected (2,1,256,256)]")

    # Logits check: output should NOT be constrained to [0,1]
    out_min = out.min().item()
    out_max = out.max().item()
    print(f"  Output min: {out_min:.4f}  max: {out_max:.4f}")
    print(f"  (Raw logits -- values outside [0,1] are EXPECTED and correct)")

    # sigmoid of logits must be in [0,1]
    probs = torch.sigmoid(out)
    assert probs.min().item() >= 0.0 and probs.max().item() <= 1.0
    print(f"  sigmoid(output) min: {probs.min().item():.4f}  max: {probs.max().item():.4f}  [0,1] ok")

    # Parameter count sanity (should be ~31M for F=64)
    assert total_params > 30_000_000, f"Param count too low: {total_params:,}"

    print(f"  {PASS} -- A: U-Net forward")
    results["A"] = True

except Exception as e:
    print(f"  {FAIL} -- A: {e}")
    results["A"] = False


# ---------------------------------------------------------------------------
# B. Loss test
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  TEST B: Loss functions with raw logits")
print(SEP)

try:
    from utils.losses import BCELoss, DiceLoss, BCEDiceLoss, get_loss_fn
    import torch.nn as nn

    B, H, W = 4, 64, 64
    logits = torch.randn(B, 1, H, W)                   # raw logits
    target = (torch.rand(B, 1, H, W) > 0.5).float()    # binary {0,1}

    for name, fn in [("BCELoss", BCELoss()),
                     ("DiceLoss", DiceLoss()),
                     ("BCEDiceLoss", BCEDiceLoss()),
                     (f"get_loss_fn() [{config.LOSS_TYPE}]", get_loss_fn())]:
        loss_val = fn(logits, target)
        v = loss_val.item()
        ok = torch.isfinite(loss_val).item() and v >= 0
        status = PASS if ok else FAIL
        print(f"  {status} -- {name:35s}  loss = {v:.6f}  finite={torch.isfinite(loss_val).item()}")

    # Verify BCEWithLogitsLoss is used (not nn.BCELoss)
    bce_fn = BCELoss()
    assert isinstance(bce_fn._bce, nn.BCEWithLogitsLoss), \
        f"BCELoss._bce is {type(bce_fn._bce)}, expected BCEWithLogitsLoss"
    print(f"  {PASS} -- BCELoss uses nn.BCEWithLogitsLoss confirmed")

    # Verify DiceLoss applies sigmoid internally
    dice_fn = DiceLoss()
    large_pos = torch.full((2, 1, 8, 8), 10.0)   # sigmoid(10) ~= 1.0
    tgt_ones  = torch.ones(2, 1, 8, 8)
    dice_perfect = dice_fn(large_pos, tgt_ones).item()
    print(f"  DiceLoss(logit=+10, target=1) = {dice_perfect:.6f}  [expected ~= 0.0]")
    assert dice_perfect < 0.01, f"Dice should be ~0 for perfect match via sigmoid, got {dice_perfect}"
    print(f"  {PASS} -- DiceLoss applies sigmoid correctly")

    results["B"] = True

except Exception as e:
    print(f"  {FAIL} -- B: {e}")
    import traceback; traceback.print_exc()
    results["B"] = False


# ---------------------------------------------------------------------------
# C. Backward test
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  TEST C: Gradient flow (backward)")
print(SEP)

try:
    from models.unet.unet import UNet
    from utils.losses import get_loss_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = get_loss_fn()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    dummy_img  = torch.randn(2, 3, 256, 256).to(device)
    dummy_mask = (torch.rand(2, 1, 256, 256) > 0.5).float().to(device)

    optimizer.zero_grad()
    preds = model(dummy_img)
    loss  = criterion(preds, dummy_mask)
    loss.backward()

    # Check that gradients are non-zero
    grad_norms = [p.grad.abs().max().item()
                  for p in model.parameters() if p.grad is not None]
    nonzero = sum(1 for g in grad_norms if g > 0)
    max_grad = max(grad_norms) if grad_norms else 0.0

    print(f"  Loss value       : {loss.item():.6f}")
    print(f"  Params with grad : {len(grad_norms)}")
    print(f"  Non-zero grads   : {nonzero}")
    print(f"  Max |grad|       : {max_grad:.6e}")

    assert nonzero > 0, "All gradients are zero!"
    assert torch.isfinite(loss), "Loss is not finite"

    optimizer.step()
    print(f"  optimizer.step() completed")
    print(f"  {PASS} -- C: backward + optimizer.step()")
    results["C"] = True

except Exception as e:
    print(f"  {FAIL} -- C: {e}")
    import traceback; traceback.print_exc()
    results["C"] = False


# ---------------------------------------------------------------------------
# D. Metric test (controlled logits)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  TEST D: compute_metrics with controlled logits")
print(SEP)

try:
    from utils.metrics import compute_metrics

    B, H, W = 2, 32, 32

    # Case 1: All-foreground correct
    # logit = +20 -> sigmoid ~= 1.0 -> >= 0.5 -> pred_bin = 1; target = 1 -> IoU ~= 1.0
    target_fg = torch.ones(B, 1, H, W)
    logits_fg = torch.full((B, 1, H, W), 20.0)
    m = compute_metrics(logits_fg, target_fg)
    print(f"  All-fg correct:   IoU={m.iou:.4f}  Dice={m.dice:.4f}  [expect ~1.0]")
    assert abs(m.iou - 1.0) < 1e-3, f"IoU={m.iou} expected ~1.0"
    assert abs(m.dice - 1.0) < 1e-3, f"Dice={m.dice} expected ~1.0"

    # Case 2: All-background correct
    # logit = -20 -> sigmoid ~= 0.0 -> < 0.5 -> pred_bin = 0; target = 0 -> accuracy ~= 1.0
    target_bg = torch.zeros(B, 1, H, W)
    logits_bg = torch.full((B, 1, H, W), -20.0)
    m = compute_metrics(logits_bg, target_bg)
    print(f"  All-bg correct:   IoU={m.iou:.6f}  Acc={m.accuracy:.4f}  [Acc ~1.0]")
    assert m.accuracy > 0.99, f"Accuracy={m.accuracy} expected ~1.0"

    # Case 3: Zero logit -> sigmoid(0) = 0.5 -> >= 0.5 -> pred_bin = 1; target = 1 -> IoU ~= 1.0
    zero_logit    = torch.zeros(B, 1, H, W)
    target_fg2    = torch.ones(B, 1, H, W)
    m = compute_metrics(zero_logit, target_fg2)
    print(f"  logit=0, fg target: IoU={m.iou:.4f}  Dice={m.dice:.4f}  [sigmoid(0)=0.5 >= 0.5 -> pred=1]")
    assert abs(m.iou - 1.0) < 1e-3, \
        f"sigmoid(0)=0.5 should be >= threshold 0.5, giving pred_bin=1; IoU={m.iou}"

    # Case 4: Small negative logit -> sigmoid(-0.01) ~= 0.4975 < 0.5 -> pred_bin = 0; target = 1 -> IoU ~= 0
    neg_logit  = torch.full((B, 1, H, W), -0.01)
    target_fg3 = torch.ones(B, 1, H, W)
    m = compute_metrics(neg_logit, target_fg3)
    print(f"  logit=-0.01, fg target: IoU={m.iou:.6f}  [sigmoid(-0.01)<0.5 -> pred=0, IoU~0]")
    assert m.iou < 0.01, f"sigmoid(-0.01)<0.5, pred_bin=0, IoU should ~0; got {m.iou}"

    print(f"  {PASS} -- D: compute_metrics sigmoid conversion verified")
    results["D"] = True

except Exception as e:
    print(f"  {FAIL} -- D: {e}")
    import traceback; traceback.print_exc()
    results["D"] = False


# ---------------------------------------------------------------------------
# E. Dataset test
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  TEST E: Dataset / DataLoader")
print(SEP)

try:
    from pathlib import Path
    from utils.dataset import discover_sequences, sequence_level_split, get_loaders

    train_dir = Path(config.MOCA_TRAIN_DIR)
    print(f"  Dataset root: {train_dir}")

    sequences, n_folders = discover_sequences(train_dir)
    total_pairs           = sum(len(s.pairs)           for s in sequences)
    total_unmatched_masks = sum(len(s.unmatched_masks) for s in sequences)

    print(f"  Sequence folders found : {n_folders}")
    print(f"  Valid sequences        : {len(sequences)}   [expected 67]")
    print(f"  Total valid pairs      : {total_pairs}   [expected 3650]")
    print(f"  Total unmatched masks  : {total_unmatched_masks}  [expected >=14]")

    assert len(sequences) == 67, f"Expected 67 sequences, got {len(sequences)}"
    assert total_pairs == 3650, f"Expected 3650 pairs, got {total_pairs}"

    train_seqs, val_seqs = sequence_level_split(sequences, config.VAL_FRACTION, config.SEED)
    overlap = {s.name for s in train_seqs} & {s.name for s in val_seqs}

    print(f"  Train sequences        : {len(train_seqs)}   [expected 57]")
    print(f"  Val sequences          : {len(val_seqs)}   [expected 10]")
    print(f"  Sequence overlap       : {len(overlap)}   [expected 0]")

    assert len(train_seqs) == 57, f"Expected 57 train seqs, got {len(train_seqs)}"
    assert len(val_seqs)   == 10, f"Expected 10 val seqs, got {len(val_seqs)}"
    assert len(overlap)    ==  0, f"Sequences in both splits: {overlap}"

    # DataLoader smoke test
    train_loader, val_loader = get_loaders()
    imgs, masks = next(iter(train_loader))
    print(f"  Batch img shape        : {tuple(imgs.shape)}")
    print(f"  Batch mask shape       : {tuple(masks.shape)}")
    print(f"  Mask unique values     : {sorted(masks.unique().tolist())}")
    assert set(masks.unique().tolist()).issubset({0.0, 1.0}), "Mask not binary"

    print(f"  {PASS} -- E: Dataset counts and loader correct")
    results["E"] = True

except Exception as e:
    print(f"  {FAIL} -- E: {e}")
    import traceback; traceback.print_exc()
    results["E"] = False


# ---------------------------------------------------------------------------
# F. One-batch training test (real data)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  TEST F: One-batch training (real data)")
print(SEP)

try:
    from models.unet.unet import UNet
    from utils.losses import get_loss_fn
    from utils.metrics import compute_metrics
    from utils.dataset import get_loaders

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = UNet().to(device)
    criterion = get_loss_fn()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler    = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"  Model: fresh random init (no checkpoint loaded)")
    print(f"  Loss : {config.LOSS_TYPE}  BCE_w={config.BCE_WEIGHT}  Dice_w={config.DICE_WEIGHT}")

    train_loader, _ = get_loaders()
    imgs, masks = next(iter(train_loader))
    imgs  = imgs.to(device)
    masks = masks.to(device)

    print(f"  Input batch : {tuple(imgs.shape)}")
    print(f"  Mask batch  : {tuple(masks.shape)}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        preds = model(imgs)

    preds_f32 = preds.float()
    loss = criterion(preds_f32, masks)

    print(f"  Forward done.  preds shape: {tuple(preds.shape)}")
    print(f"  Logit range: [{preds_f32.min().item():.4f}, {preds_f32.max().item():.4f}]  (raw logits)")
    print(f"  Loss value : {loss.item():.6f}   finite={torch.isfinite(loss).item()}")

    with torch.no_grad():
        m = compute_metrics(preds_f32.detach(), masks)
    print(f"  Train metrics (random init, 1 batch): {m}")

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f"  Backward + optimizer.step() completed")

    assert torch.isfinite(loss), "Loss not finite"

    print(f"  {PASS} -- F: One-batch training completed without errors")
    results["F"] = True

except Exception as e:
    print(f"  {FAIL} -- F: {e}")
    import traceback; traceback.print_exc()
    results["F"] = False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  VERIFICATION SUMMARY")
print(SEP)
all_pass = True
for test_id in ["A", "B", "C", "D", "E", "F"]:
    status = PASS if results.get(test_id) else FAIL
    print(f"  Test {test_id}: {status}")
    if not results.get(test_id):
        all_pass = False

print(SEP)
if all_pass:
    print("  ALL TESTS PASSED -- fixes verified.")
    print("  Ready to start full training when instructed.")
else:
    print("  ONE OR MORE TESTS FAILED -- review output above.")
print(SEP)
print()
