import json

h = json.load(open('outputs/metrics/training_history.json'))
print(f"{'Ep':>3}  {'tr_loss':>8}  {'tr_iou':>7}  {'val_loss':>8}  {'val_iou':>7}  {'val_dice':>8}  {'val_prec':>8}  {'val_rec':>7}  {'lr':>9}")
print("-" * 90)
for r in h:
    ep  = r["epoch"]
    tl  = r["train_loss"]
    ti  = r["train_iou"]
    vl  = r.get("val_loss",      float("nan"))
    vi  = r.get("val_iou",       float("nan"))
    vd  = r.get("val_dice",      float("nan"))
    vp  = r.get("val_precision", float("nan"))
    vrc = r.get("val_recall",    float("nan"))
    lr  = r["lr"]
    print(f"{ep:3d}  {tl:8.4f}  {ti:7.4f}  {vl:8.4f}  {vi:7.4f}  {vd:8.4f}  {vp:8.4f}  {vrc:7.4f}  {lr:9.2e}")

print()
fm = json.load(open('outputs/metrics/final_metrics.json'))
print("=== FINAL METRICS (best checkpoint) ===")
for k, v in fm.items():
    print(f"  {k}: {v}")
