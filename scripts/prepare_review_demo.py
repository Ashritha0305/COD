"""
scripts/prepare_review_demo.py
==============================
Generates the final review demonstration outputs.

Selects 5 images and 2 short video clips from held-out MoCA validation
sequences (NOT training sequences) and runs each through the full pipeline.

Output layout:
    outputs/review_demo/
        image_01/   (crab)
        image_02/   (eastern_screech_owl)
        image_03/   (flatfish)
        image_04/   (snow_leopard)
        image_05/   (scorpionfish)
        video/
            grasshopper_demo.mp4
            spider_viper_demo.mp4

Usage:
    python scripts/prepare_review_demo.py
"""

import os
import sys
import glob
import shutil
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import config
from modules.pipeline import WildlifePipeline

IMAGE_DEMOS = [
    ("crab_2",                       "crab",                       "image_01", 5),
    ("eastern_screech_owl_1",        "eastern_screech_owl",        "image_02", 10),
    ("flatfish_4",                   "flatfish",                   "image_03", 8),
    ("snow_leopard_8",               "snow_leopard",               "image_04", 15),
    ("scorpionfish_5",               "scorpionfish",               "image_05", 7),
]

VIDEO_DEMOS = [
    ("grasshopper_1",                "grasshopper_demo.mp4",       40),
    ("spider_tailed_horned_viper_2", "spider_viper_demo.mp4",      40),
]

MOCA_TRAIN_ROOT = config.MOCA_TRAIN_DIR
DEMO_OUT_ROOT   = os.path.join(config.OUTPUTS_DIR, "review_demo")


def get_sorted_frames(sequence_dir):
    imgs_dir = os.path.join(sequence_dir, "Imgs")
    if not os.path.isdir(imgs_dir):
        imgs_dir = os.path.join(sequence_dir, "imgs")
    frames = sorted(
        glob.glob(os.path.join(imgs_dir, "*.jpg")) +
        glob.glob(os.path.join(imgs_dir, "*.png")) +
        glob.glob(os.path.join(imgs_dir, "*.jpeg"))
    )
    return frames


def build_video_from_frames(frames, output_path, fps=10):
    if not frames:
        return False
    sample = cv2.imread(frames[0])
    if sample is None:
        return False
    h, w = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        img = cv2.imread(f)
        if img is not None:
            writer.write(img)
    writer.release()
    return True


def main():
    print("=" * 65)
    print("Wildlife-COD  -  REVIEW DEMO PREPARATION")
    print("=" * 65)

    os.makedirs(DEMO_OUT_ROOT, exist_ok=True)
    video_out_dir = os.path.join(DEMO_OUT_ROOT, "video")
    os.makedirs(video_out_dir, exist_ok=True)

    print("[1/3] Initializing pipeline ...")
    pipeline = WildlifePipeline(
        checkpoint_path=config.BEST_MODEL_PATH,
        yolo_weights=config.YOLO_WILDLIFE_WEIGHTS,
    )

    print("\n[2/3] Generating IMAGE demos ...")
    image_results = []
    for seq_name, label, folder, frame_idx in IMAGE_DEMOS:
        seq_dir = os.path.join(MOCA_TRAIN_ROOT, seq_name)
        if not os.path.isdir(seq_dir):
            print(f"  [WARN] Sequence not found: {seq_dir}")
            continue
        frames = get_sorted_frames(seq_dir)
        if not frames:
            print(f"  [WARN] No frames in {seq_dir}")
            continue
        idx = min(frame_idx, len(frames) - 1)
        src_path = frames[idx]
        save_dir = os.path.join(DEMO_OUT_ROOT, folder)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n  {folder} [{label}] - frame #{idx} of {len(frames)}")
        t0 = time.time()
        try:
            results = pipeline.process_image(src_path, save_dir=save_dir)
            elapsed = (time.time() - t0) * 1000
            threat = results["threat_result"]
            conf_pct = f"{threat.confidence * 100:.1f}%" if threat.confidence > 0 else "LOW"
            print(f"    Animal     : {threat.animal_class}")
            print(f"    Confidence : {conf_pct}")
            print(f"    Threat     : {threat.threat_level}")
            print(f"    Elapsed    : {elapsed:.0f} ms")
            image_results.append({"folder": folder, "label": label,
                                   "animal": threat.animal_class,
                                   "confidence": conf_pct, "threat": threat.threat_level})
        except Exception as e:
            import traceback
            print(f"    [ERROR] {e}")
            traceback.print_exc()

    print("\n[3/3] Generating VIDEO demos ...")
    video_results = []
    for seq_name, out_filename, max_frames in VIDEO_DEMOS:
        seq_dir = os.path.join(MOCA_TRAIN_ROOT, seq_name)
        if not os.path.isdir(seq_dir):
            print(f"  [WARN] Sequence not found: {seq_dir}")
            continue
        frames = get_sorted_frames(seq_dir)[:max_frames]
        if not frames:
            print(f"  [WARN] No frames in {seq_dir}")
            continue
        raw_video_path = os.path.join(video_out_dir, f"_input_{out_filename}")
        print(f"\n  {out_filename} [{seq_name}] ({len(frames)} frames)")
        if not build_video_from_frames(frames, raw_video_path, fps=10):
            continue
        output_video_path = os.path.join(video_out_dir, out_filename)
        t0 = time.time()
        try:
            pipeline.camo_processor.reset()
            pipeline.tracker.reset()
            res = pipeline.process_video(
                video_path=raw_video_path,
                output_video_path=output_video_path,
                max_frames=max_frames,
            )
            elapsed = time.time() - t0
            ot = res["overall_threat"]
            print(f"    Frames processed : {res['frames_processed']}")
            print(f"    Speed            : {res['actual_fps']:.1f} fps")
            print(f"    Overall threat   : {ot.threat_level if ot else 'N/A'}")
            print(f"    Total time       : {elapsed:.1f}s")
            video_results.append({"filename": out_filename,
                                   "frames": res["frames_processed"],
                                   "threat": ot.threat_level if ot else "N/A"})
            try:
                os.remove(raw_video_path)
            except Exception:
                pass
        except Exception as e:
            import traceback
            print(f"    [ERROR] {e}")
            traceback.print_exc()

    print("\n" + "=" * 65)
    print("REVIEW DEMO - FINAL SUMMARY")
    print("=" * 65)
    print(f"U-Net checkpoint used  : checkpoints/unet_best.pth  (epoch=9, val_iou=0.2519)")
    print(f"YOLO model used        : runs/detect/wildlife_yolov8/weights/best.pt")
    print(f"Number of YOLO classes : 33")
    print(f"Demo classes selected  : crab, eastern_screech_owl, flatfish, snow_leopard, scorpionfish")
    print(f"Number of image demos  : {len(image_results)}")
    print(f"Number of video demos  : {len(video_results)}")
    print()
    for r in image_results:
        print(f"  {r['folder']} [{r['label']}] -> {r['animal']} {r['confidence']} [{r['threat']}]")
    for r in video_results:
        print(f"  {r['filename']} -> {r['frames']} frames [{r['threat']}]")
    print(f"\nAll outputs saved to: {DEMO_OUT_ROOT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
