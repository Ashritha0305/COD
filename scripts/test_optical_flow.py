"""
scripts/test_optical_flow.py
=============================
RIGHT BRANCH — Standalone Optical Flow Test on Wildlife Video

This script runs ONLY the Farneback optical flow module on a wildlife video.
No U-Net, no dataset, no training required — just a video file.

This directly implements the right branch of the Member 1 task diagram:

    Optical Flow
         │
    No dataset needed
         │
    Test wildlife video
         │
    Outputs saved to outputs/optical_flow/

WHAT THIS SCRIPT DOES
---------------------
1. Opens a wildlife video file (or live camera).
2. Computes Farneback dense optical flow between consecutive frame pairs.
3. Saves TWO outputs per frame to outputs/optical_flow/:
   - flow_magnitude_XXXXX.png : grayscale magnitude map (brighter = more motion)
   - flow_rgb_XXXXX.png       : HSV-coloured flow (hue = direction, value = magnitude)
4. Prints per-frame statistics (min/max motion, mean motion).
5. Displays a live preview window (press 'q' to quit, press 's' to save current frame).

USAGE
-----
    # On one of the demo webm videos (no dataset needed):
    python scripts/test_optical_flow.py --source dataset/CamoVid60K/arabian_horn_viper.webm

    # On live webcam:
    python scripts/test_optical_flow.py --source 0

    # Save all outputs:
    python scripts/test_optical_flow.py --source dataset/CamoVid60K/lioness.webm --save-all

    # No display (headless):
    python scripts/test_optical_flow.py --source dataset/CamoVid60K/goat_0.webm --no-display --save-all --max-frames 30
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from modules.optical_flow import FarnebackOpticalFlow


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
FLOW_OUT = config.FLOW_OUT_DIR  # outputs/optical_flow/


def parse_args():
    parser = argparse.ArgumentParser(
        description="RIGHT BRANCH: Standalone optical flow test on wildlife video."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="dataset/CamoVid60K/arabian_horn_viper.webm",
        help="Video file path OR camera device index. Default: arabian_horn_viper.webm",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save flow magnitude and RGB visualisation for every frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not show live OpenCV window.",
    )
    return parser.parse_args()


def save_flow_outputs(
    frame_idx: int,
    original_bgr: np.ndarray,
    flow_magnitude: np.ndarray,
    flow_rgb: np.ndarray,
    out_dir: str,
) -> None:
    """Save magnitude map and RGB flow visualisation to outputs/optical_flow/."""
    os.makedirs(out_dir, exist_ok=True)

    # Grayscale magnitude map (0-255)
    mag_u8 = (np.clip(flow_magnitude, 0, 1) * 255).astype(np.uint8)
    # Apply jet colormap for better visibility
    mag_colored = cv2.applyColorMap(mag_u8, cv2.COLORMAP_JET)

    # Side-by-side: original | magnitude | flow_rgb
    flow_rgb_bgr = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)
    panel = np.concatenate([original_bgr, mag_colored, flow_rgb_bgr], axis=1)

    cv2.imwrite(
        os.path.join(out_dir, f"flow_{frame_idx:05d}.png"), panel
    )


def main():
    args = parse_args()

    # ---- Resolve source ----
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    os.makedirs(FLOW_OUT, exist_ok=True)

    # ---- Open video ----
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        print("        Provide a valid video file or camera index.")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"\n{'='*55}")
    print(f"  Optical Flow Test — RIGHT BRANCH (no dataset needed)")
    print(f"{'='*55}")
    print(f"  Source   : {source}")
    print(f"  FPS      : {fps:.1f}")
    print(f"  Frames   : {total if total > 0 else 'live'}")
    print(f"  Save all : {args.save_all}")
    print(f"  Flow out : {FLOW_OUT}")
    print(f"{'='*55}\n")

    # ---- Optical flow module ----
    of = FarnebackOpticalFlow()

    frame_idx = 0
    total_motion_sum = 0.0
    t_start = time.time()

    print("  [INFO] Press 'q' to quit, 's' to save current frame.\n")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("  [INFO] End of video.")
            break

        # ---- Compute optical flow ----
        flow_magnitude = of.process_frame(frame_bgr)

        if flow_magnitude is None:
            # First frame — no previous frame to compute flow from
            print(f"  Frame {frame_idx:5d} | First frame — no flow computed yet.")
            frame_idx += 1
            continue

        # ---- Statistics ----
        mean_motion = float(flow_magnitude.mean())
        max_motion  = float(flow_magnitude.max())
        total_motion_sum += mean_motion

        if frame_idx % 10 == 0:
            print(
                f"  Frame {frame_idx:5d} | "
                f"mean_motion={mean_motion:.4f}  "
                f"max_motion={max_motion:.4f}"
            )

        # ---- Save outputs ----
        if args.save_all:
            # of.prev_bgr is the frame stored BEFORE curr was processed
            prev_for_viz = of.prev_bgr if of.prev_bgr is not None else frame_bgr
            flow_rgb = of.get_flow_rgb(prev_for_viz, frame_bgr)
            save_flow_outputs(frame_idx, frame_bgr, flow_magnitude, flow_rgb, FLOW_OUT)

        # ---- Display ----
        if not args.no_display:
            mag_u8     = (np.clip(flow_magnitude, 0, 1) * 255).astype(np.uint8)
            mag_color  = cv2.applyColorMap(mag_u8, cv2.COLORMAP_JET)
            display    = np.concatenate([frame_bgr, mag_color], axis=1)
            cv2.putText(
                display,
                f"Frame={frame_idx}  mean_motion={mean_motion:.3f}  max={max_motion:.3f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.imshow("Optical Flow — Original | Magnitude", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("  [INFO] User quit.")
                break
            elif key == ord("s"):
                save_path = os.path.join(FLOW_OUT, f"manual_save_{frame_idx:05d}.png")
                cv2.imwrite(save_path, display)
                print(f"  [SAVED] {save_path}")

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            print(f"  [INFO] Reached max-frames={args.max_frames}.")
            break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    processed = max(frame_idx - 1, 1)
    avg_motion = total_motion_sum / processed

    print(f"\n{'='*55}")
    print(f"  Optical Flow Test Complete")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Avg mean motion  : {avg_motion:.4f}")
    print(f"  Time elapsed     : {elapsed:.1f}s")
    print(f"  Avg throughput   : {frame_idx / elapsed:.1f} fps")
    if args.save_all:
        print(f"  Outputs saved to : {FLOW_OUT}")
    print(f"{'='*55}\n")


def _get_prev_frame(curr_bgr: np.ndarray) -> np.ndarray:
    """
    Placeholder — returns a zeroed frame when prev is not stored externally.
    Only used for the get_flow_rgb() call when save_all is True.
    In practice the of module stores the previous frame internally.
    """
    return np.zeros_like(curr_bgr)


if __name__ == "__main__":
    main()
