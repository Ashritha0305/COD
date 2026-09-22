"""
scripts/run_pipeline.py
========================
Phase 7 — Member 1 -> Member 2 Interface Script

This script demonstrates how Member 1's CamouflageProcessor is called and
how the enhanced frame is passed to Member 2 (YOLOv8).

INTERFACE CONTRACT
------------------
Member 2's YOLOv8 module should import and use CamouflageProcessor like this:

    from modules.camouflage_processor import CamouflageProcessor

    processor = CamouflageProcessor(checkpoint_path="checkpoints/unet_best.pth")

    # For each frame from the camera/video:
    result = processor.process(frame_bgr)
    enhanced_frame = result.enhanced_frame   # ← feed this to YOLOv8

Member 2 does NOT need to know anything else about the internals.

WHAT THIS SCRIPT DOES
---------------------
1. Opens a video file, live camera, OR a single still image (see --mode).
2. For each frame, runs CamouflageProcessor.process().
3. (Member 2 stub) prints what YOLOv8 would receive.
4. Optionally saves individual outputs and/or a qualitative 4-panel comparison.

PROCESSING MODES
-----------------
  --mode video (DEFAULT)
    Full pipeline: U-Net + Optical Flow + Fusion with temporal smoothing.
    Use for: video files, live camera streams, MoCA video sequences.
    Source: video file path (e.g. clip.mp4) or camera device index (0).

  --mode image
    U-Net only: no optical flow, no temporal EMA smoothing.
    Use for: single still images, offline per-image inference.
    Source: path to an image file (.jpg, .png, .bmp, etc.).
    The enhanced image is saved to outputs/enhanced/ and displayed.

USAGE
-----
    # VIDEO mode -- video file with trained checkpoint:
    python scripts/run_pipeline.py --mode video --source path/to/video.mp4

    # VIDEO mode -- with all outputs saved:
    python scripts/run_pipeline.py --mode video --source clip.mp4 --save-outputs --no-display

    # IMAGE mode -- single still image:
    python scripts/run_pipeline.py --mode image --source path/to/image.jpg

    # IMAGE mode -- save outputs:
    python scripts/run_pipeline.py --mode image --source image.jpg --save-outputs --no-display

    # IMAGE mode -- directory of images (processed in sorted order, no optical flow between any):
    python scripts/run_pipeline.py --mode image --source path/to/img_dir/ --save-outputs --no-display
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
from modules.camouflage_processor import CamouflageProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Member 1 pipeline demo: CamouflageProcessor -> YOLOv8 handoff"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="video",
        choices=["video", "image"],
        help=(
            "Processing mode (default: video).\n"
            "  video: full pipeline -- U-Net + Optical Flow + temporal EMA smoothing.\n"
            "         Source = video file path or camera device index.\n"
            "  image: U-Net only -- no optical flow, no temporal state.\n"
            "         Source = single image file path or directory of images."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            "Input source (default: '0' = webcam).\n"
            "  video mode: video file path OR camera device index.\n"
            "  image mode: path to image file (.jpg/.png) OR directory of images."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=config.BEST_MODEL_PATH,
        help=f"Path to trained U-Net checkpoint (default: {config.BEST_MODEL_PATH}).",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save 4-panel debug visualisation for each frame to outputs/debug/.",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help=(
            "Save individual outputs to their designated folders:\n"
            "  Enhanced frame  -> outputs/enhanced/\n"
            "  Seg mask        -> outputs/masks/\n"
            "  Flow map        -> outputs/optical_flow/ (video mode only)"
        ),
    )
    parser.add_argument(
        "--save-qualitative",
        action="store_true",
        help=(
            "Save a 4-panel comparison panel for each frame to outputs/qualitative/.\n"
            "Panels: [Original | U-Net Mask | Optical Flow | Enhanced]\n"
            "Intended for project reports and demos."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (useful for testing).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not show live OpenCV window (useful on headless systems).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Qualitative panel helper
# ---------------------------------------------------------------------------

def _make_qualitative_panel(
    original_bgr: np.ndarray,
    seg_mask:     np.ndarray,
    flow_map,                           # may be None (first frame)
    enhanced_bgr: np.ndarray,
    frame_idx:    int,
) -> np.ndarray:
    """
    Build a labelled 4-panel side-by-side comparison image:
        [Original | U-Net Seg Mask | Optical Flow | Enhanced (->YOLOv8)]

    The segmentation mask is shown as a VIRIDIS heatmap so soft probability
    values (not just binary) are visible.  Flow is shown as a JET heatmap.
    """
    H, W = original_bgr.shape[:2]

    def _prob_to_bgr_heatmap(prob_map: np.ndarray, colormap: int) -> np.ndarray:
        """Convert a (H, W) float32 [0,1] map to a coloured BGR image."""
        u8 = (np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8)
        return cv2.applyColorMap(u8, colormap)

    seg_vis  = _prob_to_bgr_heatmap(seg_mask, cv2.COLORMAP_VIRIDIS)
    flow_vis = (
        _prob_to_bgr_heatmap(flow_map, cv2.COLORMAP_JET)
        if flow_map is not None
        else np.zeros((H, W, 3), dtype=np.uint8)
    )

    LABEL_COLOUR = (255, 255, 255)
    LABEL_FONT   = cv2.FONT_HERSHEY_SIMPLEX
    LABEL_SCALE  = min(H, W) / 480.0   # scale with image size
    LABEL_THICK  = max(1, int(LABEL_SCALE * 2))

    def _add_label(img: np.ndarray, text: str, sub: str = "") -> np.ndarray:
        out = img.copy()
        # semi-transparent dark bar at top for readability
        bar_h = max(30, int(H * 0.07))
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (W, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
        cv2.putText(out, text, (6, bar_h - 8),
                    LABEL_FONT, LABEL_SCALE, LABEL_COLOUR, LABEL_THICK, cv2.LINE_AA)
        if sub:
            cv2.putText(out, sub, (6, bar_h + int(18 * LABEL_SCALE)),
                        LABEL_FONT, LABEL_SCALE * 0.65, (200, 200, 200),
                        max(1, LABEL_THICK - 1), cv2.LINE_AA)
        return out

    panels = [
        _add_label(original_bgr, "Original",        f"Frame {frame_idx}"),
        _add_label(seg_vis,      "U-Net Seg Mask",  "Viridis: low->high prob"),
        _add_label(flow_vis,     "Optical Flow",    "Jet: low->high motion" if flow_map is not None else "N/A (first frame)"),
        _add_label(enhanced_bgr, "Enhanced ->YOLOv8","Member 1 output"),
    ]

    return np.concatenate(panels, axis=1)   # (H, 4*W, 3)


# ---------------------------------------------------------------------------
# Image mode handler
# ---------------------------------------------------------------------------

def _run_image_mode(args, processor) -> None:
    """
    IMAGE MODE: run the processor on a single image file or a directory of images.

    No optical flow is computed (processor.process is called with mode="image").
    Each image is treated as independent -- no temporal state carries over.
    """
    import glob

    source = args.source
    enhanced_dir = config.ENHANCED_OUT_DIR
    masks_dir    = config.MASKS_OUT_DIR
    qual_dir     = os.path.join(config.OUTPUTS_DIR, "qualitative")

    # Build list of image paths
    img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    if os.path.isdir(source):
        img_paths = sorted([
            p for p in glob.glob(os.path.join(source, "*"))
            if os.path.splitext(p)[1].lower() in img_extensions
        ])
        if not img_paths:
            print(f"[ERROR] No image files found in directory: {source}")
            sys.exit(1)
        print(f"[INFO] Found {len(img_paths)} image(s) in: {source}")
    elif os.path.isfile(source) and source.lower().endswith(img_extensions):
        img_paths = [source]
    else:
        print(f"[ERROR] IMAGE mode: source must be an image file or a directory of images.")
        print(f"        Got: {source}")
        sys.exit(1)

    if args.save_outputs:
        os.makedirs(enhanced_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        print(f"[INFO] Enhanced frames -> {enhanced_dir}")
        print(f"[INFO] Seg masks       -> {masks_dir}")
    if args.save_qualitative:
        os.makedirs(qual_dir, exist_ok=True)
        print(f"[INFO] Qualitative panels -> {qual_dir}")

    print(f"\n[INFO] Starting IMAGE mode pipeline on {len(img_paths)} image(s).\n")

    latencies = []

    for idx, img_path in enumerate(img_paths):
        if args.max_frames is not None and idx >= args.max_frames:
            print(f"[INFO] Reached max-frames={args.max_frames}.")
            break

        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None:
            print(f"[WARN] Could not read image: {img_path}  -- skipping.")
            continue

        t0     = time.time()
        # mode="image" forces no optical flow and resets temporal state
        result = processor.process(frame_bgr, mode="image")
        latency_ms = (time.time() - t0) * 1000.0
        latencies.append(latency_ms)

        stem = os.path.splitext(os.path.basename(img_path))[0]
        print(
            f"  [{idx+1:4d}/{len(img_paths)}]  {stem}  "
            f"latency={latency_ms:.1f}ms  "
            f"seg_max={result.seg_mask.max():.3f}"
        )

        if args.save_outputs:
            # Save enhanced frame (what YOLOv8 receives)
            cv2.imwrite(
                os.path.join(enhanced_dir, f"enhanced_{stem}.jpg"),
                result.enhanced_frame
            )
            # Save segmentation mask as Viridis heatmap
            mask_u8   = (np.clip(result.seg_mask, 0, 1) * 255).astype(np.uint8)
            mask_heat = cv2.applyColorMap(mask_u8, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(
                os.path.join(masks_dir, f"mask_{stem}.png"),
                mask_heat
            )

        if args.save_qualitative:
            panel = _make_qualitative_panel(
                original_bgr=frame_bgr,
                seg_mask=result.seg_mask,
                flow_map=None,      # always None in image mode
                enhanced_bgr=result.enhanced_frame,
                frame_idx=idx,
            )
            cv2.imwrite(
                os.path.join(qual_dir, f"comparison_{stem}.jpg"),
                panel,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

        if not args.no_display:
            display = np.concatenate([frame_bgr, result.enhanced_frame], axis=1)
            cv2.putText(
                display, f"IMAGE mode | {stem}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.imshow("Member1 [IMAGE MODE]: Original | Enhanced (-YOLOv8)", display)
            key = cv2.waitKey(0) & 0xFF   # wait for keypress in image mode
            if key == ord("q"):
                print("[INFO] User quit.")
                break

    if not args.no_display:
        cv2.destroyAllWindows()

    if latencies:
        avg_ms = sum(latencies) / len(latencies)
        print(f"\n[DONE] IMAGE MODE -- {len(latencies)} image(s) processed")
        print(f"       Avg latency : {avg_ms:.1f} ms/image")
        print(f"       Avg FPS     : {1000.0/max(avg_ms, 1.0):.1f}")
    if args.save_outputs:
        print(f"[DONE] Enhanced frames -> {enhanced_dir}")
        print(f"[DONE] Seg masks       -> {masks_dir}")
    if args.save_qualitative:
        print(f"[DONE] Qualitative panels -> {qual_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    mode = args.mode   # "video" or "image"

    # ---- Load checkpoint ----
    checkpoint = None
    if os.path.isfile(str(args.checkpoint)):
        checkpoint = args.checkpoint
    else:
        print(f"[INFO] Checkpoint not found: {args.checkpoint}")
        print("[INFO] Running with random U-Net weights (pipeline shape demo only).")
        print("[INFO] Train first: python scripts/train_unet.py\n")

    # ---- Initialise processor (mode sets default; per-call overrides in video loop) ----
    processor = CamouflageProcessor(checkpoint_path=checkpoint, mode=mode)

    print(f"\n[INFO] Processing mode : {mode.upper()}")
    print(f"[INFO] Source          : {args.source}")

    # ------------------------------------------------------------------
    # IMAGE MODE -- single image file OR directory of images
    # ------------------------------------------------------------------
    if mode == "image":
        _run_image_mode(args, processor)
        return

    # ------------------------------------------------------------------
    # VIDEO MODE -- video file or live camera
    # ------------------------------------------------------------------

    # ---- Resolve source ----
    source = args.source
    try:
        source = int(source)   # camera index
    except ValueError:
        pass   # keep as string (file path)

    # ---- Open video / camera ----
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        sys.exit(1)

    fps_target   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] FPS      : {fps_target:.1f}")
    print(f"[INFO] Frames   : {total_frames if total_frames > 0 else 'unknown (live stream)'}")

    # ---- Output directories ----
    debug_dir      = os.path.join(config.OUTPUTS_DIR, "debug")
    enhanced_dir   = config.ENHANCED_OUT_DIR        # outputs/enhanced/
    masks_dir      = config.MASKS_OUT_DIR            # outputs/masks/
    flow_dir       = config.FLOW_OUT_DIR             # outputs/optical_flow/
    qual_dir       = os.path.join(config.OUTPUTS_DIR, "qualitative")

    if args.save_debug:
        os.makedirs(debug_dir, exist_ok=True)
    if args.save_outputs:
        os.makedirs(enhanced_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(flow_dir, exist_ok=True)
        print(f"[INFO] Enhanced frames -> {enhanced_dir}")
        print(f"[INFO] Seg masks       -> {masks_dir}")
        print(f"[INFO] Flow maps       -> {flow_dir}")
    if args.save_qualitative:
        os.makedirs(qual_dir, exist_ok=True)
        print(f"[INFO] Qualitative panels -> {qual_dir}")

    # ---- Main loop ----
    frame_idx = 0
    t_start   = time.time()
    fps_ema   = 0.0

    print("\n[INFO] Starting pipeline. Press 'q' to quit.\n")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("[INFO] End of video.")
            break

        t0 = time.time()

        # =====================================================
        # MEMBER 1 — CamouflageProcessor
        # =====================================================
        result         = processor.process(frame_bgr)
        enhanced_frame = result.enhanced_frame   # ← this goes to YOLOv8
        # =====================================================

        # ---- FPS measurement ----
        elapsed = time.time() - t0
        fps_now = 1.0 / max(elapsed, 1e-6)
        fps_ema = 0.9 * fps_ema + 0.1 * fps_now if frame_idx > 0 else fps_now

        # ---- Console status ----
        if frame_idx % 5 == 0:
            print(
                f"  Frame {frame_idx:5d} | "
                f"pipeline FPS={fps_ema:.1f} | "
                f"seg_max={result.seg_mask.max():.3f} | "
                f"attn_max={result.attention_map.max():.3f} | "
                f"flow={'yes' if result.flow_map is not None else 'n/a'}"
            )

        # ---- Save debug visualisation (processor's own 4-panel) ----
        if args.save_debug:
            save_path = os.path.join(debug_dir, f"frame_{frame_idx:05d}.jpg")
            processor.visualise(frame_bgr, result, save_path=save_path)

        # ---- Save individual outputs ----
        if args.save_outputs:
            # Enhanced frame (what YOLOv8 receives)
            cv2.imwrite(
                os.path.join(enhanced_dir, f"enhanced_{frame_idx:05d}.jpg"),
                result.enhanced_frame
            )
            # Segmentation mask — save as VIRIDIS heatmap for visibility
            mask_u8   = (np.clip(result.seg_mask, 0, 1) * 255).astype(np.uint8)
            mask_heat = cv2.applyColorMap(mask_u8, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(
                os.path.join(masks_dir, f"mask_{frame_idx:05d}.png"),
                mask_heat
            )
            # Optical flow map (if available)
            if result.flow_map is not None:
                flow_u8   = (np.clip(result.flow_map, 0, 1) * 255).astype(np.uint8)
                flow_heat = cv2.applyColorMap(flow_u8, cv2.COLORMAP_JET)
                cv2.imwrite(
                    os.path.join(flow_dir, f"flow_{frame_idx:05d}.png"),
                    flow_heat
                )

        # ---- Save qualitative 4-panel comparison ----
        if args.save_qualitative:
            panel = _make_qualitative_panel(
                original_bgr=frame_bgr,
                seg_mask=result.seg_mask,
                flow_map=result.flow_map,
                enhanced_bgr=result.enhanced_frame,
                frame_idx=frame_idx,
            )
            cv2.imwrite(
                os.path.join(qual_dir, f"comparison_{frame_idx:05d}.jpg"),
                panel,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

        # ---- Display ----
        if not args.no_display:
            display = np.concatenate([frame_bgr, enhanced_frame], axis=1)
            cv2.putText(
                display,
                f"FPS={fps_ema:.1f}  Frame={frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.imshow("Member1: Original | Enhanced (->YOLOv8)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] User quit.")
                break

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            print(f"[INFO] Reached max-frames={args.max_frames}.")
            break

    # ---- Cleanup ----
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    total_time = time.time() - t_start
    avg_fps    = frame_idx / max(total_time, 1e-6)

    print(f"\n[DONE] Processed {frame_idx} frames in {total_time:.1f}s  (avg {avg_fps:.1f} fps)")
    if args.save_qualitative:
        print(f"[DONE] Qualitative comparisons saved -> {qual_dir}")
    if args.save_outputs:
        print(f"[DONE] Enhanced frames saved        -> {enhanced_dir}")
        print(f"[DONE] Seg masks saved              -> {masks_dir}")
        print(f"[DONE] Optical flow maps saved      -> {flow_dir}")


if __name__ == "__main__":
    main()
