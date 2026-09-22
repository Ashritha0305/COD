"""
scripts/detect_and_highlight.py
================================
Camouflaged Animal Detection & Visual Highlighting Tool

PURPOSE
-------
This script processes any input video (or webcam feed) and clearly highlights
camouflaged animals so a human viewer can immediately identify them.

Unlike run_pipeline.py (which subtly enhances frames for YOLOv8), this script
produces BOLD, CLEAR visual overlays on top of the original frame:

  - Semi-transparent filled overlay (colour-coded) over the animal body
  - Thick contour outline around the detected animal
  - Bounding box with "CAMOUFLAGED ANIMAL" label
  - Confidence score (max probability in detected region)
  - Pulsing contour animation to draw the eye to the detection

OUTPUT
------
  - Live side-by-side display:  [Original | Detected + Highlighted]
  - Saved MP4 video:  outputs/highlighted/<source_name>_highlighted.mp4

The model pipeline (U-Net + optical flow + fusion) is exactly the same as the
rest of the project.  Only the VISUALISATION layer is different here.

USAGE
-----
  # Any video file:
  python scripts/detect_and_highlight.py --source dataset/CamoVid60K/arctic_fox_1.webm

  # Save highlighted output video:
  python scripts/detect_and_highlight.py --source path/to/video.mp4 --save

  # Webcam:
  python scripts/detect_and_highlight.py --source 0 --save

  # Custom threshold (lower = more sensitive, more false positives):
  python scripts/detect_and_highlight.py --source video.mp4 --threshold 0.35

  # No display (headless / server), save only:
  python scripts/detect_and_highlight.py --source video.mp4 --save --no-display

  # Limit frames for quick testing:
  python scripts/detect_and_highlight.py --source video.mp4 --max-frames 60

HIGHLIGHT COLOURS (--color flag)
---------------------------------
  green  (default) : Classic detection green, works well on most backgrounds
  cyan             : Softer, good against green/leaf backgrounds
  red              : Maximum contrast, very prominent
  yellow           : High visibility, good for dark scenes
"""

import os
import sys
import argparse
import time
import math

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from modules.optical_flow import FarnebackOpticalFlow
from modules.fusion       import FrameFusion

# ---------------------------------------------------------------------------
# Compatible UNet (matches saved checkpoint architecture)
# ---------------------------------------------------------------------------
# The current unet.py uses ASPP at the bottleneck, but all saved checkpoints
# (unet_best.pth, unet_last.pth) were trained with the older architecture:
#   encoder -> DoubleConv bottleneck -> CBAM -> decoder
# We define that architecture here so we can load the checkpoint correctly.

import torch
import torch.nn as nn
import torch.nn.functional as _F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.conv = _DoubleConv(in_ch, out_ch, dropout_p)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s = self.conv(x)
        return s, self.pool(s)


class _DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv     = _DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = _F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class _ChanAttn(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        m = max(1, c // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(c, m, 1, bias=False), nn.ReLU(inplace=True),
                                  nn.Conv2d(m, c, 1, bias=False))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return x * self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x)))


class _SpatAttn(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=(k - 1) // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        a = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * self.sig(self.conv(a))


class _CBAM(nn.Module):
    def __init__(self, c, r=8, k=7):
        super().__init__()
        self.channel_attn = _ChanAttn(c, r)
        self.spatial_attn = _SpatAttn(k)

    def forward(self, x):
        return self.spatial_attn(self.channel_attn(x))


class _CompatUNet(nn.Module):
    """
    U-Net matching the architecture saved in unet_best.pth / unet_last.pth:
      encoder -> DoubleConv bottleneck -> CBAM -> decoder
    (No ASPP — that was added to unet.py after the checkpoints were saved.)
    """
    def __init__(self, f=config.UNET_INIT_FEATURES, dp=config.DROPOUT_RATE):
        super().__init__()
        self.enc1       = _EncBlock(3, f,      dp)
        self.enc2       = _EncBlock(f, f*2,    dp)
        self.enc3       = _EncBlock(f*2, f*4,  dp)
        self.enc4       = _EncBlock(f*4, f*8,  dp)
        self.bottleneck = _DoubleConv(f*8, f*16)
        self.cbam       = _CBAM(f*16)
        self.dec4       = _DecBlock(f*16, f*8)
        self.dec3       = _DecBlock(f*8,  f*4)
        self.dec2       = _DecBlock(f*4,  f*2)
        self.dec1       = _DecBlock(f*2,  f)
        self.output_conv = nn.Conv2d(f, 1, 1)

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        x = self.cbam(self.bottleneck(x))
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.output_conv(x)


# ---------------------------------------------------------------------------
# Lightweight processor (replaces CamouflageProcessor for this script)
# ---------------------------------------------------------------------------

class _VideoProcessor:
    """
    Minimal video-mode processor using the compatible UNet.
    Runs: U-Net seg -> optical flow -> fusion (same pipeline as CamouflageProcessor).
    """

    def __init__(self, checkpoint_path, device):
        self._device = device
        self._model  = _CompatUNet().to(device)
        self._model.eval()

        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state", ckpt)
            self._model.load_state_dict(state)
            iou_str = f"{ckpt['val_iou']:.4f}" if "val_iou" in ckpt else "?"
            print(f"[Processor] Loaded checkpoint "
                  f"(epoch={ckpt.get('epoch','?')}, val_iou={iou_str}): {checkpoint_path}")
        else:
            print("[Processor] WARNING: no checkpoint — using random weights.")

        self._mean  = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self._std   = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self._flow  = FarnebackOpticalFlow()
        self._fuser = FrameFusion()
        print(f"[Processor] Device: {device}")

    def process(self, frame_bgr: np.ndarray):
        orig_h, orig_w = frame_bgr.shape[:2]

        # -- U-Net seg --
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_r = cv2.resize(rgb, (config.INPUT_WIDTH, config.INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb_r.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
        t = (t - self._mean) / self._std
        with torch.no_grad():
            logits = self._model(t)
        seg = torch.sigmoid(logits).squeeze().float().cpu().numpy()
        seg = cv2.resize(seg.astype(np.float64), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        # -- Optical flow --
        flow = self._flow.process_frame(frame_bgr) if config.RUN_FLOW_EVERY_FRAME else None

        # -- Fusion (enhanced frame, not used for highlighting but kept for interface) --
        enhanced = self._fuser.fuse(frame_bgr, seg, flow)

        return seg, enhanced

    def reset(self):
        self._flow.reset()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Colour palettes in BGR format
COLOUR_PALETTES = {
    "green":  {"fill": (0, 220, 80),   "outline": (0, 255, 60),  "box": (0, 200, 50),  "text": (255, 255, 255)},
    "cyan":   {"fill": (220, 210, 0),  "outline": (255, 240, 0), "box": (200, 200, 0), "text": (0, 0, 0)},
    "red":    {"fill": (30, 30, 220),  "outline": (0, 0, 255),   "box": (0, 0, 200),   "text": (255, 255, 255)},
    "yellow": {"fill": (0, 200, 220),  "outline": (0, 220, 255), "box": (0, 190, 210), "text": (0, 0, 0)},
}

MIN_CONTOUR_AREA   = 400    # pixels^2 -- smaller regions are ignored as noise
MORPH_KERNEL_SIZE  = 7      # morphological open/close kernel size
OVERLAY_ALPHA      = 0.40   # fill transparency (0=invisible, 1=solid)
OUTLINE_THICKNESS  = 3      # base contour outline thickness
PULSE_AMPLITUDE    = 2      # +/- pixels for the pulsing contour animation
PULSE_PERIOD       = 30     # frames per pulse cycle
BOX_THICKNESS      = 2
FONT               = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE        = 0.65
LABEL_THICK        = 2
STATUS_SCALE       = 0.55
STATUS_THICK       = 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Camouflaged animal detection + clear visual highlighting.\n"
            "Outputs a side-by-side video: [Original | Detected + Highlighted]."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            "Input source (default: '0' = webcam).\n"
            "Can be a video file path (.mp4, .avi, .webm, .mkv, etc.) or\n"
            "a camera device index (0, 1, ...)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=config.BEST_MODEL_PATH,
        help=f"U-Net checkpoint path (default: {config.BEST_MODEL_PATH}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.SEG_THRESHOLD,
        help=(
            f"Detection probability threshold in [0, 1] (default: {config.SEG_THRESHOLD}).\n"
            "Lower = more sensitive (may have more false positives).\n"
            "Higher = more conservative (may miss faint detections)."
        ),
    )
    parser.add_argument(
        "--color",
        type=str,
        default="green",
        choices=list(COLOUR_PALETTES.keys()),
        help="Highlight colour (default: green).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the highlighted output video to outputs/highlighted/.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the live OpenCV display window (useful on headless systems).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (useful for quick testing).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=MIN_CONTOUR_AREA,
        help=(
            f"Minimum contour area in pixels^2 (default: {MIN_CONTOUR_AREA}). "
            "Smaller detections are suppressed as noise."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Morphological cleanup helpers
# ---------------------------------------------------------------------------

def _get_morph_kernel():
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )


def _clean_mask(binary_mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply morphological open -> close to remove isolated noise pixels
    and fill small holes in the detection region.

    Parameters
    ----------
    binary_mask : (H, W) uint8, values 0 or 255
    kernel      : morphological structuring element

    Returns
    -------
    cleaned : (H, W) uint8, values 0 or 255
    """
    # Opening removes tiny isolated blobs (false positives)
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)
    # Closing fills small holes inside the animal region
    closed = cv2.morphologyEx(opened,      cv2.MORPH_CLOSE, kernel)
    return closed


# ---------------------------------------------------------------------------
# Per-frame highlighting
# ---------------------------------------------------------------------------

def _draw_highlights(
    frame_bgr:   np.ndarray,
    seg_mask:    np.ndarray,
    frame_idx:   int,
    palette:     dict,
    threshold:   float,
    min_area:    int,
    morph_kernel: np.ndarray,
) -> tuple:
    """
    Draw bold, human-visible highlights on a copy of frame_bgr.

    Returns
    -------
    highlighted  : (H, W, 3) uint8 -- annotated frame
    n_detections : int -- number of valid contours drawn
    max_conf     : float -- max confidence score across all detections (0 if none)
    """
    H, W = frame_bgr.shape[:2]
    fill_colour    = palette["fill"]
    outline_colour = palette["outline"]
    box_colour     = palette["box"]
    text_colour    = palette["text"]

    # ---- Threshold the probability map ----
    binary = (seg_mask >= threshold).astype(np.uint8) * 255   # (H, W) uint8

    # ---- Morphological cleanup ----
    binary = _clean_mask(binary, morph_kernel)

    # ---- Find contours ----
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by minimum area
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    highlighted = frame_bgr.copy()

    if not valid_contours:
        return highlighted, 0, 0.0

    # ---- Pulsing outline thickness ----
    # Sinusoidal pulse so the outline "breathes" to attract the viewer's eye
    pulse     = math.sin(2 * math.pi * frame_idx / PULSE_PERIOD)
    thickness = int(OUTLINE_THICKNESS + PULSE_AMPLITUDE * (pulse + 1) / 2)  # [3..5]

    # ---- Semi-transparent filled overlay ----
    fill_layer = np.zeros_like(frame_bgr, dtype=np.uint8)
    cv2.drawContours(fill_layer, valid_contours, -1, fill_colour, cv2.FILLED)
    cv2.addWeighted(fill_layer, OVERLAY_ALPHA, highlighted, 1.0, 0, highlighted)

    # ---- Thick contour outlines ----
    cv2.drawContours(highlighted, valid_contours, -1, outline_colour, thickness)

    # ---- Per-contour bounding box and label ----
    max_conf = 0.0
    for cnt in valid_contours:
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Clamp box to frame
        x  = max(0, x)
        y  = max(0, y)
        bw = min(bw, W - x)
        bh = min(bh, H - y)

        # Confidence = max seg_mask probability inside this contour
        region_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(region_mask, [cnt], -1, 255, cv2.FILLED)
        roi_probs = seg_mask[region_mask > 0]
        conf      = float(roi_probs.max()) if roi_probs.size > 0 else 0.0
        max_conf  = max(max_conf, conf)

        # ---- Bounding box rectangle ----
        cv2.rectangle(highlighted, (x, y), (x + bw, y + bh), box_colour, BOX_THICKNESS)

        # ---- Label text + background bar ----
        label_text = f"CAMOUFLAGED ANIMAL  {conf * 100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(
            label_text, FONT, LABEL_SCALE, LABEL_THICK
        )
        bar_y1 = max(0,       y - th - baseline - 8)
        bar_y2 = max(th + 4,  y)
        bar_x2 = min(W,       x + tw + 8)

        # Semi-transparent label background
        label_bg = highlighted.copy()
        cv2.rectangle(label_bg, (x, bar_y1), (bar_x2, bar_y2), box_colour, -1)
        cv2.addWeighted(label_bg, 0.75, highlighted, 0.25, 0, highlighted)

        # Label text
        cv2.putText(
            highlighted, label_text,
            (x + 4, max(th + 2, y - baseline - 4)),
            FONT, LABEL_SCALE, text_colour, LABEL_THICK, cv2.LINE_AA,
        )

    return highlighted, len(valid_contours), max_conf


# ---------------------------------------------------------------------------
# Status bar overlay (bottom of frame)
# ---------------------------------------------------------------------------

def _draw_status(
    frame:     np.ndarray,
    frame_idx: int,
    fps:       float,
    n_det:     int,
    conf:      float,
    threshold: float,
) -> None:
    """Draw a semi-transparent status bar at the bottom of frame (in-place)."""
    H, W  = frame.shape[:2]
    bar_h = 36

    bar = frame.copy()
    cv2.rectangle(bar, (0, H - bar_h), (W, H), (20, 20, 20), -1)
    cv2.addWeighted(bar, 0.70, frame, 0.30, 0, frame)

    det_text = (
        f"Frame {frame_idx}  |  FPS {fps:.1f}  |  "
        f"Detections: {n_det}  |  "
        f"Conf: {conf * 100:.1f}%  |  "
        f"Threshold: {threshold:.2f}"
    )
    cv2.putText(
        frame, det_text,
        (8, H - 10),
        FONT, STATUS_SCALE, (200, 200, 200), STATUS_THICK, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Panel header (top of each panel)
# ---------------------------------------------------------------------------

def _add_panel_header(canvas: np.ndarray, text: str, x_offset: int, panel_w: int) -> None:
    """Draw a labelled header bar at the top of a panel."""
    bar_h = 34
    x1    = min(x_offset + panel_w, canvas.shape[1])

    roi = canvas[0:bar_h, x_offset:x1]
    dark = np.zeros_like(roi)
    cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)
    canvas[0:bar_h, x_offset:x1] = roi

    (tw, _), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    tx = x_offset + (panel_w - tw) // 2
    cv2.putText(canvas, text, (tx, bar_h - 9), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args        = parse_args()
    palette     = COLOUR_PALETTES[args.color]
    morph_kern  = _get_morph_kernel()

    # ---- Load checkpoint ----
    checkpoint = None
    if os.path.isfile(str(args.checkpoint)):
        checkpoint = args.checkpoint
    else:
        print(f"[WARN] Checkpoint not found: {args.checkpoint}")
        print("[WARN] Running with random U-Net weights -- detections will be noise.")
        print("[WARN] Train first: python scripts/train_unet.py\n")

    # ---- Initialise processor with compatible checkpoint loader ----
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = _VideoProcessor(checkpoint_path=checkpoint, device=device)

    # ---- Open source ----
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    fps_native   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n[INFO] Source          : {source}")
    print(f"[INFO] Resolution      : {orig_w} x {orig_h}")
    print(f"[INFO] Native FPS      : {fps_native:.1f}")
    print(f"[INFO] Total frames    : {total_frames if total_frames > 0 else 'unknown (live)'}")
    print(f"[INFO] Threshold       : {args.threshold}")
    print(f"[INFO] Highlight color : {args.color}")
    print(f"[INFO] Min area        : {args.min_area} px^2")

    # ---- Output video writer ----
    writer   = None
    out_path = None
    if args.save:
        out_dir = os.path.join(config.OUTPUTS_DIR, "highlighted")
        os.makedirs(out_dir, exist_ok=True)

        stem     = (
            os.path.splitext(os.path.basename(str(source)))[0]
            if isinstance(source, str)
            else f"cam{source}"
        )
        out_path = os.path.join(out_dir, f"{stem}_highlighted.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        # Side-by-side width: orig_w * 2  (no divider in saved file)
        writer   = cv2.VideoWriter(out_path, fourcc, fps_native, (orig_w * 2, orig_h))
        print(f"[INFO] Saving output   : {out_path}")

    # ---- Main loop ----
    frame_idx = 0
    fps_ema   = 0.0
    t_start   = time.time()

    print("\n[INFO] Processing started. Press 'q' to quit.\n")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("[INFO] End of video / stream finished.")
            break

        t0 = time.time()

        # =============================================================
        # Step 1 -- _VideoProcessor: U-Net seg + optical flow
        # =============================================================
        seg_mask, _enhanced = processor.process(frame_bgr)

        # =============================================================
        # Step 2 -- Draw bold visual highlights for human viewing
        # =============================================================
        highlighted, n_det, max_conf = _draw_highlights(
            frame_bgr    = frame_bgr,
            seg_mask     = seg_mask,
            frame_idx    = frame_idx,
            palette      = palette,
            threshold    = args.threshold,
            min_area     = args.min_area,
            morph_kernel = morph_kern,
        )

        # FPS tracking
        elapsed = time.time() - t0
        fps_now = 1.0 / max(elapsed, 1e-6)
        fps_ema = 0.9 * fps_ema + 0.1 * fps_now if frame_idx > 0 else fps_now

        # =============================================================
        # Step 3 -- Build side-by-side display frame
        # =============================================================
        left  = frame_bgr.copy()
        right = highlighted   # already a copy

        _draw_status(left,  frame_idx, fps_ema, n_det, max_conf, args.threshold)
        _draw_status(right, frame_idx, fps_ema, n_det, max_conf, args.threshold)

        # 3-pixel grey divider line
        divider      = np.full((orig_h, 3, 3), 50, dtype=np.uint8)
        side_by_side = np.concatenate([left, divider, right], axis=1)

        # Panel header labels
        _add_panel_header(side_by_side, "ORIGINAL",               0,           orig_w)
        _add_panel_header(side_by_side, "DETECTED & HIGHLIGHTED", orig_w + 3,  orig_w)

        # =============================================================
        # Step 4 -- Display and / or save
        # =============================================================
        if not args.no_display:
            cv2.imshow(
                "Camouflage Detection  |  Original  vs  Highlighted  ('q' to quit)",
                side_by_side,
            )
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] User quit.")
                break

        if writer is not None:
            # Saved video: clean side-by-side without divider line artifact
            writer.write(np.concatenate([left, right], axis=1))

        # Console status
        if frame_idx % 5 == 0:
            pct_str = (
                f" ({100.0 * frame_idx / total_frames:.1f}%)"
                if total_frames > 0 else ""
            )
            print(
                f"  Frame {frame_idx:5d}{pct_str}  |  "
                f"FPS={fps_ema:.1f}  |  "
                f"detections={n_det}  |  "
                f"seg_max={seg_mask.max():.3f}  |  "
                f"conf={max_conf * 100:.1f}%"
            )

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            print(f"[INFO] Reached max-frames={args.max_frames}.")
            break

    # ---- Cleanup ----
    cap.release()
    if writer is not None:
        writer.release()
        print(f"\n[DONE] Highlighted video saved -> {out_path}")
    if not args.no_display:
        cv2.destroyAllWindows()

    total_time = time.time() - t_start
    avg_fps    = frame_idx / max(total_time, 1e-6)
    print(f"[DONE] Processed {frame_idx} frames in {total_time:.1f}s  (avg {avg_fps:.1f} fps)")


if __name__ == "__main__":
    main()
