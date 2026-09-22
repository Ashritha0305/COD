"""
modules/camouflage_processor.py
================================
Phase 6 -- CamouflageProcessor: Clean Interface for Member 2 (YOLOv8)

This is the single entry-point that Member 2 (YOLOv8) calls.
It encapsulates the complete Member 1 pipeline:

    Input frame (BGR, uint8)
        |
    U-Net camouflage segmentation  -> seg_map (H, W) float32 [0,1]
        |
    Farneback Optical Flow [VIDEO MODE ONLY] -> flow_map (H, W) float32 [0,1]
        |
    Fusion + Enhancement            -> enhanced frame (H, W, 3) uint8
        |
    Output to YOLOv8

INTERFACE CONTRACT (do NOT change -- Member 2 depends on this)
-----------------------------------------------------------
    from modules.camouflage_processor import CamouflageProcessor

    processor = CamouflageProcessor(checkpoint_path="checkpoints/unet_best.pth")

    # For each video frame (VIDEO MODE):
    result = processor.process(frame_bgr)
    enhanced_frame = result.enhanced_frame   # (H, W, 3) uint8 BGR -- feed to YOLOv8
    seg_mask       = result.seg_mask         # (H, W) float32 [0,1] -- optional debug
    flow_map       = result.flow_map         # (H, W) float32 [0,1] -- optional debug
    attention_map  = result.attention_map    # (H, W) float32 [0,1] -- optional debug

    # For a single still image (IMAGE MODE -- no optical flow):
    result = processor.process(frame_bgr, mode="image")

    # Visualise for debugging:
    processor.visualise(frame_bgr, result, save_path="outputs/debug.jpg")

    # Reset between video clips:
    processor.reset()

IMAGE MODE vs VIDEO MODE
-------------------------
IMAGE mode ("image"):
    - U-Net runs on the single frame as normal.
    - Optical flow is SKIPPED (no previous frame to compare against).
    - Temporal EMA smoothing in fusion is SKIPPED (no state to smooth over).
    - Internal temporal state is reset before processing.
    - Use this for: single still images, offline inference, evaluation scripts.
    - flow_map in ProcessorResult will be None.

VIDEO mode ("video", default):
    - Full pipeline: U-Net + Optical Flow + Fusion with EMA temporal smoothing.
    - Optical flow returns None for the FIRST frame (no previous frame).
    - Temporal EMA smoothing accumulates across frames.
    - Use this for: video files, camera streams, MoCA video sequences.

The mode can be set:
    a) At construction time: CamouflageProcessor(mode="image")
    b) Per-call: processor.process(frame, mode="video")
    The per-call value overrides the construction-time default.

DESIGN DECISIONS
----------------
- The U-Net always processes at config.INPUT_HEIGHT x INPUT_WIDTH.
  The seg_map is resized back to the original frame resolution before fusion.
- The optical flow runs on the original-resolution greyscale frames
  (not the U-Net resolution) to preserve motion fidelity.
- The output is always uint8 BGR -- identical to what OpenCV/YOLOv8 expects.
- If no checkpoint is loaded (checkpoint_path=None), U-Net runs in random-
  weight mode. This is only useful for testing the pipeline shape/flow.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from models.unet.unet        import UNet
from modules.optical_flow    import FarnebackOpticalFlow
from modules.fusion          import FrameFusion


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ProcessorResult:
    """
    Output from CamouflageProcessor.process().

    Attributes
    ----------
    enhanced_frame : np.ndarray, shape (H, W, 3), dtype uint8
        The enhanced BGR frame -- this is what YOLOv8 should receive.
    seg_mask : np.ndarray, shape (H, W), dtype float32
        U-Net probability map resized to original frame resolution.
        Values in [0, 1]. 1 = likely camouflaged.
    flow_map : np.ndarray or None, shape (H, W), dtype float32
        Normalised motion magnitude map.
        None when: first frame in VIDEO mode, or IMAGE mode.
    attention_map : np.ndarray, shape (H, W), dtype float32
        Combined attention map (weighted seg + flow). Values in [0, 1].
    mode : str
        Processing mode used for this frame ("image" or "video").
    """
    enhanced_frame: np.ndarray
    seg_mask:       np.ndarray
    flow_map:       Optional[np.ndarray]
    attention_map:  np.ndarray
    mode:           str = "video"


# ---------------------------------------------------------------------------
# Main processor class
# ---------------------------------------------------------------------------

class CamouflageProcessor:
    """
    End-to-end camouflage-aware preprocessing module for Member 1.

    Parameters
    ----------
    checkpoint_path : str or None
        Path to a trained U-Net checkpoint (.pth file saved by train_unet.py).
        If None, uses randomly initialised U-Net weights (pipeline testing only).
    device : str or None
        'cuda' or 'cpu'. Auto-detects if None.
    mode : str
        Default processing mode. One of "video" (full pipeline with optical flow
        and temporal EMA) or "image" (U-Net only, no flow, no temporal state).
        Can be overridden per-call in process(frame, mode=...).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        mode: str = config.PROCESSING_MODE,
    ) -> None:

        # ---- Validate mode ----
        if mode not in ("video", "image"):
            raise ValueError(
                f"mode must be 'video' or 'image', got: {mode!r}. "
                "Set config.PROCESSING_MODE or pass mode= to __init__."
            )
        self._default_mode = mode

        # ---- Device ----
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # ---- U-Net ----
        self._model = UNet().to(self._device)
        self._model.eval()

        if checkpoint_path is not None:
            checkpoint_path = str(checkpoint_path)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"U-Net checkpoint not found: {checkpoint_path}\n"
                    "Train the model first: python scripts/train_unet.py"
                )
            ckpt = torch.load(checkpoint_path, map_location=self._device)
            # Support both raw state_dict and our wrapped checkpoint format
            state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
            try:
                self._model.load_state_dict(state_dict)
                if "model_state" in ckpt:
                    val_iou = ckpt.get("val_iou", "?")
                    val_str = f"{val_iou:.4f}" if isinstance(val_iou, float) else str(val_iou)
                    print(
                        f"[CamouflageProcessor] Loaded checkpoint "
                        f"(epoch={ckpt.get('epoch','?')}, val_iou={val_str}): {checkpoint_path}"
                    )
                else:
                    print(f"[CamouflageProcessor] Loaded state_dict: {checkpoint_path}")
            except RuntimeError as e:
                # Detect the common case: old F=64 checkpoint loaded into new F=32 model
                err_str = str(e)
                if "size mismatch" in err_str or "Missing key" in err_str:
                    # Peek at the checkpoint's feature size to give a helpful message
                    first_conv_key = next(
                        (k for k in state_dict if "enc1" in k and "weight" in k
                         and "block.0" in k), None
                    )
                    ckpt_features = (
                        state_dict[first_conv_key].shape[0]
                        if first_conv_key else "unknown"
                    )
                    import config as _cfg
                    raise RuntimeError(
                        f"\n\n"
                        f"  ARCHITECTURE MISMATCH -- checkpoint is incompatible with the current model.\n"
                        f"\n"
                        f"  Checkpoint : {checkpoint_path}\n"
                        f"  Checkpoint features (F) : {ckpt_features}   (old architecture)\n"
                        f"  Current model features  : {_cfg.UNET_INIT_FEATURES}   (new v2 architecture)\n"
                        f"\n"
                        f"  The architecture was intentionally changed to fix overfitting:\n"
                        f"    - F: 64 -> 32  (7x fewer parameters)\n"
                        f"    - CBAM attention module added at bottleneck\n"
                        f"    - Spatial Dropout2d added in encoder\n"
                        f"\n"
                        f"  The old checkpoint (F=64, no CBAM) cannot be loaded into the new\n"
                        f"  model (F=32 + CBAM). The weights must be retrained from scratch.\n"
                        f"\n"
                        f"  TO FIX: retrain the model with the new architecture:\n"
                        f"      python scripts/train_unet.py\n"
                        f"\n"
                        f"  The old checkpoint is preserved at: checkpoints/unet_best_v1.pth\n"
                        f"  (safe to keep for comparison -- do NOT try to load it into this model)\n"
                    ) from None
                else:
                    raise   # re-raise unexpected errors unchanged
        else:
            print(
                "[CamouflageProcessor] WARNING: No checkpoint provided -- "
                "using random U-Net weights. For real inference, train first."
            )

        # ---- Optical Flow ----
        self._flow_module = FarnebackOpticalFlow()

        # ---- Fusion ----
        self._fuser = FrameFusion()

        # ---- Image normalisation (must match training in utils/dataset.py) ----
        self._mean = torch.tensor([0.485, 0.456, 0.406],
                                   device=self._device).view(1, 3, 1, 1)
        self._std  = torch.tensor([0.229, 0.224, 0.225],
                                   device=self._device).view(1, 3, 1, 1)

        print(f"[CamouflageProcessor] Device: {self._device}")
        print(f"[CamouflageProcessor] Default mode: {self._default_mode}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        frame_bgr: np.ndarray,
        mode: Optional[str] = None,
    ) -> ProcessorResult:
        """
        Process one frame through the Member 1 pipeline.

        Parameters
        ----------
        frame_bgr : np.ndarray, shape (H, W, 3), dtype uint8
            Raw BGR frame from cv2.VideoCapture or any OpenCV source.
        mode : str or None
            Override the default mode for this call.
            "video" = full pipeline (U-Net + Optical Flow + temporal smoothing).
            "image" = U-Net only (no flow, no temporal state).
            None    = use the mode set at construction time.

        Returns
        -------
        ProcessorResult
            See the dataclass definition above.
            The .enhanced_frame attribute is what YOLOv8 should receive.
        """
        # Resolve effective mode for this call
        effective_mode = mode if mode is not None else self._default_mode
        if effective_mode not in ("video", "image"):
            raise ValueError(
                f"mode must be 'video' or 'image', got: {effective_mode!r}."
            )

        orig_h, orig_w = frame_bgr.shape[:2]

        # IMAGE MODE: reset temporal state so this frame is treated as standalone.
        # This prevents stale EMA values from a previous video sequence affecting
        # the image-mode output. No optical flow is computed.
        if effective_mode == "image":
            self.reset()

        # ---- Step 1: U-Net segmentation (both modes) ----
        seg_map_unet_res = self._run_unet(frame_bgr)   # (unet_H, unet_W)

        # Resize seg map back to original frame resolution
        # Note: OpenCV 5 requires explicit dtype handling for single-channel float maps.
        seg_map = cv2.resize(
            seg_map_unet_res.astype(np.float64),
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)   # (orig_H, orig_W)

        # ---- Step 2: Optical Flow (VIDEO MODE ONLY) ----
        # In IMAGE mode, skip flow entirely -- there is no previous frame to compare
        # against, and temporal EMA has been reset. flow_map stays None.
        flow_map = None
        if effective_mode == "video" and config.RUN_FLOW_EVERY_FRAME:
            flow_map = self._flow_module.process_frame(frame_bgr)
            # Returns None for the very first frame -- handled gracefully by fuser

        # ---- Step 3: Fusion + Enhancement ----
        attention_map  = self._fuser.get_attention_map(
            orig_h, orig_w, seg_map, flow_map
        )
        enhanced_frame = self._fuser.fuse(frame_bgr, seg_map, flow_map)

        return ProcessorResult(
            enhanced_frame=enhanced_frame,
            seg_mask=seg_map,
            flow_map=flow_map,
            attention_map=attention_map,
            mode=effective_mode,
        )

    def reset(self) -> None:
        """
        Reset the internal state of the optical flow module.
        Call this when starting a new video clip or camera session.
        Also called automatically at the start of each IMAGE mode call.
        """
        self._flow_module.reset()

    def visualise(
        self,
        original_bgr: np.ndarray,
        result: ProcessorResult,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate a 4-panel debug visualisation:
          [Original | Seg Mask | Flow Map | Enhanced]

        Parameters
        ----------
        original_bgr : np.ndarray
            The raw input frame (before enhancement).
        result : ProcessorResult
            Output from process().
        save_path : str or None
            If provided, saves the visualisation to this path.

        Returns
        -------
        viz : np.ndarray, shape (H, W*4, 3), dtype uint8
            Side-by-side visualisation in BGR.
        """
        H, W = original_bgr.shape[:2]

        def _map_to_bgr(m: Optional[np.ndarray]) -> np.ndarray:
            if m is None:
                # Return a black panel labelled "N/A" for image-mode flow
                blank = np.zeros((H, W, 3), dtype=np.uint8)
                cv2.putText(blank, "No Flow", (W // 4, H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                return blank
            m_u8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
            return cv2.applyColorMap(m_u8, cv2.COLORMAP_JET)

        seg_vis   = _map_to_bgr(result.seg_mask)
        flow_vis  = _map_to_bgr(result.flow_map)

        # Add text labels
        def _label(img: np.ndarray, text: str) -> np.ndarray:
            out = img.copy()
            cv2.putText(out, text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return out

        mode_label = f"Enhanced ({result.mode} mode)"
        panels = [
            _label(original_bgr,          "Original"),
            _label(seg_vis,               "U-Net Seg"),
            _label(flow_vis,              "Optical Flow"),
            _label(result.enhanced_frame, mode_label),
        ]

        viz = np.concatenate(panels, axis=1)   # (H, 4*W, 3)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            cv2.imwrite(save_path, viz)
            print(f"[CamouflageProcessor] Debug viz saved: {save_path}")

        return viz

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_unet(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Run U-Net on one frame.

        1. Convert BGR -> RGB
        2. Resize to (INPUT_HEIGHT, INPUT_WIDTH)
        3. Normalise with ImageNet mean/std
        4. Forward pass through U-Net
        5. Return (H, W) float32 numpy array in [0, 1]
        """
        # BGR -> RGB -> float32 in [0, 1]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Resize to U-Net input size
        rgb_resized = cv2.resize(
            rgb,
            (config.INPUT_WIDTH, config.INPUT_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        # (H, W, 3) -> (1, 3, H, W) tensor
        tensor = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self._device)

        # Normalise
        tensor = (tensor - self._mean) / self._std

        # Forward -- model returns raw logits (by design, no sigmoid inside U-Net)
        with torch.no_grad():
            with torch.amp.autocast(
                device_type=self._device.type,
                enabled=(self._device.type == "cuda"),
            ):
                logits = self._model(tensor)            # (1, 1, H, W)  raw logits

        # Convert to probability ONCE here, outside the network.
        # This keeps the U-Net architecture clean (logits only) while ensuring
        # seg_map is a proper [0, 1] probability map for fusion.
        prob    = torch.sigmoid(logits)                 # (1, 1, H, W)  [0, 1]
        seg_map = prob.squeeze().float().cpu().numpy()  # (H, W)  float32  [0, 1]
        return seg_map


# ---------------------------------------------------------------------------
# Quick self-test (no dataset, no trained weights needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=== CamouflageProcessor self-test ===")
    print("  (Uses random U-Net weights -- no checkpoint needed)\n")

    H, W = 480, 640

    # ---- VIDEO MODE test ----
    print("--- VIDEO MODE ---")
    video_proc = CamouflageProcessor(checkpoint_path=None, mode="video")
    dummy_frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    # Frame 1 (no optical flow yet)
    print("  Processing frame 1 ...")
    result1 = video_proc.process(dummy_frame)
    print(f"    mode           : {result1.mode}")
    print(f"    enhanced_frame : {result1.enhanced_frame.shape}  {result1.enhanced_frame.dtype}")
    print(f"    seg_mask       : {result1.seg_mask.shape}  [{result1.seg_mask.min():.3f}, {result1.seg_mask.max():.3f}]")
    print(f"    flow_map       : {result1.flow_map}  (None expected for first frame)")
    print(f"    attention_map  : {result1.attention_map.shape}")

    # Frame 2 (optical flow available)
    dummy_frame2 = np.roll(dummy_frame, shift=5, axis=1)
    print("\n  Processing frame 2 ...")
    result2 = video_proc.process(dummy_frame2)
    print(f"    enhanced_frame : {result2.enhanced_frame.shape}  {result2.enhanced_frame.dtype}")
    print(f"    flow_map       : {result2.flow_map.shape}  (should have shape now)")
    print(f"    attention_map  : {result2.attention_map.shape}")

    assert result2.enhanced_frame.shape == (H, W, 3)
    assert result2.enhanced_frame.dtype == np.uint8
    assert result2.seg_mask.shape == (H, W)
    assert result2.flow_map is not None
    assert result2.flow_map.shape == (H, W)
    print("  VIDEO MODE: PASS")

    # ---- IMAGE MODE test ----
    print("\n--- IMAGE MODE ---")
    image_proc = CamouflageProcessor(checkpoint_path=None, mode="image")
    # Even if we call it many times, flow should always be None
    for i in range(3):
        res = image_proc.process(dummy_frame)
        assert res.flow_map is None, f"IMAGE mode should never return flow, got {res.flow_map}"
        assert res.mode == "image"
    print("  IMAGE MODE (3x): PASS -- flow_map is always None")

    # Per-call mode override test
    res_video = video_proc.process(dummy_frame, mode="image")
    assert res_video.mode == "image"
    assert res_video.flow_map is None
    print("  Per-call override (video proc -> image mode): PASS")

    # Visualise
    viz_path = os.path.join(config.OUTPUTS_DIR, "debug_processor.jpg")
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    viz = video_proc.visualise(dummy_frame2, result2, save_path=viz_path)
    print(f"\n  Visualisation  : {viz.shape}  saved to {viz_path}")

    # Reset
    video_proc.reset()
    result_after_reset = video_proc.process(dummy_frame)
    assert result_after_reset.flow_map is None, "After reset, first frame should have no flow"
    print("  Reset          : PASS")

    print("\nPASS -- CamouflageProcessor OK.")
