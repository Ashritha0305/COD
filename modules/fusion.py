"""
modules/fusion.py
==================
Phase 5 — Fusion and Enhancement Module  (v2 — Natural Enhancement)

PURPOSE
-------
Combine the U-Net segmentation probability map and the Farneback optical
flow motion magnitude map into an animal-aware attention map, then use
that attention map to produce a *natural-looking* enhanced frame in which
the camouflaged animal is visually distinguishable from its background.

WHAT CHANGED FROM v1
---------------------
v1 used a single formula:
    enhanced = frame * (1 + strength * attention)

This caused the animal to blow out to pure white because:
  a) seg_map arrived as raw logits (now fixed upstream), and
  b) the multiplicative formula only does brightness — no local structure.

v2 uses a 4-step natural enhancement pipeline:

  Step 1 — U-Net gates the flow map
    flow_gated = flow_map * seg_map
    Prevents unrelated background motion (grass, water) from being
    amplified in regions where the U-Net sees no animal.

  Step 2 — Temporal EMA smoothing of the attention map
    attention_t = alpha * attention_{t-1} + (1-alpha) * attention_t
    Eliminates frame-to-frame flickering of the highlight.

  Step 3 — CLAHE-based local contrast enhancement (LAB L-channel)
    Apply CLAHE to the L (luminance) channel of the LAB colour space,
    then blend it into the original frame gated by the attention map.
    CLAHE enhances local edge/detail structure without blowing out
    already-bright regions.  Animals hidden by colour matching are exposed
    by their texture difference, not by raw brightness.

  Step 4 — Gentle HSV saturation boost in animal region
    Increase saturation by (SAT_BOOST * attention) so that the animal's
    colours appear more vivid relative to the surrounding rocks/vegetation.
    Camouflaged animals often match the *luminance* of their background but
    differ subtly in saturation — amplifying this makes them visible.
    A very gentle luminance lift (STRENGTH * attention) is applied last.

CONFIGURATION (all in config.py)
---------------------------------
  FUSION_UNET_WEIGHT          : weight of seg_map in attention (default 0.7)
  FUSION_FLOW_WEIGHT          : weight of gated flow in attention (default 0.3)
  FUSION_ENHANCEMENT_STRENGTH : luminance lift factor (default 0.15)
  FUSION_CLAHE_STRENGTH       : CLAHE clip limit (default 2.0)
  FUSION_SAT_BOOST            : HSV saturation increase in [0, 1] (default 0.25)
  FUSION_TEMPORAL_ALPHA       : EMA weight for previous attention (default 0.7)

API (unchanged — backward-compatible with Member 2)
----------------------------------------------------
    from modules.fusion import FrameFusion
    fuser = FrameFusion()
    enhanced_bgr = fuser.fuse(frame_bgr, seg_map, flow_map)
    attention    = fuser.get_attention_map(H, W, seg_map, flow_map)
    fuser.reset()   # call between video clips
"""

import os
import sys
from typing import Optional

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


class FrameFusion:
    """
    Natural camouflage-aware frame enhancement.

    Parameters
    ----------
    unet_weight : float
        Weight for the U-Net segmentation map in the combined attention.
    flow_weight : float
        Weight for the (U-Net-gated) optical flow map in the attention.
    strength : float
        Gentle luminance lift strength.  0 = no change, 1 = maximum.
    clahe_strength : float
        CLAHE clipLimit for local contrast enhancement.
    sat_boost : float
        Maximum saturation increase fraction [0, 1].
    temporal_alpha : float
        EMA weight for the previous attention map.  Higher = smoother/slower.
        0 = no temporal smoothing (every frame independent).
    """

    def __init__(
        self,
        unet_weight:    float = config.FUSION_UNET_WEIGHT,
        flow_weight:    float = config.FUSION_FLOW_WEIGHT,
        strength:       float = config.FUSION_ENHANCEMENT_STRENGTH,
        clahe_strength: float = config.FUSION_CLAHE_STRENGTH,
        sat_boost:      float = config.FUSION_SAT_BOOST,
        temporal_alpha: float = config.FUSION_TEMPORAL_ALPHA,
    ) -> None:
        self.unet_weight    = unet_weight
        self.flow_weight    = flow_weight
        self.strength       = strength
        self.clahe_strength = clahe_strength
        self.sat_boost      = sat_boost
        self.temporal_alpha = temporal_alpha

        # Temporal state
        self._prev_attention: Optional[np.ndarray] = None

        # CLAHE engine (reused across frames for efficiency)
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_strength,
            tileGridSize=(8, 8),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        frame_bgr: np.ndarray,
        seg_map:   Optional[np.ndarray] = None,
        flow_map:  Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Produce an enhanced BGR frame where the camouflaged animal stands out.

        Parameters
        ----------
        frame_bgr : (H, W, 3) uint8 BGR
        seg_map   : (H, W) float32 in [0, 1]  — U-Net probability map.
                    Must already be sigmoid-ed probabilities (not raw logits).
                    Pass None if U-Net was not run.
        flow_map  : (H, W) float32 in [0, 1]  — normalised motion magnitude.
                    Pass None for the first frame.

        Returns
        -------
        enhanced_bgr : (H, W, 3) uint8 — natural-looking enhanced frame.
        """
        H, W = frame_bgr.shape[:2]

        # Step 1: build gated, temporally-smoothed attention map
        attention = self._build_attention(seg_map, flow_map, H, W)

        # Step 2: apply 4-step natural enhancement
        enhanced = self._enhance_natural(frame_bgr, attention)

        return enhanced

    def get_attention_map(
        self,
        h: int,
        w: int,
        seg_map:  Optional[np.ndarray] = None,
        flow_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return the current combined attention map without modifying temporal state.
        Useful for visualisation.
        """
        # We build the raw attention without updating temporal state
        return self._build_attention_raw(seg_map, flow_map, h, w)

    def reset(self) -> None:
        """Reset temporal state. Call between video clips."""
        self._prev_attention = None

    # ------------------------------------------------------------------
    # Internal: attention map construction
    # ------------------------------------------------------------------

    def _build_attention(
        self,
        seg_map:  Optional[np.ndarray],
        flow_map: Optional[np.ndarray],
        H: int,
        W: int,
    ) -> np.ndarray:
        """Build attention, apply temporal EMA, update temporal state."""
        raw = self._build_attention_raw(seg_map, flow_map, H, W)

        # Temporal EMA smoothing
        if self._prev_attention is not None and self.temporal_alpha > 0.0:
            prev = self._prev_attention
            # Resize prev if frame size changed
            if prev.shape != raw.shape:
                prev = cv2.resize(
                    prev.astype(np.float64), (W, H), interpolation=cv2.INTER_LINEAR
                ).astype(np.float32)
            smoothed = self.temporal_alpha * prev + (1.0 - self.temporal_alpha) * raw
        else:
            smoothed = raw

        smoothed = np.clip(smoothed, 0.0, 1.0)
        self._prev_attention = smoothed
        return smoothed

    def _build_attention_raw(
        self,
        seg_map:  Optional[np.ndarray],
        flow_map: Optional[np.ndarray],
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        Compute a single-frame attention map from seg + gated flow.

        Key design: flow is MULTIPLIED by seg_map before combining.
        This means only motion that overlaps the animal region contributes
        to the attention — background grass/water motion is suppressed.
        """
        have_seg  = seg_map  is not None
        have_flow = flow_map is not None

        if not have_seg and not have_flow:
            return np.zeros((H, W), dtype=np.float32)

        # Resize maps to frame resolution if needed
        if have_seg and seg_map.shape[:2] != (H, W):
            seg_map = cv2.resize(
                seg_map.astype(np.float64), (W, H), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)

        if have_flow and flow_map.shape[:2] != (H, W):
            flow_map = cv2.resize(
                flow_map.astype(np.float64), (W, H), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)

        # Gate the flow map by the U-Net segmentation probability.
        # A pixel that is NOT identified as animal by U-Net contributes
        # little flow signal even if it moves fast.
        if have_flow and have_seg:
            flow_gated = flow_map * seg_map          # element-wise product
        elif have_flow:
            flow_gated = flow_map
        else:
            flow_gated = None

        # Weighted combination
        if have_seg and flow_gated is not None:
            w_seg  = self.unet_weight
            w_flow = self.flow_weight
        elif have_seg:
            w_seg, w_flow = 1.0, 0.0
            flow_gated    = None
        else:
            w_seg, w_flow = 0.0, 1.0

        w_total = w_seg + w_flow
        if w_total > 1e-6:
            w_seg  /= w_total
            w_flow /= w_total

        attention = np.zeros((H, W), dtype=np.float32)
        if have_seg:
            attention += w_seg * seg_map.astype(np.float32)
        if flow_gated is not None:
            attention += w_flow * flow_gated.astype(np.float32)

        return np.clip(attention, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Internal: natural 4-step enhancement
    # ------------------------------------------------------------------

    def _enhance_natural(
        self,
        frame_bgr: np.ndarray,
        attention: np.ndarray,
    ) -> np.ndarray:
        """
        Apply natural-looking enhancement guided by the attention map.

        4-step pipeline:
          1. CLAHE on LAB L-channel, blended by attention
          2. HSV saturation boost in animal region
          3. Gentle luminance lift
          4. Final clip to uint8
        """
        # Work in float32
        frame_f = frame_bgr.astype(np.float32)
        attn    = attention.astype(np.float32)          # (H, W) in [0, 1]
        attn_3  = attn[:, :, np.newaxis]                # (H, W, 1) for broadcasting

        # ----------------------------------------------------------
        # Step 1 — CLAHE local contrast enhancement (LAB L-channel)
        # ----------------------------------------------------------
        # Convert to LAB and enhance the L channel with CLAHE.
        # Blend the CLAHE-enhanced L back in proportion to the attention map:
        #   new_L = (1 - attn) * orig_L + attn * clahe_L
        # This means high-attention (animal) pixels get full CLAHE contrast,
        # while low-attention (background) pixels keep their original luminance.
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        L_clahe = self._clahe.apply(L)                         # (H, W) uint8
        L_f     = L.astype(np.float32)
        L_c_f   = L_clahe.astype(np.float32)
        L_blend = (1.0 - attn) * L_f + attn * L_c_f           # (H, W) float32
        L_new   = np.clip(L_blend, 0, 255).astype(np.uint8)

        lab_enhanced = cv2.merge([L_new, A, B])
        frame_contrast = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR).astype(np.float32)

        # ----------------------------------------------------------
        # Step 2 — HSV saturation boost in animal region
        # ----------------------------------------------------------
        # Boost saturation selectively where attention is high.
        # This makes the animal's colours appear more vivid against the
        # desaturated rocky/leafy background, without changing luminance.
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        S   = hsv[:, :, 1]                                     # (H, W) in [0, 255]

        # Boost formula: new_S = S * (1 + sat_boost * attn)
        # Cap at 255 to prevent hue inversion.
        S_boost = S * (1.0 + self.sat_boost * attn)
        S_boost = np.clip(S_boost, 0, 255)

        # Blend sat-boosted frame with the contrast-enhanced frame.
        # We apply the saturation change to the contrast-enhanced result.
        hsv_blend       = cv2.cvtColor(
            np.clip(frame_contrast, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV
        ).astype(np.float32)
        hsv_blend[:, :, 1] = S_boost
        frame_sat = cv2.cvtColor(
            np.clip(hsv_blend, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR
        ).astype(np.float32)

        # ----------------------------------------------------------
        # Step 3 — Gentle luminance lift
        # ----------------------------------------------------------
        # A small additive brightness increase in the animal region.
        # Formula: enhanced = frame * (1 + strength * attn)
        # With strength=0.15 and attn≤1, the maximum multiplier is 1.15
        # — a 15% brightness lift at most.  This is a subtle finishing
        # touch rather than the primary enhancement mechanism.
        frame_out = frame_sat * (1.0 + self.strength * attn_3)

        # ----------------------------------------------------------
        # Step 4 — Final clip
        # ----------------------------------------------------------
        return np.clip(frame_out, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Fusion v2 self-test ===")

    H, W = 256, 256
    frame    = np.random.randint(30, 200, (H, W, 3), dtype=np.uint8)
    seg_map  = np.random.rand(H, W).astype(np.float32)
    flow_map = np.random.rand(H, W).astype(np.float32)

    fuser = FrameFusion()

    # Both maps
    out1 = fuser.fuse(frame, seg_map, flow_map)
    assert out1.shape == (H, W, 3) and out1.dtype == np.uint8
    print(f"  Both maps          : shape={out1.shape}  dtype={out1.dtype}  PASS")

    # Seg only
    fuser.reset()
    out2 = fuser.fuse(frame, seg_map=seg_map, flow_map=None)
    assert out2.shape == (H, W, 3)
    print(f"  Seg only           : shape={out2.shape}  PASS")

    # Flow only
    fuser.reset()
    out3 = fuser.fuse(frame, seg_map=None, flow_map=flow_map)
    assert out3.shape == (H, W, 3)
    print(f"  Flow only          : shape={out3.shape}  PASS")

    # No maps → returns something (zero attention → mostly original)
    fuser.reset()
    out4 = fuser.fuse(frame, seg_map=None, flow_map=None)
    assert out4.shape == (H, W, 3)
    print(f"  No maps            : shape={out4.shape}  PASS")

    # Temporal smoothing — two frames, second should be different from first-raw
    fuser.reset()
    fuser.fuse(frame, seg_map, flow_map)   # frame 1 — seeds temporal state
    out5 = fuser.fuse(frame, seg_map, flow_map)  # frame 2 — uses smoothed attention
    assert out5.shape == (H, W, 3)
    print(f"  Temporal smoothing : shape={out5.shape}  PASS")

    # Attention map
    fuser.reset()
    attn = fuser.get_attention_map(H, W, seg_map, flow_map)
    assert attn.shape == (H, W)
    assert 0.0 <= attn.min() and attn.max() <= 1.0
    print(f"  Attention map      : range=[{attn.min():.3f}, {attn.max():.3f}]  PASS")

    # Size mismatch safety
    fuser.reset()
    small_seg = np.random.rand(H // 2, W // 2).astype(np.float32)
    out6 = fuser.fuse(frame, seg_map=small_seg, flow_map=None)
    assert out6.shape == (H, W, 3)
    print(f"  Size mismatch      : auto-resized correctly  PASS")

    # Enhancement should make animal-region pixels different from background
    # Use a perfect step-function attention map: left half=0, right half=1
    binary_attn = np.zeros((H, W), dtype=np.float32)
    binary_attn[:, W // 2:] = 1.0
    frame_uniform = np.ones((H, W, 3), dtype=np.uint8) * 100
    fuser.reset()
    out7 = fuser.fuse(frame_uniform, binary_attn, None)
    left_mean  = out7[:, :W // 2, :].mean()
    right_mean = out7[:, W // 2:, :].mean()
    assert right_mean > left_mean, (
        f"Enhanced region ({right_mean:.1f}) should be brighter than background ({left_mean:.1f})"
    )
    print(f"  Enhancement test   : background={left_mean:.1f} < animal={right_mean:.1f}  PASS")

    # Reset clears temporal state
    fuser.reset()
    assert fuser._prev_attention is None
    print(f"  Reset              : temporal state cleared  PASS")

    print("\nPASS — Fusion v2 module OK.")
