"""
modules/optical_flow.py
========================
Phase 4 — Farneback Dense Optical Flow Module

Computes dense optical flow between two consecutive frames using
OpenCV's Gunnar Farneback algorithm. Returns a normalised motion
magnitude map in [0, 1] that serves as the motion cue in the fusion step.

This is NOT a neural network. It is a classical computer vision algorithm.
No weights are trained. Parameters come from config.py.

API:
    from modules.optical_flow import FarnebackOpticalFlow
    flow_module = FarnebackOpticalFlow()

    # Process a consecutive pair of BGR frames (as numpy arrays):
    motion_map = flow_module.compute(prev_frame_bgr, curr_frame_bgr)
    # Returns: (H, W) float32 array, values in [0, 1]

    # Process a single new frame (stateful — remembers previous frame):
    motion_map = flow_module.process_frame(curr_frame_bgr)
    # Returns: (H, W) float32 array, or None for the very first frame.

Reference:
    Farneback, G., "Two-Frame Motion Estimation Based on Polynomial Expansion",
    SCIA 2003.
    OpenCV docs: cv2.calcOpticalFlowFarneback
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


class FarnebackOpticalFlow:
    """
    Stateful Farneback Optical Flow processor.

    Stateful means it remembers the previous frame internally, so you can
    call `process_frame(frame)` sequentially during video inference.

    For training-time use, call the stateless `compute(prev, curr)` directly.

    All algorithm parameters come from config.py:
        OF_PYR_SCALE  : pyramid scale (< 1)
        OF_LEVELS     : number of pyramid levels
        OF_WINSIZE    : averaging window size
        OF_ITERATIONS : iterations per pyramid level
        OF_POLY_N     : pixel neighbourhood size for polynomial expansion
        OF_POLY_SIGMA : Gaussian std for polynomial expansion
        OF_FLAGS      : OpenCV flags (0 = default)
    """

    def __init__(self) -> None:
        self._prev_gray: Optional[np.ndarray] = None  # stored as (H, W) uint8 grayscale
        self._prev_bgr:  Optional[np.ndarray] = None  # stored for external visualisation

    @property
    def prev_bgr(self) -> Optional[np.ndarray]:
        """The previous BGR frame stored internally. None before any frame is processed."""
        return self._prev_bgr

    # ------------------------------------------------------------------
    # Stateless API — given explicit prev and curr frames
    # ------------------------------------------------------------------

    def compute(
        self,
        prev_bgr: np.ndarray,
        curr_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Farneback optical flow between two BGR frames.

        Parameters
        ----------
        prev_bgr : np.ndarray, shape (H, W, 3), dtype uint8
            Previous frame in BGR colour order (as returned by cv2.imread or
            cv2.VideoCapture.read).
        curr_bgr : np.ndarray, shape (H, W, 3), dtype uint8
            Current frame in BGR colour order.

        Returns
        -------
        motion_map : np.ndarray, shape (H, W), dtype float32
            Normalised motion magnitude in [0, 1].
            Pixels with large motion → values closer to 1.
            Static pixels → values closer to 0.
        """
        prev_gray = _to_gray(prev_bgr)
        curr_gray = _to_gray(curr_bgr)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            config.OF_PYR_SCALE,
            config.OF_LEVELS,
            config.OF_WINSIZE,
            config.OF_ITERATIONS,
            config.OF_POLY_N,
            config.OF_POLY_SIGMA,
            config.OF_FLAGS,
        )  # flow: (H, W, 2) — horizontal and vertical displacement per pixel

        motion_map = _flow_to_magnitude(flow)  # (H, W) in [0, 1]
        return motion_map

    # ------------------------------------------------------------------
    # Stateful API — call frame by frame during video inference
    # ------------------------------------------------------------------

    def process_frame(self, curr_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Process one new frame. Remembers the previous frame automatically.

        Parameters
        ----------
        curr_bgr : np.ndarray, shape (H, W, 3), dtype uint8
            Current frame in BGR colour order.

        Returns
        -------
        motion_map : np.ndarray, shape (H, W), dtype float32, or None
            Normalised motion magnitude in [0, 1].
            Returns None for the very first frame (no previous frame exists).
        """
        curr_gray = _to_gray(curr_bgr)

        if self._prev_gray is None:
            # First frame — store it, return no motion map yet
            self._prev_gray = curr_gray
            self._prev_bgr  = curr_bgr.copy()
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray,
            curr_gray,
            None,
            config.OF_PYR_SCALE,
            config.OF_LEVELS,
            config.OF_WINSIZE,
            config.OF_ITERATIONS,
            config.OF_POLY_N,
            config.OF_POLY_SIGMA,
            config.OF_FLAGS,
        )

        self._prev_bgr  = curr_bgr.copy()  # store for external visualisation
        self._prev_gray = curr_gray         # update for next call
        return _flow_to_magnitude(flow)

    def reset(self) -> None:
        """
        Clear the stored previous frame.
        Call this when switching to a new video clip or scene.
        """
        self._prev_gray = None

    def get_flow_rgb(
        self,
        prev_bgr: np.ndarray,
        curr_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Compute flow and return as an HSV-coloured RGB image for visualisation.
        Hue = direction, Value = magnitude.

        Returns
        -------
        flow_rgb : np.ndarray, shape (H, W, 3), dtype uint8
        """
        prev_gray = _to_gray(prev_bgr)
        curr_gray = _to_gray(curr_bgr)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            config.OF_PYR_SCALE, config.OF_LEVELS, config.OF_WINSIZE,
            config.OF_ITERATIONS, config.OF_POLY_N, config.OF_POLY_SIGMA,
            config.OF_FLAGS,
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*prev_gray.shape, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2            # hue = direction
        hsv[..., 1] = 255                               # full saturation
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_gray(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 frame to a grayscale uint8 image."""
    if bgr.ndim == 2:
        return bgr  # already grayscale
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _flow_to_magnitude(flow: np.ndarray) -> np.ndarray:
    """
    Compute the per-pixel motion magnitude from a 2-channel flow field,
    then normalise to [0, 1].

    Parameters
    ----------
    flow : np.ndarray, shape (H, W, 2), dtype float32
        Farneback flow field (x-displacement, y-displacement).

    Returns
    -------
    motion_map : np.ndarray, shape (H, W), dtype float32
        Values in [0, 1]. 0 = no motion, 1 = maximum motion in this frame.
    """
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)   # (H, W)
    # Robust normalisation: use 99th-percentile as the ceiling instead of
    # the absolute maximum.  A single fast-moving outlier pixel (e.g. a leaf
    # at the frame edge) no longer compresses every other motion value toward
    # zero, which previously caused the entire frame to appear uniformly "moving".
    p99 = np.percentile(mag, 99)
    if p99 > 1e-6:
        mag = np.clip(mag / p99, 0.0, 1.0)
    else:
        mag = np.zeros_like(mag)
    return mag.astype(np.float32)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=== Optical Flow self-test ===")

    H, W = 240, 320

    # Simulate two consecutive frames
    prev_frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    curr_frame = np.roll(prev_frame, shift=5, axis=1)   # shift by 5px → real motion

    of = FarnebackOpticalFlow()

    # Stateless API
    motion = of.compute(prev_frame, curr_frame)
    print(f"  Stateless compute:")
    print(f"    Output shape : {motion.shape}")
    print(f"    dtype        : {motion.dtype}")
    print(f"    min / max    : {motion.min():.4f} / {motion.max():.4f}")
    assert motion.shape == (H, W), f"Expected ({H},{W}), got {motion.shape}"
    assert 0.0 <= motion.min() and motion.max() <= 1.0, "Values outside [0,1]"

    # Stateful API
    r1 = of.process_frame(prev_frame)
    r2 = of.process_frame(curr_frame)
    print(f"\n  Stateful process_frame:")
    print(f"    Frame 1 result : {r1}  (expected None)")
    print(f"    Frame 2 shape  : {r2.shape}")
    assert r1 is None, "First frame should return None"
    assert r2.shape == (H, W), f"Expected ({H},{W}), got {r2.shape}"

    # RGB visualisation
    flow_rgb = of.get_flow_rgb(prev_frame, curr_frame)
    print(f"\n  Flow RGB shape : {flow_rgb.shape}  dtype={flow_rgb.dtype}")

    # Reset test
    of.reset()
    r3 = of.process_frame(curr_frame)
    assert r3 is None, "After reset(), first frame should return None again"
    print("\n  Reset test     : PASS")

    print("\nPASS — Optical flow module OK.")
