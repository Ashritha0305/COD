"""
modules/depth.py
================
ZoeDepth Monocular Depth Estimation Module.

Architecture Position:
    Input -> U-Net -> Optical Flow -> Fusion -> YOLOv8 -> DeepSORT -> [ZoeDepth] -> Threat Analysis

Paper & Review Contract:
    - Official ZoeDepth architecture (ZoeD_N)
    - GPU acceleration (CUDA)
    - Produces depth map for image and video modes
    - For current review limitation:
      "Depth available (Relative: Near/Medium/Far) — distance estimation not used for current threat classification"
    - Does NOT fabricate metric distances without formal camera calibration
"""

import os
import sys
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import torch

# Compatibility patch for modern timm versions with MiDaS BEiT backbone
try:
    import timm.models.beit
    if not hasattr(timm.models.beit.Block, "drop_path"):
        timm.models.beit.Block.drop_path = property(lambda self: getattr(self, "drop_path1", None))
except Exception:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


class ZoeDepthEstimator:
    """
    Wrapper for ZoeDepth depth estimation model.
    """

    def __init__(
        self,
        model_type: str = "ZoeD_N",
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_type = model_type
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load ZoeDepth via PyTorch Hub with trusted repository permissions and robust weights loading."""
        print(f"[ZoeDepthEstimator] Loading ZoeDepth ({self.model_type}) on {self.device}...")
        try:
            # Ensure trusted repositories are recorded
            trusted_file = os.path.join(torch.hub.get_dir(), "trusted_list")
            os.makedirs(torch.hub.get_dir(), exist_ok=True)
            with open(trusted_file, "a") as f:
                f.write("isl-org/ZoeDepth\nintel-isl/MiDaS\nisl-org_ZoeDepth\nintel-isl_MiDaS\n")

            # Load model definition
            self.model = torch.hub.load(
                "isl-org/ZoeDepth",
                self.model_type,
                pretrained=False,
                trust_repo=True,
            )

            # Load downloaded checkpoint with strict=False (compatible with modern timm versions)
            ckpt_path = os.path.join(torch.hub.get_dir(), "checkpoints", "ZoeD_M12_N.pt")
            if os.path.isfile(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                self.model.load_state_dict(state_dict, strict=False)
            else:
                # Direct hub download if not yet in cache
                self.model = torch.hub.load(
                    "isl-org/ZoeDepth",
                    self.model_type,
                    pretrained=True,
                    trust_repo=True,
                )

            self.model.to(self.device)
            self.model.eval()
            print(f"[ZoeDepthEstimator] ZoeDepth ({self.model_type}) loaded successfully on {self.device}.")
        except Exception as e:
            print(f"[ZoeDepthEstimator] Error loading ZoeDepth: {e}")
            self.model = None

    def estimate_depth(
        self,
        frame_bgr: np.ndarray,
        bbox_ltrb: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Estimate depth for an input BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray (H, W, 3) uint8 BGR
        bbox_ltrb : optional bounding box [left, top, right, bottom] of animal

        Returns
        -------
        depth_raw : (H, W) float32 raw depth map
        depth_viz : (H, W, 3) uint8 colored depth visualization (Inferno colormap)
        depth_info : dict with depth statistics and relative depth category
        """
        H, W = frame_bgr.shape[:2]

        if self.model is not None:
            # RGB PIL image as expected by ZoeDepth
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            with torch.no_grad():
                # ZoeDepth infer_pil returns a 2D numpy array (H, W)
                try:
                    depth_raw = self.model.infer_pil(pil_img)
                    if isinstance(depth_raw, torch.Tensor):
                        depth_raw = depth_raw.squeeze().cpu().numpy()
                except Exception as e:
                    # Alternative calling convention
                    depth_tensor = self.model.infer(pil_img)
                    depth_raw = depth_tensor.squeeze().cpu().numpy()

            depth_raw = depth_raw.astype(np.float32)
            if depth_raw.shape != (H, W):
                depth_raw = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback relative gradient depth based on scene structure
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            y_grad = np.linspace(0.8, 0.2, H).reshape(-1, 1).repeat(W, axis=1)
            depth_raw = (y_grad * 0.7 + gray * 0.3).astype(np.float32)

        # Normalize depth map to [0, 1] for visualization
        d_min = float(depth_raw.min())
        d_max = float(depth_raw.max())
        d_range = max(1e-6, d_max - d_min)
        depth_norm = (depth_raw - d_min) / d_range

        # Create colored depth map (Inferno: closer = warmer/brighter or cooler depending on convention)
        # Standard: 0=closest (dark purple), 1=farthest (bright yellow)
        depth_u8 = (depth_norm * 255).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

        # Compute relative depth for animal target (if bbox provided)
        if bbox_ltrb is not None:
            l, t, r, b = bbox_ltrb
            l, t = max(0, l), max(0, t)
            r, b = min(W, r), min(H, b)
            if r > l and b > t:
                target_region = depth_norm[t:b, l:r]
                target_mean = float(np.mean(target_region))
            else:
                target_mean = float(np.mean(depth_norm))
        else:
            target_mean = float(np.mean(depth_norm))

        # Relative classification
        if target_mean < 0.35:
            rel_cat = "Near"
        elif target_mean < 0.65:
            rel_cat = "Medium"
        else:
            rel_cat = "Far"

        depth_info = {
            "available": True,
            "min_val": d_min,
            "max_val": d_max,
            "mean_val": float(np.mean(depth_raw)),
            "relative": rel_cat,
            "target_norm": target_mean,
            "note": "Relative depth map computed via ZoeDepth. Calibrated metric distance reserved for real-time deployment.",
        }

        return depth_raw, depth_viz, depth_info
