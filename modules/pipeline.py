"""
modules/pipeline.py
===================
Unified End-to-End Wildlife Camouflage Detection & Threat Analysis Pipeline.

Strict Paper Architecture (NO unapproved models):
    1. U-Net (Camouflage Segmentation)
    2. Optical Flow (Farneback OpenCV)
    3. Fusion (Camouflage + Motion natural enhancement)
    4. YOLOv8 (Animal Detection)
    5. DeepSORT (Multi-Object Tracking)
    6. ZoeDepth (Monocular Depth Estimation)
    7. Rule-Based Threat Analysis (SAFE / WARNING / CRITICAL)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from modules.camouflage_processor import CamouflageProcessor
from modules.detection import WildlifeDetector
from modules.tracking import WildlifeTracker
from modules.depth import ZoeDepthEstimator
from modules.threat_analysis import RuleBasedThreatAnalyzer, ThreatResult


class WildlifePipeline:
    """
    End-to-End Orchestrator for both Image Mode and Video Mode.
    """

    def __init__(
        self,
        checkpoint_path: str = config.BEST_MODEL_PATH,
        yolo_weights: str = "yolov8n.pt",
        device: Optional[str] = None,
    ) -> None:
        print("[WildlifePipeline] Initializing pipeline components...")
        self.device = device

        # 1-3: U-Net, Optical Flow, Fusion
        self.camo_processor = CamouflageProcessor(
            checkpoint_path=checkpoint_path,
            device=device,
            mode="video",
        )

        # 4: YOLOv8
        self.detector = WildlifeDetector(
            weights_path=yolo_weights,
            device=device,
            conf_thresh=0.25,
        )

        # 5: DeepSORT
        self.tracker = WildlifeTracker(max_age=30, n_init=2)

        # 6: ZoeDepth
        self.depth_estimator = ZoeDepthEstimator(device=device)

        # 7: Rule-Based Threat Analysis
        self.threat_analyzer = RuleBasedThreatAnalyzer()

        print("[WildlifePipeline] Complete pipeline successfully initialized.")

    def process_image(
        self,
        image_input: Any,  # file path (str) or np.ndarray (H, W, 3) BGR
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline:
        Input -> U-Net -> Enhancement -> YOLOv8 -> ZoeDepth -> Threat Analysis
        """
        if isinstance(image_input, (str, Path)):
            frame_bgr = cv2.imread(str(image_input))
            if frame_bgr is None:
                raise ValueError(f"Could not read image from {image_input}")
        else:
            frame_bgr = image_input.copy()

        H, W = frame_bgr.shape[:2]
        t0 = time.time()

        # Step 1-3: Camouflage U-Net + Enhancement (image mode: flow skipped)
        camo_res = self.camo_processor.process(frame_bgr, mode="image")
        seg_mask = camo_res.seg_mask
        enhanced_bgr = camo_res.enhanced_frame

        # Step 4: YOLOv8 detection on enhanced frame (what YOLOv8 expects)
        raw_detections = self.detector.detect(enhanced_bgr)

        # Find primary animal / detection
        primary_bbox = None
        animal_class = "Camouflaged Wildlife"
        confidence = 0.50

        if raw_detections:
            (l, t, w, h), conf, cls_name = raw_detections[0]
            primary_bbox = (int(l), int(t), int(l + w), int(t + h))
            animal_class = cls_name.capitalize()
            confidence = conf
        else:
            # Fallback to U-Net camouflage region bounding box if available
            bin_mask = (seg_mask > 0.35).astype(np.uint8)
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_c = [c for c in contours if cv2.contourArea(c) > 200]
            if valid_c:
                c_max = max(valid_c, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(c_max)
                primary_bbox = (x, y, x + bw, y + bh)
                confidence = float(np.max(seg_mask[y:y+bh, x:x+bw]))

        # Step 5: ZoeDepth Monocular Depth Estimation
        depth_raw, depth_viz, depth_info = self.depth_estimator.estimate_depth(
            frame_bgr,
            bbox_ltrb=primary_bbox,
        )

        # Step 6: Rule-Based Threat Analysis
        formatted_detections = []
        if primary_bbox:
            formatted_detections.append({
                "class_name": animal_class,
                "confidence": confidence,
                "motion_magnitude": 0.0,
                "track_id": None,
                "persistence": 1,
            })

        threat_result = self.threat_analyzer.assess_frame(
            detections=formatted_detections,
            seg_mask=seg_mask,
            flow_map=None,
            depth_info=depth_info,
        )

        elapsed_ms = (time.time() - t0) * 1000.0

        # Create Annotated Output Image with Bounding Box & Threat Banner
        annotated_bgr = enhanced_bgr.copy()
        if primary_bbox:
            cv2.rectangle(
                annotated_bgr,
                (primary_bbox[0], primary_bbox[1]),
                (primary_bbox[2], primary_bbox[3]),
                threat_result.color_bgr,
                3,
            )
            lbl = f"{animal_class} ({confidence * 100:.1f}%)"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(
                annotated_bgr,
                (primary_bbox[0], max(0, primary_bbox[1] - th - 8)),
                (primary_bbox[0] + tw + 6, max(0, primary_bbox[1])),
                threat_result.color_bgr,
                cv2.FILLED,
            )
            cv2.putText(
                annotated_bgr,
                lbl,
                (primary_bbox[0] + 3, max(th + 2, primary_bbox[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Overlay Threat Banner at top of annotated frame
        annotated_bgr = self._overlay_threat_banner(annotated_bgr, threat_result)

        # Build 2x2 Professional Comparison Grid
        # [Original | U-Net Mask]
        # [Enhanced | Depth Map ]
        mask_viz = cv2.applyColorMap((seg_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        comparison_grid = self._create_2x2_grid(frame_bgr, mask_viz, annotated_bgr, depth_viz)

        # Save files if save_dir provided
        saved_paths = {}
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            saved_paths["original"] = os.path.join(save_dir, "original.jpg")
            saved_paths["mask"] = os.path.join(save_dir, "unet_mask.png")
            saved_paths["enhanced"] = os.path.join(save_dir, "enhanced.jpg")
            saved_paths["depth"] = os.path.join(save_dir, "depth_map.jpg")
            saved_paths["comparison"] = os.path.join(save_dir, "comparison_grid.jpg")
            saved_paths["annotated"] = os.path.join(save_dir, "annotated_result.jpg")

            cv2.imwrite(saved_paths["original"], frame_bgr)
            cv2.imwrite(saved_paths["mask"], (seg_mask * 255).astype(np.uint8))
            cv2.imwrite(saved_paths["enhanced"], enhanced_bgr)
            cv2.imwrite(saved_paths["depth"], depth_viz)
            cv2.imwrite(saved_paths["comparison"], comparison_grid)
            cv2.imwrite(saved_paths["annotated"], annotated_bgr)

        return {
            "original_frame": frame_bgr,
            "seg_mask": seg_mask,
            "enhanced_frame": enhanced_bgr,
            "annotated_frame": annotated_bgr,
            "depth_viz": depth_viz,
            "comparison_grid": comparison_grid,
            "threat_result": threat_result,
            "depth_info": depth_info,
            "elapsed_ms": elapsed_ms,
            "saved_paths": saved_paths,
        }

    def process_video(
        self,
        video_path: str,
        output_video_path: str,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a video frame-by-frame:
        Frame -> U-Net -> Optical Flow -> Fusion -> YOLOv8 -> DeepSORT -> ZoeDepth -> Threat Analysis -> Output Video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames and max_frames < total_frames:
            frames_to_process = max_frames
        else:
            frames_to_process = total_frames if total_frames > 0 else 100

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Reset state between video clips
        self.camo_processor.reset()
        self.tracker.reset()

        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_idx = 0
        all_threats: List[ThreatResult] = []
        sample_comparisons: List[np.ndarray] = []
        t_start = time.time()

        try:
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                if max_frames and frame_idx >= max_frames:
                    break

                # 1-3. Camouflage U-Net + Optical Flow + Fusion
                camo_res = self.camo_processor.process(frame_bgr, mode="video")
                seg_mask = camo_res.seg_mask
                flow_map = camo_res.flow_map
                enhanced_bgr = camo_res.enhanced_frame

                # 4. YOLOv8 Detection on Enhanced Frame
                raw_detections = self.detector.detect(enhanced_bgr)

                # 5. DeepSORT Tracking (persistent IDs across frames)
                tracked_objects = self.tracker.update(raw_detections, enhanced_bgr)

                # 6. ZoeDepth (for efficiency, compute every 2nd or 3rd frame in video, or on primary target)
                primary_bbox = tracked_objects[0]["bbox_ltrb"] if tracked_objects else None
                depth_raw, depth_viz, depth_info = self.depth_estimator.estimate_depth(
                    frame_bgr,
                    bbox_ltrb=primary_bbox,
                )

                # 7. Rule-Based Threat Analysis
                threat_res = self.threat_analyzer.assess_frame(
                    detections=tracked_objects,
                    seg_mask=seg_mask,
                    flow_map=flow_map,
                    depth_info=depth_info,
                )
                all_threats.append(threat_res)

                # Compose final annotated video frame
                # Draw tracked bounding boxes with persistent Track IDs
                annotated_frame = enhanced_bgr.copy()
                if tracked_objects:
                    annotated_frame = self.tracker.draw_tracks(
                        annotated_frame,
                        tracked_objects,
                        color=threat_res.color_bgr,
                    )
                else:
                    # If U-Net detected an unclassified camouflaged region, highlight contour
                    bin_mask = (seg_mask > 0.4).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid_c = [c for c in contours if cv2.contourArea(c) > 300]
                    if valid_c:
                        cv2.drawContours(annotated_frame, valid_c, -1, threat_res.color_bgr, 2)

                # Overlay Threat Banner & Info HUD
                annotated_frame = self._overlay_threat_banner(annotated_frame, threat_res)
                annotated_frame = self._overlay_hud_info(
                    annotated_frame,
                    frame_idx=frame_idx,
                    threat=threat_res,
                    flow_map=flow_map,
                    fps=fps,
                )

                out_writer.write(annotated_frame)

                # Collect a few sample comparison grids (e.g. at 25%, 50%, 75%)
                if frame_idx in (5, frames_to_process // 2, frames_to_process - 2):
                    mask_viz = cv2.applyColorMap((seg_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    sample_grid = self._create_2x2_grid(frame_bgr, mask_viz, annotated_frame, depth_viz)
                    sample_comparisons.append(sample_grid)

                frame_idx += 1
                if progress_callback:
                    progress_callback(frame_idx, frames_to_process, float(frame_idx / max(1, frames_to_process)))

        finally:
            cap.release()
            out_writer.release()

        total_time = time.time() - t_start
        actual_fps = frame_idx / max(1e-5, total_time)

        # Aggregate overall threat for the video
        severity_map = {"CRITICAL": 3, "WARNING": 2, "SAFE": 1}
        overall_threat = max(all_threats, key=lambda t: (severity_map[t.threat_level], t.threat_score)) if all_threats else None

        return {
            "output_video_path": output_video_path,
            "frames_processed": frame_idx,
            "actual_fps": actual_fps,
            "total_time_s": total_time,
            "overall_threat": overall_threat,
            "sample_comparisons": sample_comparisons,
        }

    # ------------------------------------------------------------------
    # Visual Layout Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _overlay_threat_banner(frame_bgr: np.ndarray, threat: ThreatResult) -> np.ndarray:
        """Draw an authoritative, prominent threat level banner at the top."""
        out = frame_bgr.copy()
        H, W = out.shape[:2]

        banner_h = 46
        # Semi-transparent dark background strip
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (W, banner_h), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

        # Threat Level Badge on the left
        badge_text = f"THREAT LEVEL: {threat.threat_level}"
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        badge_w = bw + 24
        cv2.rectangle(out, (10, 6), (10 + badge_w, banner_h - 6), threat.color_bgr, cv2.FILLED)
        cv2.putText(
            out,
            badge_text,
            (22, banner_h - 14),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Threat Details on the right
        details = f"Target: {threat.animal_class} | Conf: {threat.confidence*100:.0f}% | Motion: {threat.motion_level} | Depth: {threat.relative_depth}"
        cv2.putText(
            out,
            details,
            (10 + badge_w + 16, banner_h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        return out

    @staticmethod
    def _overlay_hud_info(
        frame_bgr: np.ndarray,
        frame_idx: int,
        threat: ThreatResult,
        flow_map: Optional[np.ndarray],
        fps: float,
    ) -> np.ndarray:
        """HUD status at bottom of video frames."""
        out = frame_bgr.copy()
        H, W = out.shape[:2]

        hud_h = 32
        overlay = out.copy()
        cv2.rectangle(overlay, (0, H - hud_h), (W, H), (15, 15, 15), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

        # Info line
        flow_str = f"Flow Mag: {float(np.mean(flow_map)):.3f}" if flow_map is not None else "Flow: Initializing"
        info_text = f"Frame: {frame_idx:04d} | {flow_str} | {threat.depth_note}"
        cv2.putText(
            out,
            info_text,
            (12, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        return out

    @staticmethod
    def _create_2x2_grid(
        orig: np.ndarray,
        mask: np.ndarray,
        enhanced: np.ndarray,
        depth: np.ndarray,
        target_size: Tuple[int, int] = (480, 360),  # (W, H) per quadrant
    ) -> np.ndarray:
        """Assemble 2x2 comparison grid labeled clearly."""
        w, h = target_size

        def _prep(img: np.ndarray, title: str, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
            resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            # Add top header bar
            bar = np.zeros((30, w, 3), dtype=np.uint8)
            cv2.rectangle(bar, (0, 0), (w, 30), (30, 30, 30), cv2.FILLED)
            cv2.putText(bar, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
            return np.vstack([bar, resized])

        p1 = _prep(orig, "1. Original Input")
        p2 = _prep(mask, "2. U-Net Camouflage Segmentation")
        p3 = _prep(enhanced, "3. Enhanced & Detected Wildlife")
        p4 = _prep(depth, "4. ZoeDepth Depth Map")

        row1 = np.hstack([p1, p2])
        row2 = np.hstack([p3, p4])
        return np.vstack([row1, row2])
