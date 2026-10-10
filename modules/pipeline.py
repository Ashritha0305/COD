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
from dataclasses import dataclass, field
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


@dataclass
class PipelineSession:
    """
    Session container isolating state for a single video or continuous stream.
    Tracks frame counter, intermediate dimensions, cached depth, and timing.
    """
    frame_idx: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    proc_w: Optional[int] = None
    proc_h: Optional[int] = None
    scale_x: float = 1.0
    scale_y: float = 1.0
    cached_depth_raw: Optional[np.ndarray] = None
    cached_depth_viz: Optional[np.ndarray] = None
    cached_depth_info: Optional[Dict[str, Any]] = None
    last_depth_frame: int = 0
    last_frame_time: Optional[float] = None
    rolling_fps: float = 0.0

    def reset(self) -> None:
        """Reset temporal counters and caches for a new video or camera stream."""
        self.frame_idx = 0
        self.width = None
        self.height = None
        self.proc_w = None
        self.proc_h = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.cached_depth_raw = None
        self.cached_depth_viz = None
        self.cached_depth_info = None
        self.last_depth_frame = 0
        self.last_frame_time = None
        self.rolling_fps = 0.0


class WildlifePipeline:
    """
    End-to-End Orchestrator for both Image Mode and Video Mode.
    """

    def __init__(
        self,
        checkpoint_path: str = config.BEST_MODEL_PATH,
        yolo_weights: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        print("[WildlifePipeline] Initializing pipeline components...")
        self.device = device

        if yolo_weights is None:
            if getattr(config, "YOLO_MODE", "world") == "world":
                yolo_weights = getattr(config, "YOLO_WORLD_WEIGHTS", "yolov8s-worldv2.pt")
            else:
                yolo_weights = config.YOLO_WILDLIFE_WEIGHTS

        # 1-3: U-Net, Optical Flow, Fusion
        self.camo_processor = CamouflageProcessor(
            checkpoint_path=checkpoint_path,
            device=device,
            mode="video",
        )

        # 4: Wildlife Detector (YOLO-World Open-Vocabulary or Custom)
        try:
            self.detector = WildlifeDetector(
                weights_path=yolo_weights,
                device=device,
                conf_thresh=config.YOLO_CONF_THRESH,
                mode=getattr(config, "YOLO_MODE", "world"),
            )
        except Exception as det_err:
            print(f"[WildlifePipeline] Warning initializing detector ({det_err}). Running without detector.")
            self.detector = None

        # 5: DeepSORT with Temporal Class Stabilization
        self.tracker = WildlifeTracker(max_age=30, n_init=2)

        # 6: ZoeDepth
        self.depth_estimator = ZoeDepthEstimator(device=device)

        # 7: Rule-Based Threat Analysis
        self.threat_analyzer = RuleBasedThreatAnalyzer()

        # Session state for temporal cadence and caching
        self._session = PipelineSession()

        print("[WildlifePipeline] Complete pipeline successfully initialized.")

    def reset(self) -> None:
        """
        Reset all temporal state across processors, tracker, and session.
        Ensures clean separation between video clips or streaming sessions.
        """
        self.camo_processor.reset()
        self.tracker.reset()
        self._session.reset()

    def _detect_camouflaged_wildlife(
        self,
        frame_bgr: np.ndarray,
        enhanced_bgr: np.ndarray,
        seg_mask: np.ndarray,
        binary_mask: Optional[np.ndarray],
        flow_map: Optional[np.ndarray] = None,
    ) -> List[Tuple[List[float], float, str]]:
        """
        Wildlife Detection Stage:
        1. Evaluate custom YOLOv8 detector on the full enhanced frame (Fusion output).
        2. If YOLO detects species above confidence threshold, retain detection.
        3. If YOLO detects nothing or species cannot be identified, evaluate camouflaged
           candidates spotted by U-Net / Optical Flow and assign UNKNOWN / LOW CONFIDENCE
           with true measured camouflage peak score (never fabricate species labels).
        """
        H, W = frame_bgr.shape[:2]
        raw_detections: List[Tuple[List[float], float, str]] = []

        # 1. Evaluate YOLOv8 on full enhanced frame
        if self.detector is not None:
            raw_detections = self.detector.detect(enhanced_bgr)

        # 2. Extract camouflaged candidate regions from U-Net / Optical Flow
        creature_mask = (
            binary_mask.copy()
            if binary_mask is not None
            else np.zeros((H, W), dtype=np.uint8)
        )

        if flow_map is not None:
            motion_c = (flow_map > 0.15) & (seg_mask > 0.10)
            creature_mask = np.maximum(creature_mask, (motion_c.astype(np.uint8) * 255))

        # Adaptive fallback if binary mask is empty but seg_mask has local peaks
        if np.sum(creature_mask > 0) == 0:
            p_max = float(seg_mask.max())
            p_min = float(seg_mask.min())
            if (p_max - p_min) >= 0.08:
                thresh = float(np.clip(p_min + (p_max - p_min) * 0.45, 0.12, 0.50))
                creature_mask = (seg_mask >= thresh).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            creature_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        min_area = H * W * 0.001
        max_area = H * W * 0.85

        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                rx, ry, rw, rh = cv2.boundingRect(c)
                p_peak = (
                    float(seg_mask[ry : ry + rh, rx : rx + rw].max())
                    if rw > 0 and rh > 0
                    else 0.0
                )
                candidates.append((rx, ry, rw, rh, p_peak, area))

        candidates.sort(key=lambda item: (item[4], item[5]), reverse=True)

        # 3. If YOLO detected nothing on full enhanced frame but U-Net spotted camouflaged candidates:
        if not raw_detections and candidates:
            # Retain top camouflaged candidate as UNKNOWN / LOW CONFIDENCE
            rx, ry, rw, rh, p_peak, _ = candidates[0]
            raw_detections.append(
                ([float(rx), float(ry), float(rw), float(rh)], float(p_peak), "UNKNOWN / LOW CONFIDENCE")
            )
        elif raw_detections and candidates:
            # Check if any candidate represents a distinct camouflaged target not covered by YOLO boxes
            def box_iou(b1, b2):
                x1 = max(b1[0], b2[0])
                y1 = max(b1[1], b2[1])
                x2 = min(b1[0] + b1[2], b2[0] + b2[2])
                y2 = min(b1[1] + b1[3], b2[1] + b2[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                area1 = b1[2] * b1[3]
                area2 = b2[2] * b2[3]
                union = area1 + area2 - inter
                return inter / union if union > 0 else 0.0

            for rx, ry, rw, rh, p_peak, _ in candidates[:2]:
                cand_box = [float(rx), float(ry), float(rw), float(rh)]
                overlaps = any(box_iou(cand_box, d[0]) > 0.25 for d in raw_detections)
                if not overlaps and p_peak >= 0.35:
                    raw_detections.append((cand_box, float(p_peak), "UNKNOWN / LOW CONFIDENCE"))

        return raw_detections

    def process_image(
        self,
        image_input: Any,  # file path (str) or np.ndarray (H, W, 3) BGR
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline:
        Input -> U-Net -> Enhancement -> YOLO -> ZoeDepth -> Threat Analysis
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

        # Step 4: Camouflaged Animal Detection via Kid + Elder Paradigm:
        # Kid (U-Net) segments candidate creature -> Elder (YOLO/Classifier) names it
        raw_detections = self._detect_camouflaged_wildlife(
            frame_bgr=frame_bgr,
            enhanced_bgr=enhanced_bgr,
            seg_mask=seg_mask,
            binary_mask=camo_res.binary_mask,
            flow_map=None,
        )

        # Find primary animal / detection
        primary_bbox = None
        if raw_detections:
            # Sort by confidence descending
            raw_detections.sort(key=lambda d: d[1], reverse=True)
            (l, t, w, h), conf, cls_name = raw_detections[0]
            primary_bbox = (int(l), int(t), int(l + w), int(t + h))
            animal_class = cls_name.replace("_", " ").title()
            confidence = float(conf)
        else:
            primary_bbox = None
            animal_class = "UNKNOWN"
            confidence = 0.0

        # Step 5: ZoeDepth Monocular Depth Estimation
        depth_raw, depth_viz, depth_info = self.depth_estimator.estimate_depth(
            frame_bgr,
            bbox_ltrb=primary_bbox,
        )

        # Step 6: Rule-Based Threat Analysis
        formatted_detections = []
        if primary_bbox and confidence > 0:
            formatted_detections.append({
                "class_name": animal_class,
                "confidence": confidence,
                "motion_magnitude": None,
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

        # Clean binary U-Net mask directly from U-Net stage
        binary_mask = (
            camo_res.binary_mask
            if hasattr(camo_res, "binary_mask") and camo_res.binary_mask is not None
            else ((seg_mask >= config.SEG_THRESHOLD).astype(np.uint8) * 255)
        )
        fg_pct = (
            camo_res.fg_pct
            if hasattr(camo_res, "fg_pct")
            else (float(np.sum(binary_mask > 0)) / (H * W) * 100.0)
        )

        # ==================================================
        # FINAL DEBUG OUTPUT (SECTION 17 / FINAL DEBUG SPEC)
        # ==================================================
        yolo_model_str = (
            os.path.abspath(self.detector.weights_path).replace(chr(92), "/")
            if self.detector else "CUSTOM YOLO MODEL NOT AVAILABLE"
        )
        yolo_classes_count = len(self.detector.names) if self.detector else 0

        print("\n" + "=" * 50)
        print("[U-NET]")
        print(f"Checkpoint: {self.camo_processor._checkpoint_path or 'checkpoints/unet_best.pth'}")
        print(f"Input: ({H}, {W}, 3)")
        print(f"Output: ({H}, {W})")
        print(f"Threshold: {config.SEG_THRESHOLD:.2f}")
        print(f"Foreground percentage: {fg_pct:.2f}%")
        print()
        print("[YOLO]")
        print(f"Model: {yolo_model_str}")
        print(f"Number of classes: {yolo_classes_count}")
        print(f"Detection count: {len(raw_detections)}")
        print(f"Class: {animal_class}")
        print(f"Confidence: {f'{confidence * 100:.1f}%' if confidence > 0 else 'LOW'}")
        print()
        print("[ZOEDEPTH]")
        print(f"Depth map: {'AVAILABLE' if depth_info.get('available') else 'N/A'}")
        print("Available" if depth_info.get('available') else "Not available")
        print()
        print("[THREAT]")
        print(f"Level: {threat_result.threat_level}")
        print("=" * 50 + "\n")

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
        mask_viz = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
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
            cv2.imwrite(saved_paths["mask"], binary_mask)
            cv2.imwrite(saved_paths["enhanced"], enhanced_bgr)
            cv2.imwrite(saved_paths["depth"], depth_viz)
            cv2.imwrite(saved_paths["comparison"], comparison_grid)
            cv2.imwrite(saved_paths["annotated"], annotated_bgr)

        return {
            "original_frame": frame_bgr,
            "seg_mask": seg_mask,
            "binary_mask": binary_mask,
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
        Process a video frame-by-frame through canonical process_frame() engine:
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

        # Handle screen recordings where container dimensions might be 0 until first read
        if width <= 0 or height <= 0:
            ret, first_frame = cap.read()
            if ret and first_frame is not None:
                height, width = first_frame.shape[:2]
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                width, height = 640, 480

        # Reset state between video clips
        self.reset()

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
                if not ret or frame_bgr is None:
                    break

                # Ensure strict shape consistency across video frames
                if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                    frame_bgr = cv2.resize(frame_bgr, (width, height))

                if max_frames and frame_idx >= max_frames:
                    break

                # Execute canonical frame processor
                frame_res = self.process_frame(
                    frame_bgr=frame_bgr,
                    frame_idx=frame_idx,
                    return_viz=True,
                )

                annotated_frame = frame_res["annotated_frame"]
                threat_res = frame_res["threat_result"]
                all_threats.append(threat_res)

                out_writer.write(annotated_frame)

                # Collect a few sample comparison grids (e.g. at frame 5, midpoint, near end)
                if frame_idx in (5, frames_to_process // 2, max(0, frames_to_process - 2)):
                    seg_m = frame_res["seg_mask"]
                    if seg_m.shape[:2] != (height, width):
                        seg_m = cv2.resize(seg_m, (width, height), interpolation=cv2.INTER_LINEAR)
                    mask_viz = cv2.applyColorMap((seg_m * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    sample_grid = self._create_2x2_grid(frame_bgr, mask_viz, annotated_frame, frame_res["depth_viz"])
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
        severity_map = {"CRITICAL": 3, "WARNING": 2, "SAFE": 1, "UNKNOWN / LOW CONFIDENCE": 1}
        overall_threat = max(
            all_threats,
            key=lambda t: (severity_map.get(t.threat_level, 0), t.threat_score)
        ) if all_threats else None

        return {
            "output_video_path": output_video_path,
            "frames_processed": frame_idx,
            "actual_fps": actual_fps,
            "total_time_s": total_time,
            "overall_threat": overall_threat,
            "sample_comparisons": sample_comparisons,
        }

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: Optional[int] = None,
        session: Optional[PipelineSession] = None,
        return_viz: bool = True,
        override_run_unet: Optional[bool] = None,
        override_run_det: Optional[bool] = None,
        override_run_depth: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Canonical single-frame execution engine for offline video, testing, and future webcam streaming.

        Architecture Execution Flow:
            1. Intermediate Resolution Scaling (960x540)
            2. U-Net Camouflage Segmentation (Cadenced: VIDEO_UNET_INTERVAL)
            3. Optical Flow Motion Analysis (Farneback at 640x360)
            4. Natural Fusion (LAB CLAHE + HSV Saturation + Attention Map)
            5. Enhanced Frame Upscaling (Native Resolution)
            6. YOLOv8 Wildlife Detection (Cadenced: VIDEO_DETECTION_INTERVAL)
               with U-Net camouflage-candidate fallback (never fabricating labels)
            7. DeepSORT Tracking (Native coordinates, zero appearance embeddings on skipped frames)
            8. ZoeDepth Relative Depth Estimation (Cadenced: VIDEO_DEPTH_INTERVAL, scaled bounding boxes)
            9. Rule-Based Threat Analysis (SAFE / WARNING / CRITICAL / UNKNOWN)
           10. HUD Strip & Threat Banner Visualization
        """
        t_frame_start = time.time()
        sess = session if session is not None else self._session

        height, width = frame_bgr.shape[:2]

        # Synchronize frame index
        if frame_idx is not None:
            sess.frame_idx = frame_idx
        current_idx = sess.frame_idx

        # Initialize or adapt intermediate working resolution
        target_w = getattr(config, "VIDEO_PROCESS_WIDTH", 960)
        target_h = getattr(config, "VIDEO_PROCESS_HEIGHT", 540)

        if sess.width != width or sess.height != height or sess.proc_w is None:
            if width > target_w:
                proc_w = target_w
                proc_h = int(height * (target_w / float(width)))
                proc_h = proc_h - (proc_h % 2)  # Ensure even dimension
            else:
                proc_w, proc_h = width, height

            sess.width = width
            sess.height = height
            sess.proc_w = proc_w
            sess.proc_h = proc_h
            sess.scale_x = width / float(proc_w)
            sess.scale_y = height / float(proc_h)
        else:
            proc_w = sess.proc_w
            proc_h = sess.proc_h

        scale_x = sess.scale_x
        scale_y = sess.scale_y

        # Intermediate resolution frame for efficient processing
        if (proc_w, proc_h) != (width, height):
            proc_frame = cv2.resize(frame_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            proc_frame = frame_bgr

        # Video optimization cadence intervals from config.py
        unet_interval = getattr(config, "VIDEO_UNET_INTERVAL", 2)
        det_interval = getattr(config, "VIDEO_DETECTION_INTERVAL", 3)
        depth_interval = getattr(config, "VIDEO_DEPTH_INTERVAL", 5)

        run_unet = (current_idx % unet_interval == 0) if override_run_unet is None else override_run_unet
        run_det = (current_idx % det_interval == 0) if override_run_det is None else override_run_det
        run_depth = (current_idx % depth_interval == 0) if override_run_depth is None else override_run_depth

        # 1-3. Camouflage U-Net + Optical Flow + Fusion
        # U-Net runs every VIDEO_UNET_INTERVAL frames; intermediate frames reuse previous mask
        camo_res = self.camo_processor.process(
            proc_frame,
            mode="video",
            reuse_unet=(not run_unet),
        )
        seg_mask = camo_res.seg_mask
        flow_map = camo_res.flow_map
        enhanced_proc = camo_res.enhanced_frame

        # Enhanced frame at original native resolution
        if (proc_w, proc_h) != (width, height):
            enhanced_orig = cv2.resize(enhanced_proc, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            enhanced_orig = enhanced_proc

        # 4. Camouflaged Animal Detection via Kid + Elder Paradigm:
        # Runs every VIDEO_DETECTION_INTERVAL frames.
        # In-between frames, DeepSORT tracking advances existing tracks without re-detecting.
        if run_det:
            raw_detections_proc = self._detect_camouflaged_wildlife(
                frame_bgr=proc_frame,
                enhanced_bgr=enhanced_proc,
                seg_mask=seg_mask,
                binary_mask=camo_res.binary_mask,
                flow_map=flow_map,
            )
            # Map detections from working resolution back to native resolution
            raw_detections = [
                ([b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y], conf, cls_name)
                for (b, conf, cls_name) in raw_detections_proc
            ]
        else:
            raw_detections = []

        # 5. DeepSORT Tracking (persistent IDs across frames)
        # On non-detection frames, advances Kalman filter tracks with zero re-ID embedding cost!
        tracked_objects = self.tracker.update(raw_detections, enhanced_orig)

        # 6. ZoeDepth Monocular Depth Estimation
        # Runs every VIDEO_DEPTH_INTERVAL frames. Reuses latest valid depth in-between.
        primary_bbox = tracked_objects[0]["bbox_ltrb"] if tracked_objects else None
        if run_depth or sess.cached_depth_info is None:
            scaled_depth_bbox = (
                [
                    int(primary_bbox[0] / scale_x),
                    int(primary_bbox[1] / scale_y),
                    int(primary_bbox[2] / scale_x),
                    int(primary_bbox[3] / scale_y),
                ]
                if primary_bbox else None
            )
            depth_raw, depth_viz, depth_info = self.depth_estimator.estimate_depth(
                proc_frame,
                bbox_ltrb=scaled_depth_bbox,
            )
            depth_info["status"] = "UPDATED"
            depth_info["note"] = f"Depth: UPDATED (Frame {current_idx})"
            sess.cached_depth_raw = depth_raw
            sess.cached_depth_viz = depth_viz
            sess.cached_depth_info = depth_info
            sess.last_depth_frame = current_idx
            depth_status_lbl = "UPDATED"
        else:
            depth_raw = sess.cached_depth_raw
            depth_viz = sess.cached_depth_viz
            depth_info = dict(sess.cached_depth_info)
            depth_info["status"] = "REUSED"
            depth_info["note"] = f"Depth: REUSED (from Frame {sess.last_depth_frame})"
            depth_status_lbl = "REUSED"

        # 7. Rule-Based Threat Analysis
        threat_res = self.threat_analyzer.assess_frame(
            detections=tracked_objects,
            seg_mask=seg_mask,
            flow_map=flow_map,
            depth_info=depth_info,
        )

        frame_elapsed_s = time.time() - t_frame_start
        frame_elapsed_ms = frame_elapsed_s * 1000.0

        # Rolling FPS calculation
        if sess.last_frame_time is not None:
            dt = time.time() - sess.last_frame_time
            inst_fps = 1.0 / max(1e-4, dt)
            sess.rolling_fps = 0.9 * sess.rolling_fps + 0.1 * inst_fps if sess.rolling_fps > 0 else inst_fps
        else:
            sess.rolling_fps = 1.0 / max(1e-4, frame_elapsed_s)
        sess.last_frame_time = time.time()
        rolling_fps = sess.rolling_fps

        # 8. Visualization Rendering
        if return_viz:
            annotated_frame = enhanced_orig.copy()
            if tracked_objects:
                annotated_frame = self.tracker.draw_tracks(
                    annotated_frame,
                    tracked_objects,
                    color=threat_res.color_bgr,
                )
            else:
                # If U-Net detected an unclassified camouflaged region, highlight contour
                bin_mask = (seg_mask > 0.4).astype(np.uint8) * 255
                if (proc_w, proc_h) != (width, height):
                    bin_mask = cv2.resize(bin_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_c = [c for c in contours if cv2.contourArea(c) > 300]
                if valid_c:
                    cv2.drawContours(annotated_frame, valid_c, -1, threat_res.color_bgr, 2)

            # Overlay Threat Banner & Info HUD with performance indicators
            annotated_frame = self._overlay_threat_banner(annotated_frame, threat_res)
            annotated_frame = self._overlay_hud_info(
                annotated_frame,
                frame_idx=current_idx,
                threat=threat_res,
                flow_map=flow_map,
                fps=rolling_fps,
                latency_ms=frame_elapsed_ms,
                depth_status=depth_status_lbl,
            )

            if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                annotated_frame = cv2.resize(annotated_frame, (width, height))
        else:
            annotated_frame = enhanced_orig

        # Advance session frame counter
        sess.frame_idx = current_idx + 1

        return {
            "annotated_frame": annotated_frame,
            "enhanced_frame": enhanced_orig,
            "seg_mask": seg_mask,
            "binary_mask": camo_res.binary_mask,
            "flow_map": flow_map,
            "depth_raw": depth_raw,
            "depth_viz": depth_viz,
            "depth_info": depth_info,
            "threat_result": threat_res,
            "tracked_objects": tracked_objects,
            "frame_idx": current_idx,
            "elapsed_ms": frame_elapsed_ms,
            "rolling_fps": rolling_fps,
            "proc_frame": proc_frame,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "run_unet": run_unet,
            "run_det": run_det,
            "run_depth": run_depth,
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
        if "UNKNOWN" in threat.animal_class.upper():
            score_lbl = f"Camo Score: {threat.confidence*100:.0f}%" if threat.confidence > 0 else "Camo: N/A"
        else:
            score_lbl = f"Conf: {threat.confidence*100:.0f}%"
        details = f"Target: {threat.animal_class} | {score_lbl} | Motion: {threat.motion_level} | Depth: {threat.relative_depth}"
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
        latency_ms: Optional[float] = None,
        depth_status: Optional[str] = None,
    ) -> np.ndarray:
        """HUD status at bottom of video frames with performance indicator."""
        out = frame_bgr.copy()
        H, W = out.shape[:2]

        hud_h = 32
        overlay = out.copy()
        cv2.rectangle(overlay, (0, H - hud_h), (W, H), (15, 15, 15), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

        # Info line
        flow_str = f"Flow: {float(np.mean(flow_map)):.3f}" if flow_map is not None else "Flow: Init"
        depth_str = f"Depth: {depth_status}" if depth_status else threat.depth_note
        perf_str = f"FPS: {fps:.1f} ({latency_ms:.0f} ms/frame)" if latency_ms is not None else f"FPS: {fps:.1f}"
        info_text = f"Frame: {frame_idx:04d} | {flow_str} | {depth_str} | {perf_str}"
        cv2.putText(
            out,
            info_text,
            (12, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
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
