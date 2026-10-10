"""
scripts/test_unified_pipeline.py
================================
Comprehensive regression test suite for Phase 2: Unified COD Pipeline.

Tests:
    1. Image mode execution, output formats, and saved artifacts.
    2. Video mode execution through unified process_frame() engine.
    3. Canonical process_frame() single-frame engine and cadence intervals.
    4. Empty detections fallback (never fabricates species or confidence).
    5. Unknown / camouflaged animal candidate fallback.
    6. Stationary animal handling (zero optical flow, attention preservation).
    7. Temporal state resets (clean session separation).
    8. Coordinate scaling consistency (1080p native <-> 540p working <-> depth bbox).
"""

import os
import sys
import time
import unittest
from pathlib import Path
import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from modules.pipeline import WildlifePipeline, PipelineSession
from modules.threat_analysis import ThreatResult


class TestUnifiedPipelineRegression(unittest.TestCase):
    """Regression test cases for unified COD pipeline."""

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print("INITIALIZING WILDLIFE PIPELINE FOR REGRESSION TESTS")
        print("=" * 70)
        cls.pipeline = WildlifePipeline()
        cls.test_image_path = os.path.join(_PROJECT_ROOT, "data", "test_image.jpg")
        cls.test_video_path = os.path.join(_PROJECT_ROOT, "data", "testvideo.mp4")

    def setUp(self):
        # Reset state before every individual test
        self.pipeline.reset()

    def test_1_image_mode_outputs_and_artifacts(self):
        """Image mode: preserves output dictionary keys, threat categories, and artifacts."""
        out_dir = os.path.join(_PROJECT_ROOT, "outputs", "regression_test_image")
        os.makedirs(out_dir, exist_ok=True)

        results = self.pipeline.process_image(self.test_image_path, save_dir=out_dir)

        # 1. Output dictionary key verification
        expected_keys = [
            "original_frame", "seg_mask", "binary_mask", "enhanced_frame",
            "annotated_frame", "depth_viz", "comparison_grid", "threat_result",
            "depth_info", "elapsed_ms", "saved_paths"
        ]
        for k in expected_keys:
            self.assertIn(k, results, f"Missing key in process_image results: {k}")

        # 2. Output shapes
        H, W = results["original_frame"].shape[:2]
        self.assertEqual(results["annotated_frame"].shape, (H, W, 3))
        self.assertEqual(results["enhanced_frame"].shape, (H, W, 3))
        self.assertEqual(results["seg_mask"].shape, (H, W))

        # 3. Threat result consistency
        threat = results["threat_result"]
        self.assertIsInstance(threat, ThreatResult)
        self.assertIn(threat.threat_level, ["SAFE", "WARNING", "CRITICAL", "UNKNOWN / LOW CONFIDENCE"])
        self.assertIn(threat.relative_depth, ["Near", "Medium", "Far", "Not Available"])
        self.assertEqual(threat.motion_level, "N/A", "Image mode must report motion as N/A")

        # 4. Artifact verification on disk
        saved_paths = results["saved_paths"]
        for name, path in saved_paths.items():
            self.assertTrue(os.path.isfile(path), f"Saved artifact missing: {name} -> {path}")
            self.assertGreater(os.path.getsize(path), 0, f"Artifact file is empty: {path}")

        print("  [PASS] Test 1: Image mode outputs, relative depth, and saved artifacts verified.")

    def test_2_video_mode_delegation(self):
        """Video mode: runs through canonical process_frame engine and produces valid output video."""
        out_video_path = os.path.join(_PROJECT_ROOT, "outputs", "regression_test_video", "reg_video.mp4")
        os.makedirs(os.path.dirname(out_video_path), exist_ok=True)

        frames_to_run = 8
        results = self.pipeline.process_video(
            video_path=self.test_video_path,
            output_video_path=out_video_path,
            max_frames=frames_to_run,
        )

        self.assertIn("output_video_path", results)
        self.assertEqual(results["frames_processed"], frames_to_run)
        self.assertTrue(os.path.isfile(out_video_path))
        self.assertGreater(os.path.getsize(out_video_path), 1000, "Output video too small or corrupt")
        self.assertGreater(results["actual_fps"], 0.0)
        self.assertIsNotNone(results["overall_threat"])

        print(f"  [PASS] Test 2: Video mode processed {frames_to_run} frames at {results['actual_fps']:.2f} FPS.")

    def test_3_canonical_process_frame_engine_and_cadence(self):
        """process_frame: single-frame engine executes correct cadences and caches."""
        cap = cv2.VideoCapture(self.test_video_path)
        ret, frame_bgr = cap.read()
        cap.release()
        self.assertTrue(ret and frame_bgr is not None, "Could not read frame from test video")

        # Reset session
        self.pipeline.reset()

        # Frame 0: All models active (U-Net, Det, Depth updated)
        f0_res = self.pipeline.process_frame(frame_bgr, frame_idx=0)
        self.assertTrue(f0_res["run_unet"])
        self.assertTrue(f0_res["run_det"])
        self.assertTrue(f0_res["run_depth"])
        self.assertEqual(f0_res["depth_info"]["status"], "UPDATED")
        self.assertIn("annotated_frame", f0_res)
        self.assertEqual(f0_res["annotated_frame"].shape[:2], frame_bgr.shape[:2])

        # Frame 1: Intermediate skipped frame (U-Net, Det, Depth all reused / skipped)
        f1_res = self.pipeline.process_frame(frame_bgr, frame_idx=1)
        self.assertFalse(f1_res["run_unet"])
        self.assertFalse(f1_res["run_det"])
        self.assertFalse(f1_res["run_depth"])
        self.assertEqual(f1_res["depth_info"]["status"], "REUSED")
        self.assertIn("REUSED", f1_res["depth_info"]["note"])

        # Frame 2: U-Net interval = 2, so U-Net runs; Det (interval 3) and Depth (interval 5) skipped
        f2_res = self.pipeline.process_frame(frame_bgr, frame_idx=2)
        self.assertTrue(f2_res["run_unet"])
        self.assertFalse(f2_res["run_det"])
        self.assertFalse(f2_res["run_depth"])

        # Frame 3: Detection interval = 3, so Det runs; U-Net and Depth skipped
        f3_res = self.pipeline.process_frame(frame_bgr, frame_idx=3)
        self.assertFalse(f3_res["run_unet"])
        self.assertTrue(f3_res["run_det"])
        self.assertFalse(f3_res["run_depth"])

        # Frame 5: Depth interval = 5, so Depth runs
        f5_res = self.pipeline.process_frame(frame_bgr, frame_idx=5)
        self.assertTrue(f5_res["run_depth"])
        self.assertEqual(f5_res["depth_info"]["status"], "UPDATED")

        print("  [PASS] Test 3: Canonical process_frame cadence (intervals 2, 3, 5) verified.")

    def test_4_empty_detections_fallback(self):
        """Empty detections: when no animal is present, never fabricate labels or fake confidences."""
        # Solid flat background frame where no camouflage and no animal exists
        flat_frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)

        self.pipeline.reset()
        res = self.pipeline.process_frame(flat_frame, frame_idx=0)

        threat = res["threat_result"]
        # If no animal detected and no camouflage candidate, must be UNKNOWN / LOW CONFIDENCE with 0 confidence
        # or SAFE if confidence is 0.
        self.assertIn(threat.threat_level, ["UNKNOWN / LOW CONFIDENCE", "SAFE"])
        self.assertIn(threat.animal_class, ["UNKNOWN", "UNKNOWN / LOW CONFIDENCE"])
        if threat.animal_class == "UNKNOWN":
            self.assertEqual(threat.confidence, 0.0)
        else:
            self.assertGreaterEqual(threat.confidence, 0.0)
            self.assertLessEqual(threat.confidence, 1.0)

        # Verify no fabricated species names
        forbidden_labels = ["tiger", "lion", "elephant", "bear", "cheetah"]
        self.assertNotIn(threat.animal_class.lower(), forbidden_labels)

        print(f"  [PASS] Test 4: Empty detection fallback verified (class: '{threat.animal_class}', no fabricated labels).")

    def test_5_unknown_animal_candidate_fallback(self):
        """U-Net candidate fallback: candidate region gets UNKNOWN / LOW CONFIDENCE with measured peak."""
        # Frame with a high-contrast localized camouflaged patch
        patch_frame = np.full((540, 960, 3), 100, dtype=np.uint8)
        # Add high-frequency textured patch in center
        np.random.seed(42)
        patch_frame[200:300, 400:500] = np.random.randint(40, 220, (100, 100, 3), dtype=np.uint8)

        # Directly test _detect_camouflaged_wildlife
        seg_mask = np.zeros((540, 960), dtype=np.float32)
        seg_mask[200:300, 400:500] = 0.85  # Strong camouflage signal
        binary_mask = (seg_mask >= 0.5).astype(np.uint8) * 255

        detections = self.pipeline._detect_camouflaged_wildlife(
            frame_bgr=patch_frame,
            enhanced_bgr=patch_frame,
            seg_mask=seg_mask,
            binary_mask=binary_mask,
            flow_map=None,
        )

        self.assertGreater(len(detections), 0, "U-Net candidate fallback did not trigger")
        box, conf, cls_name = detections[0]
        self.assertEqual(cls_name, "UNKNOWN / LOW CONFIDENCE")
        self.assertAlmostEqual(conf, 0.85, delta=0.01)

        print("  [PASS] Test 5: Camouflage candidate fallback produced UNKNOWN / LOW CONFIDENCE at measured peak.")

    def test_6_stationary_animal_handling(self):
        """Stationary animals: consecutive identical frames produce near-zero flow without corrupting fusion."""
        cap = cv2.VideoCapture(self.test_video_path)
        ret, frame_bgr = cap.read()
        cap.release()
        self.assertTrue(ret)

        self.pipeline.reset()
        res1 = self.pipeline.process_frame(frame_bgr, frame_idx=0)
        # Second identical frame -> optical flow between identical frames is 0
        res2 = self.pipeline.process_frame(frame_bgr, frame_idx=1)

        flow2 = res2["flow_map"]
        self.assertIsNotNone(flow2)
        mean_motion = float(np.mean(flow2))
        self.assertLess(mean_motion, 0.05, f"Stationary identical frame motion should be near 0, got {mean_motion}")
        self.assertEqual(res2["threat_result"].motion_level, "LOW")

        print(f"  [PASS] Test 6: Stationary frame motion correctly measured as LOW ({mean_motion:.4f}).")

    def test_7_temporal_state_resets(self):
        """Temporal reset: reset() thoroughly purges optical flow, tracker, and session caches."""
        cap = cv2.VideoCapture(self.test_video_path)
        ret, frame = cap.read()
        cap.release()

        # Run 3 frames to build state
        self.pipeline.process_frame(frame, frame_idx=0)
        self.pipeline.process_frame(frame, frame_idx=1)
        self.pipeline.process_frame(frame, frame_idx=2)

        # Verify state is populated
        self.assertIsNotNone(self.pipeline.camo_processor._flow_module._prev_gray)
        self.assertIsNotNone(self.pipeline._session.cached_depth_info)
        self.assertGreater(self.pipeline._session.frame_idx, 0)

        # Call reset
        self.pipeline.reset()

        # Verify state is purged
        self.assertIsNone(self.pipeline.camo_processor._flow_module._prev_gray)
        self.assertEqual(len(self.pipeline.tracker.track_history), 0)
        self.assertEqual(len(self.pipeline.tracker.track_class_history), 0)
        self.assertIsNone(self.pipeline._session.cached_depth_info)
        self.assertIsNone(self.pipeline._session.cached_depth_raw)
        self.assertEqual(self.pipeline._session.frame_idx, 0)
        self.assertEqual(self.pipeline._session.last_depth_frame, 0)

        print("  [PASS] Test 7: Temporal state reset thoroughly verified across all stages.")

    def test_8_coordinate_scaling_consistency(self):
        """Coordinate scaling: consistent mapping between 1080p native, 540p intermediate, and depth boxes."""
        frame_1080p = np.zeros((1080, 1920, 3), dtype=np.uint8)

        self.pipeline.reset()
        res = self.pipeline.process_frame(frame_1080p, frame_idx=0)

        # 1. Working frame scaling
        self.assertEqual(res["proc_frame"].shape[:2], (540, 960))
        self.assertEqual(res["scale_x"], 1920.0 / 960.0)
        self.assertEqual(res["scale_y"], 1080.0 / 540.0)

        # 2. Output frames match native 1080p
        self.assertEqual(res["annotated_frame"].shape[:2], (1080, 1920))
        self.assertEqual(res["enhanced_frame"].shape[:2], (1080, 1920))

        # 3. ZoeDepth bounding box mapping
        # A native 1080p box [200, 100, 400, 300] must scale to [100, 50, 200, 150] in 540p
        native_bbox = [200, 100, 400, 300]
        sx, sy = res["scale_x"], res["scale_y"]
        scaled_depth_box = [
            int(native_bbox[0] / sx),
            int(native_bbox[1] / sy),
            int(native_bbox[2] / sx),
            int(native_bbox[3] / sy),
        ]
        self.assertEqual(scaled_depth_box, [100, 50, 200, 150])

        print(f"  [PASS] Test 8: Coordinate scaling (sx={sx:.1f}, sy={sy:.1f}) verified.")


if __name__ == "__main__":
    unittest.main()
