"""
scripts/test_phone_and_live_stream.py
=====================================
Automated verification suite for Phone Camera Streaming, Laptop Dashboard,
and Real-Time Threat Alerts.

Tests:
    1. PhoneStreamManager session lifecycle (start, stop, isolation).
    2. Phone frame processing through canonical process_frame() engine on GPU.
    3. Structured analysis payload integrity (species, relative depth, threat).
    4. Threat alert trigger logic (new target detection, threat escalation).
    5. MJPEG stream generation for laptop dashboard.
    6. Concurrency locking & mutual exclusion (Phone stream vs Webcam vs Offline Video).
    7. Flask API endpoints integration (/phone, /phone/start, /phone/stream_frame, /phone/telemetry, /phone/stop, /api/network_info).
"""

import os
import sys
import time
import unittest
import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from modules.pipeline import WildlifePipeline
from modules.phone_stream import PhoneStreamManager
from modules.webcam import WebcamManager
from app import app, pipeline, phone_stream_manager, webcam_manager


class TestPhoneAndLiveStream(unittest.TestCase):
    """Test suite for live phone-camera integration and alert system."""

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print("RUNNING PHONE STREAM & LIVE MONITORING TEST SUITE")
        print("=" * 70)
        cls.pipeline = pipeline
        cls.phone_mgr = phone_stream_manager
        cls.webcam_mgr = webcam_manager

        # Test sample frame
        test_img_path = os.path.join(_PROJECT_ROOT, "data", "test_image.jpg")
        if os.path.isfile(test_img_path):
            cls.test_bgr = cv2.imread(test_img_path)
        else:
            cls.test_bgr = np.full((480, 640, 3), 120, dtype=np.uint8)

        # Encode sample frame to JPEG bytes
        _, buf = cv2.imencode(".jpg", cls.test_bgr)
        cls.test_jpeg_bytes = buf.tobytes()

        # Flask test client
        cls.flask_client = app.test_client()

    def setUp(self):
        self.phone_mgr.stop_session()
        self.webcam_mgr.stop()
        self.pipeline.reset()

    def test_1_phone_session_lifecycle(self):
        """Test starting, stopping, and re-initializing a phone session."""
        ok, msg, sid = self.phone_mgr.start_session(client_ip="192.168.0.50")
        self.assertTrue(ok)
        self.assertTrue(self.phone_mgr.is_active)
        self.assertEqual(self.phone_mgr.client_ip, "192.168.0.50")
        self.assertIsNotNone(sid)

        # Re-starting without stopping should reject second simultaneous session
        ok2, msg2, _ = self.phone_mgr.start_session(client_ip="192.168.0.99")
        self.assertFalse(ok2, "Must reject second simultaneous phone stream")

        # Stop session
        stop_ok, stop_msg = self.phone_mgr.stop_session(session_id=sid)
        self.assertTrue(stop_ok)
        self.assertFalse(self.phone_mgr.is_active)
        print("  [PASS] Test 1: Phone session start/stop and single-session lock verified.")

    def test_2_phone_frame_processing_canonical_engine(self):
        """Test processing phone camera frame through canonical pipeline."""
        ok, msg, sid = self.phone_mgr.start_session()
        self.assertTrue(ok)

        # Process frame
        res = self.phone_mgr.process_incoming_frame(self.test_jpeg_bytes, session_id=sid)

        self.assertEqual(res["status"], "success")
        self.assertIn("threat", res)
        self.assertIn("tracked_objects", res)
        self.assertIn("fps", res)
        self.assertIn("elapsed_ms", res)

        threat = res["threat"]
        self.assertIn(threat["threat_level"], ["SAFE", "WARNING", "CRITICAL", "UNKNOWN / LOW CONFIDENCE"])
        self.assertIn(threat["relative_depth"], ["Near", "Medium", "Far", "Not Available"])
        self.assertGreater(res["elapsed_ms"], 0.0)

        # Confirm annotated frame is generated for dashboard
        self.assertIsNotNone(self.phone_mgr._latest_annotated_frame)
        self.assertEqual(self.phone_mgr.frames_processed, 1)
        print(f"  [PASS] Test 2: Frame processed in {res['elapsed_ms']:.1f}ms ({threat['threat_level']}, depth={threat['relative_depth']}).")

    def test_3_alert_event_generation_and_deduplication(self):
        """Test that state machine emits alerts on new detections and escalations with cooldowns."""
        from modules.threat_analysis import ThreatResult

        now = time.time()
        # Simulated critical threat
        crit_threat = ThreatResult(
            threat_level="CRITICAL",
            threat_score=0.90,
            animal_class="snow_leopard",
            confidence=0.88,
            track_id=1,
            motion_magnitude=0.5,
            motion_level="HIGH",
            relative_depth="Near",
            depth_note="Valid",
            reason="High-risk predator approaching",
            color_bgr=(0, 0, 255),
        )

        # First evaluation should trigger THREAT_ESCALATION
        alert1 = self.phone_mgr._evaluate_alert_conditions(
            threat=crit_threat,
            tracked_objects=[{"track_id": 1, "class_name": "snow_leopard", "confidence": 0.88, "persistence": 1}],
            now=now,
        )

        self.assertIsNotNone(alert1)
        self.assertEqual(alert1["severity"], "CRITICAL")
        self.assertEqual(alert1["sound"], "CRITICAL")

        # Second immediate frame (no state change) should be suppressed by cooldown
        alert2 = self.phone_mgr._evaluate_alert_conditions(
            threat=crit_threat,
            tracked_objects=[{"track_id": 1, "class_name": "snow_leopard", "confidence": 0.88, "persistence": 2}],
            now=now + 0.2,
        )
        self.assertIsNone(alert2, "Cooldown must prevent alert spamming on consecutive frames")
        print("  [PASS] Test 3: Alert event trigger and cooldown deduplication verified.")

    def test_4_dashboard_mjpeg_stream(self):
        """Test MJPEG stream generator yields valid JPEG multipart frames."""
        self.phone_mgr.start_session()
        self.phone_mgr.process_incoming_frame(self.test_jpeg_bytes)

        gen = self.phone_mgr.generate_mjpeg_stream()
        chunk = next(gen)

        self.assertIn(b"--frame\r\n", chunk)
        self.assertIn(b"Content-Type: image/jpeg\r\n\r\n", chunk)
        print("  [PASS] Test 4: Dashboard MJPEG generator yields multipart JPEG frames.")

    def test_5_concurrency_protection(self):
        """Test mutual exclusion: active phone stream blocks conflicting operations."""
        self.phone_mgr.start_session()

        # 1. Image upload should return 409
        resp = self.flask_client.post("/analyze/image", data={})
        self.assertEqual(resp.status_code, 409)

        # 2. Video upload should return 409
        resp = self.flask_client.post("/analyze/video", data={})
        self.assertEqual(resp.status_code, 409)

        # 3. Webcam start should return 409
        resp = self.flask_client.post("/webcam/start", json={"camera_index": 0})
        self.assertEqual(resp.status_code, 409)

        # Once phone stops, image/video access restored
        self.phone_mgr.stop_session()
        resp = self.flask_client.post("/analyze/image", data={})
        self.assertEqual(resp.status_code, 400)  # 400: no image file provided (not 409)
        print("  [PASS] Test 5: Concurrency locks and mutual exclusion verified.")

    def test_6_flask_phone_endpoints_integration(self):
        """Test Flask HTTP routes for mobile phone capture."""
        # 1. /phone UI route
        resp = self.flask_client.get("/phone")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Phone Camera Stream", resp.data)

        # 2. /api/network_info
        resp = self.flask_client.get("/api/network_info")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("phone_url", data)
        self.assertIn("primary_ip", data)

        # 3. /phone/start
        resp = self.flask_client.post("/phone/start")
        self.assertEqual(resp.status_code, 200)

        # 4. /phone/stream_frame
        resp = self.flask_client.post(
            "/phone/stream_frame",
            data=self.test_jpeg_bytes,
            content_type="image/jpeg",
        )
        self.assertEqual(resp.status_code, 200)
        frame_res = resp.get_json()
        self.assertEqual(frame_res["status"], "success")

        # 5. /phone/telemetry
        resp = self.flask_client.get("/phone/telemetry")
        self.assertEqual(resp.status_code, 200)
        telem = resp.get_json()
        self.assertTrue(telem["is_active"])
        self.assertFalse(telem["is_stale"])

        # 6. /phone/stop
        resp = self.flask_client.post("/phone/stop")
        self.assertEqual(resp.status_code, 200)

        print("  [PASS] Test 6: All Flask phone streaming and telemetry endpoints verified.")


if __name__ == "__main__":
    unittest.main()
