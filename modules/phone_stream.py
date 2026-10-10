"""
modules/phone_stream.py
=======================
Real-Time Phone-Camera Stream and Threat Alert Notification Manager.

Complies with Phase 5, Phase 6, Phase 7, Phase 8 requirements:
1. Reuses canonical WildlifePipeline.process_frame() on laptop GPU.
2. Receives live frames captured by phone browser camera over local network.
3. Bounded queue / latest-frame execution to eliminate cumulative network latency.
4. Returns real-time structured analysis, species classification, relative depth,
   and threat assessment back to the phone client.
5. Emits state-transition alert events for new targets, species identification,
   and threat escalations (with cooldowns to prevent alert fatigue).
6. Serves multipart MJPEG live feed and telemetry to the Laptop Dashboard.
7. Maintains session isolation and safe cleanup on disconnect.
"""

import os
import sys
import time
import uuid
import threading
from typing import Dict, Any, Optional, Tuple, List, Generator
import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from modules.pipeline import WildlifePipeline
from modules.threat_analysis import ThreatResult


class PhoneStreamManager:
    """
    Manages live streaming from a phone camera to the laptop backend,
    orchestrating real-time GPU inference, laptop dashboard feed,
    and structured threat alerts delivered back to the phone.
    """

    def __init__(self, pipeline: WildlifePipeline) -> None:
        self.pipeline = pipeline
        self.lock = threading.Lock()

        # Session state
        self.is_active: bool = False
        self.session_id: Optional[str] = None
        self.client_ip: Optional[str] = None
        self.start_time: float = 0.0
        self.last_frame_time: float = 0.0
        self.frame_counter: int = 0

        # Frame buffers
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._latest_analysis: Dict[str, Any] = {}
        self._is_processing: bool = False

        # Alert state machine
        self._known_track_ids: set = set()
        self._last_threat_level: str = "UNKNOWN"
        self._last_alert_time: float = 0.0
        self._alert_cooldown: float = 4.0  # seconds between repeated alerts of same level
        self._recent_alerts: List[Dict[str, Any]] = []

        # Performance telemetry
        self.fps_rolling: float = 0.0
        self.latency_ms: float = 0.0
        self.frames_received: int = 0
        self.frames_processed: int = 0
        self.frames_dropped: int = 0

        # Default placeholder frame for dashboard viewer
        self._placeholder_frame = self._create_placeholder("AWAITING PHONE CAMERA STREAM...")

    @staticmethod
    def _create_placeholder(message: str, width: int = 640, height: int = 480) -> np.ndarray:
        """Generate a dark placeholder frame."""
        frame = np.full((height, width, 3), 18, dtype=np.uint8)
        cv2.rectangle(frame, (16, 16), (width - 16, height - 16), (40, 40, 40), 2)
        (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        cv2.putText(
            frame,
            message,
            ((width - tw) // 2, (height + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (170, 170, 170),
            2,
            cv2.LINE_AA,
        )
        return frame

    def start_session(self, client_ip: str = "127.0.0.1") -> Tuple[bool, str, str]:
        """
        Start an active phone streaming session.
        Enforces single active session and resets temporal pipeline state.
        """
        with self.lock:
            if self.is_active:
                # If existing session hasn't sent frames in >6 seconds, force reclaim
                if time.time() - self.last_frame_time > 6.0:
                    self._stop_internal()
                else:
                    return False, f"A live phone session ({self.session_id}) is already active from {self.client_ip}.", self.session_id or ""

            session_id = str(uuid.uuid4())[:8]
            self.session_id = session_id
            self.client_ip = client_ip
            self.start_time = time.time()
            self.last_frame_time = time.time()
            self.frame_counter = 0

            self.frames_received = 0
            self.frames_processed = 0
            self.frames_dropped = 0
            self.fps_rolling = 0.0
            self.latency_ms = 0.0

            self._known_track_ids.clear()
            self._last_threat_level = "UNKNOWN"
            self._last_alert_time = 0.0
            self._recent_alerts.clear()

            # Clean reset of pipeline session (optical flow, tracker, depth cache)
            self.pipeline.reset()

            self.is_active = True
            print(f"[PhoneStreamManager] Started session {session_id} from {client_ip}.")
            return True, f"Session {session_id} started successfully.", session_id

    def stop_session(self, session_id: Optional[str] = None) -> Tuple[bool, str]:
        """Stop phone streaming session and release resources."""
        with self.lock:
            if not self.is_active:
                return True, "No active phone session to stop."
            if session_id and self.session_id and session_id != self.session_id:
                return False, f"Session ID mismatch: active session is {self.session_id}."

            return self._stop_internal()

    def _stop_internal(self) -> Tuple[bool, str]:
        """Internal worker to release session resources."""
        sid = self.session_id
        self.is_active = False
        self.session_id = None
        self.client_ip = None
        self._latest_raw_frame = None
        self._latest_annotated_frame = self._create_placeholder("PHONE STREAM DISCONNECTED")
        self._latest_analysis = {}

        # Reset pipeline temporal state
        self.pipeline.reset()
        print(f"[PhoneStreamManager] Session {sid} stopped and pipeline reset.")
        return True, f"Session {sid} stopped successfully."

    def process_incoming_frame(
        self,
        frame_bytes: bytes,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single video frame sent by the mobile browser.
        Runs canonical process_frame() on GPU, triggers alerts, and returns structured data.
        """
        now = time.time()
        self.last_frame_time = now
        self.frames_received += 1

        if not self.is_active:
            # Auto-activate if no session explicitly started yet
            self.is_active = True
            self.session_id = session_id or str(uuid.uuid4())[:8]
            self.start_time = now
            self.pipeline.reset()

        # Decode JPEG bytes to OpenCV BGR image
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return {
                "status": "error",
                "message": "Invalid frame image data could not be decoded.",
            }

        # Resolution normalization: Cap frame resolution to prevent GPU OOM
        # Mobile phones can send 4K or 1080p frames; resize to max 960 width for smooth inference
        H, W = frame_bgr.shape[:2]
        if W > 960 or H > 720:
            scale = min(960.0 / W, 720.0 / H)
            new_w = int(W * scale)
            new_h = int(H * scale)
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        self._latest_raw_frame = frame_bgr

        # Concurrency safety: If another inference is in flight, drop frame to eliminate queue lag
        if self._is_processing:
            self.frames_dropped += 1
            # Return last known analysis so phone stays informed
            return {
                "status": "dropped",
                "message": "Frame dropped to prevent lag; GPU inference in flight.",
                "analysis": self._latest_analysis,
            }

        self._is_processing = True
        t0 = time.perf_counter()

        try:
            # Run canonical WildlifePipeline.process_frame() on GPU
            results = self.pipeline.process_frame(
                frame_bgr=frame_bgr,
                frame_idx=self.frame_counter,
                return_viz=True,
            )

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            self.latency_ms = elapsed_ms
            self.frame_counter += 1
            self.frames_processed += 1

            # Update rolling FPS
            inst_fps = 1000.0 / max(elapsed_ms, 1.0)
            self.fps_rolling = 0.85 * self.fps_rolling + 0.15 * inst_fps if self.fps_rolling > 0 else inst_fps

            self._latest_annotated_frame = results["annotated_frame"]

            # Threat analysis parsing
            threat: ThreatResult = results["threat_result"]
            conf_display = (
                f"Camo Score: {threat.confidence * 100:.1f}%"
                if "UNKNOWN" in threat.animal_class.upper() and threat.confidence > 0
                else (f"{threat.confidence * 100:.1f}%" if threat.confidence > 0 else "LOW")
            )

            threat_dict = {
                "threat_level": threat.threat_level,
                "threat_score": threat.threat_score,
                "animal_class": threat.animal_class,
                "confidence": threat.confidence,
                "confidence_display": conf_display,
                "motion_level": threat.motion_level,
                "relative_depth": threat.relative_depth,
                "reason": threat.reason,
                "depth_note": threat.depth_note,
            }

            tracked_objects = results.get("tracked_objects", [])

            # Alert event evaluation (Phase 7 & Phase 8)
            alert_event = self._evaluate_alert_conditions(
                threat=threat,
                tracked_objects=tracked_objects,
                now=now,
            )

            response_payload = {
                "status": "success",
                "session_id": self.session_id,
                "frame_idx": self.frame_counter,
                "elapsed_ms": round(elapsed_ms, 1),
                "fps": round(self.fps_rolling, 1),
                "threat": threat_dict,
                "tracked_objects": [
                    {
                        "track_id": t["track_id"],
                        "class_name": t["class_name"],
                        "confidence": round(t["confidence"], 2),
                        "persistence": t["persistence"],
                    }
                    for t in tracked_objects
                ],
                "cadence": {
                    "unet": results.get("run_unet", False),
                    "yolo": results.get("run_det", False),
                    "depth": results.get("run_depth", False),
                },
                "alert_event": alert_event,
                "timestamp": time.strftime("%H:%M:%S", time.localtime(now)),
            }

            self._latest_analysis = response_payload
            return response_payload

        finally:
            self._is_processing = False

    def _evaluate_alert_conditions(
        self,
        threat: ThreatResult,
        tracked_objects: List[Dict[str, Any]],
        now: float,
    ) -> Optional[Dict[str, Any]]:
        """
        State-transition and cooldown logic for wildlife threat notifications.
        Triggers alerts on new targets, species confirmation, and threat escalations.
        """
        alert_event = None
        current_level = threat.threat_level
        is_unknown = "UNKNOWN" in threat.animal_class.upper()

        # 1. Check for newly detected track IDs
        new_track_ids = []
        for t in tracked_objects:
            tid = t["track_id"]
            if tid not in self._known_track_ids:
                self._known_track_ids.add(tid)
                new_track_ids.append(tid)

        # 2. Check for threat escalation
        threat_levels_order = ["SAFE", "UNKNOWN / LOW CONFIDENCE", "WARNING", "CRITICAL"]
        old_idx = threat_levels_order.index(self._last_threat_level) if self._last_threat_level in threat_levels_order else 0
        new_idx = threat_levels_order.index(current_level) if current_level in threat_levels_order else 0
        is_escalation = new_idx > old_idx and current_level in ("WARNING", "CRITICAL")

        # 3. Formulate alert message if threshold met
        time_since_last = now - self._last_alert_time

        if is_escalation:
            # Immediate alert on threat escalation (bypasses cooldown)
            self._last_alert_time = now
            self._last_threat_level = current_level
            alert_event = {
                "event_id": str(uuid.uuid4())[:8],
                "type": "THREAT_ESCALATION",
                "severity": current_level,
                "title": f"THREAT ALERT: {current_level}",
                "message": f"Threat escalated to {current_level} for {threat.animal_class}. {threat.reason}",
                "animal": threat.animal_class,
                "depth": threat.relative_depth,
                "sound": current_level,  # "WARNING" or "CRITICAL"
                "timestamp": time.strftime("%H:%M:%S", time.localtime(now)),
            }
        elif new_track_ids and current_level != "SAFE" and time_since_last >= self._alert_cooldown:
            # New target detected
            self._last_alert_time = now
            self._last_threat_level = current_level
            t_name = "Camouflaged Target" if is_unknown else threat.animal_class.title()
            alert_event = {
                "event_id": str(uuid.uuid4())[:8],
                "type": "NEW_TARGET",
                "severity": current_level,
                "title": f"Target Identified: {t_name}",
                "message": f"New wildlife track #{new_track_ids[0]} detected at {threat.relative_depth} relative depth.",
                "animal": threat.animal_class,
                "depth": threat.relative_depth,
                "sound": current_level if current_level in ("WARNING", "CRITICAL") else "SAFE",
                "timestamp": time.strftime("%H:%M:%S", time.localtime(now)),
            }
        elif current_level == "CRITICAL" and time_since_last >= 8.0:
            # Recurring critical alert reminder
            self._last_alert_time = now
            alert_event = {
                "event_id": str(uuid.uuid4())[:8],
                "type": "CRITICAL_PERSISTENCE",
                "severity": "CRITICAL",
                "title": "CRITICAL THREAT ACTIVE",
                "message": f"Persistent critical target {threat.animal_class} ({threat.relative_depth} relative depth).",
                "animal": threat.animal_class,
                "depth": threat.relative_depth,
                "sound": "CRITICAL",
                "timestamp": time.strftime("%H:%M:%S", time.localtime(now)),
            }
        else:
            self._last_threat_level = current_level

        if alert_event:
            self._recent_alerts.append(alert_event)
            if len(self._recent_alerts) > 20:
                self._recent_alerts.pop(0)

        return alert_event

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """
        Yield multipart MJPEG stream of the phone camera frames with
        pipeline annotations for the Laptop Monitoring Dashboard.
        """
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

        try:
            while True:
                # Check stale condition: If phone hasn't sent frames in >4s
                is_stale = (time.time() - self.last_frame_time) > 4.0 if self.is_active else True

                if not self.is_active or is_stale:
                    frame = self._placeholder_frame
                else:
                    frame = self._latest_annotated_frame if self._latest_annotated_frame is not None else self._placeholder_frame

                ret, buffer = cv2.imencode(".jpg", frame, encode_params)
                if not ret:
                    time.sleep(0.04)
                    continue

                jpeg_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )

                time.sleep(0.04)  # ~25 FPS stream transmit rate
        except GeneratorExit:
            pass
        except Exception as e:
            print(f"[PhoneStreamManager] Dashboard client disconnected: {e}")

    def get_telemetry(self) -> Dict[str, Any]:
        """Return real-time telemetry dictionary for monitoring dashboard."""
        now = time.time()
        is_stale = (now - self.last_frame_time) > 3.5 if self.is_active else True
        conn_status = "Active" if (self.is_active and not is_stale) else ("Stale" if self.is_active else "Disconnected")

        return {
            "is_active": self.is_active,
            "session_id": self.session_id,
            "client_ip": self.client_ip,
            "connection_status": conn_status,
            "is_stale": is_stale,
            "fps": round(self.fps_rolling, 1),
            "latency_ms": round(self.latency_ms, 1),
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "analysis": self._latest_analysis,
            "recent_alerts": self._recent_alerts[-5:] if self._recent_alerts else [],
            "last_frame_age_s": round(now - self.last_frame_time, 1) if self.last_frame_time > 0 else 999.0,
        }
