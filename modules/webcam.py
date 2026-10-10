"""
modules/webcam.py
=================
Thread-Safe Real-Time Webcam Stream Manager for Wildlife Camouflage Detection.

Complies strictly with Phase 3 Requirements:
1. Reuses canonical WildlifePipeline.process_frame() engine without duplicating model inference.
2. Captures frames via OpenCV with dedicated background reader thread.
3. Drops stale frames (latest-frame buffer) so processing never lags behind live webcam capture.
4. Thread-safe single-session lock: prevents simultaneous streams from corrupting shared model state.
5. Clean start/stop handling: resets optical flow, DeepSORT tracks, and ZoeDepth cache on session transitions.
6. Graceful error handling for camera unavailable, device busy, and repeated Start/Stop operations.
7. Produces multipart MJPEG stream generator with JPEG quality control.
8. Real-time telemetry: measures capture FPS, inference FPS, latency, and latest threat status.
"""

import os
import sys
import time
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


class WebcamManager:
    """
    Manages live webcam capture, asynchronous inference, and MJPEG streaming.
    Enforces a strict single active stream session to protect shared pipeline state.
    """

    def __init__(self, pipeline: WildlifePipeline) -> None:
        self.pipeline = pipeline
        self.lock = threading.Lock()

        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_index: int = 0
        self.is_running: bool = False

        # Frame buffers
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._latest_results: Dict[str, Any] = {}
        self._last_processed_frame_id: int = -1
        self._raw_frame_id: int = 0

        # Background threads
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Telemetry metrics
        self.capture_fps: float = 0.0
        self.inference_fps: float = 0.0
        self.latency_ms: float = 0.0
        self.frames_captured: int = 0
        self.frames_processed: int = 0
        self.dropped_frames: int = 0

        # Default placeholder frame for when stream connects before first inference
        self._placeholder_frame = self._create_placeholder("INITIALIZING CAMERA STREAM...")

    @staticmethod
    def _create_placeholder(message: str, width: int = 640, height: int = 480) -> np.ndarray:
        """Create a professional dark placeholder frame."""
        frame = np.full((height, width, 3), 20, dtype=np.uint8)
        cv2.rectangle(frame, (20, 20), (width - 20, height - 20), (45, 45, 45), 2)
        (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(
            frame,
            message,
            ((width - tw) // 2, (height + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        return frame

    @staticmethod
    def detect_available_cameras(max_to_test: int = 3) -> List[int]:
        """Probe system camera device indices and return functional ones."""
        available = []
        for idx in range(max_to_test):
            try:
                # Test opening camera
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(idx)
                    cap.release()
            except Exception:
                pass
        return available if available else [0]

    def start(self, camera_index: int = 0) -> Tuple[bool, str]:
        """
        Start live webcam capture and background inference pipeline.
        Enforces clean state reset and thread-safe session initialization.
        """
        with self.lock:
            if self.is_running:
                if self.camera_index == camera_index and self.cap is not None and self.cap.isOpened():
                    return True, f"Camera {camera_index} is already active."
                # If switching cameras, cleanly stop previous first
                self._stop_internal()

            self.camera_index = camera_index
            self._stop_event.clear()

            # Attempt to open selected camera device
            print(f"[WebcamManager] Opening camera device index {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                # On Windows, try DirectShow fallback if default backend failed
                if sys.platform.startswith("win"):
                    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

            if not cap.isOpened():
                err_msg = (
                    f"Camera device {camera_index} could not be opened. "
                    "Ensure a webcam is plugged in and not in use by another application."
                )
                print(f"[WebcamManager] Error: {err_msg}")
                return False, err_msg

            # Verify initial frame capture
            ret, first_frame = cap.read()
            if not ret or first_frame is None:
                cap.release()
                err_msg = f"Camera device {camera_index} opened but failed to capture frame."
                print(f"[WebcamManager] Error: {err_msg}")
                return False, err_msg

            self.cap = cap
            self._latest_raw_frame = first_frame
            self._raw_frame_id = 1
            self._latest_annotated_frame = first_frame.copy()
            self._last_processed_frame_id = -1

            # Reset all pipeline temporal states and caches for clean camera session
            self.pipeline.reset()

            # Reset telemetry
            self.frames_captured = 1
            self.frames_processed = 0
            self.dropped_frames = 0
            self.capture_fps = 0.0
            self.inference_fps = 0.0
            self.latency_ms = 0.0

            self.is_running = True

            # Launch capture worker thread (reads camera as fast as available)
            self._capture_thread = threading.Thread(
                target=self._capture_worker,
                name="WebcamCaptureWorker",
                daemon=True,
            )
            self._capture_thread.start()

            # Launch inference worker thread (runs canonical process_frame on latest frame)
            self._inference_thread = threading.Thread(
                target=self._inference_worker,
                name="WebcamInferenceWorker",
                daemon=True,
            )
            self._inference_thread.start()

            print(f"[WebcamManager] Camera {camera_index} successfully started ({first_frame.shape[1]}x{first_frame.shape[0]}).")
            return True, f"Camera {camera_index} started successfully."

    def stop(self) -> Tuple[bool, str]:
        """
        Stop camera capture, terminate worker threads, and release hardware resources.
        """
        with self.lock:
            return self._stop_internal()

    def _stop_internal(self) -> Tuple[bool, str]:
        """Internal worker for stop operations (caller holds self.lock)."""
        if not self.is_running and self.cap is None:
            return True, "Camera is not active."

        print("[WebcamManager] Stopping camera capture and releasing pipeline resources...")
        self.is_running = False
        self._stop_event.set()

        # Join threads with timeout
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.5)
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=1.5)

        # Release OpenCV capture
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"[WebcamManager] Warning releasing capture: {e}")
            self.cap = None

        # Reset pipeline state (optical flow, tracker, depth cache)
        self.pipeline.reset()

        self._latest_raw_frame = None
        self._latest_annotated_frame = self._create_placeholder("CAMERA STOPPED")
        self._latest_results = {}

        print("[WebcamManager] Camera successfully stopped and pipeline reset.")
        return True, "Camera stopped successfully."

    def _capture_worker(self) -> None:
        """
        Dedicated camera reading thread.
        Maintains only the latest available raw frame to eliminate queue lag.
        """
        last_t = time.perf_counter()
        fps_smooth = 0.0

        while self.is_running and not self._stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            dt = now - last_t
            last_t = now

            if dt > 0:
                inst_fps = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth > 0 else inst_fps
                self.capture_fps = fps_smooth

            # Store only latest frame
            self._latest_raw_frame = frame
            self._raw_frame_id += 1
            self.frames_captured += 1

            # Prevent CPU thrashing
            time.sleep(0.002)

    def _inference_worker(self) -> None:
        """
        Background inference loop reusing canonical WildlifePipeline.process_frame().
        Runs on latest available frame, skipping superseded intermediate frames.
        """
        last_t = time.perf_counter()
        fps_smooth = 0.0

        while self.is_running and not self._stop_event.is_set():
            frame = self._latest_raw_frame
            cur_id = self._raw_frame_id

            # Wait if no frame or if this frame was already processed
            if frame is None or cur_id == self._last_processed_frame_id:
                time.sleep(0.005)
                continue

            # Record frame skip
            if self._last_processed_frame_id >= 0 and cur_id > self._last_processed_frame_id + 1:
                self.dropped_frames += (cur_id - self._last_processed_frame_id - 1)
            self._last_processed_frame_id = cur_id

            t0 = time.perf_counter()

            try:
                # Call canonical unified process_frame engine
                results = self.pipeline.process_frame(
                    frame_bgr=frame,
                    frame_idx=self.frames_processed,
                    return_viz=True,
                )

                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0
                self.latency_ms = elapsed_ms

                dt = t1 - last_t
                last_t = t1
                if dt > 0:
                    inst_fps = 1.0 / dt
                    fps_smooth = 0.85 * fps_smooth + 0.15 * inst_fps if fps_smooth > 0 else inst_fps
                    self.inference_fps = fps_smooth

                self._latest_annotated_frame = results["annotated_frame"]
                self._latest_results = results
                self.frames_processed += 1

            except Exception as e:
                print(f"[WebcamManager] Inference error on frame: {e}")
                time.sleep(0.01)

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """
        Generator producing multipart/x-mixed-replace JPEG chunks for Flask streaming.
        """
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

        try:
            while self.is_running:
                frame = self._latest_annotated_frame
                if frame is None:
                    frame = self._placeholder_frame

                ret, buffer = cv2.imencode(".jpg", frame, encode_params)
                if not ret:
                    time.sleep(0.03)
                    continue

                jpeg_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )

                # Cap stream transmit rate at ~30 FPS to avoid saturating network sockets
                time.sleep(0.033)

            # Send final stop placeholder when session closes
            ret, buffer = cv2.imencode(".jpg", self._placeholder_frame, encode_params)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
        except GeneratorExit:
            pass
        except Exception as e:
            print(f"[WebcamManager] Client stream interrupted: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Return real-time telemetry dictionary for frontend dashboard."""
        res = self._latest_results
        threat: Optional[ThreatResult] = res.get("threat_result")

        threat_dict = None
        if threat:
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

        tracked_count = len(res.get("tracked_objects", []))
        depth_info = res.get("depth_info", {})

        return {
            "is_running": self.is_running,
            "camera_index": self.camera_index,
            "capture_fps": round(self.capture_fps, 1),
            "inference_fps": round(self.inference_fps, 1),
            "latency_ms": round(self.latency_ms, 1),
            "frames_captured": self.frames_captured,
            "frames_processed": self.frames_processed,
            "dropped_frames": self.dropped_frames,
            "tracked_count": tracked_count,
            "run_unet": res.get("run_unet", False),
            "run_det": res.get("run_det", False),
            "run_depth": res.get("run_depth", False),
            "depth_status": depth_info.get("status", "N/A"),
            "threat": threat_dict,
        }
