"""
app.py
======
Wildlife-COD Web Application (Flask).
Single command entry point to run the complete end-to-end prototype:
    python app.py

Allows uploading an Image or Video, runs the unified 7-component architecture:
    1. U-Net
    2. Optical Flow
    3. Fusion
    4. YOLOv8
    5. DeepSORT
    6. ZoeDepth
    7. Rule-based Threat Analysis

Visualises all 4 stages simultaneously with authoritative Threat Level indicators.
"""

import os
import sys
import uuid
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
import cv2
from PIL import Image

# Ensure project root in python path
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

import config
from modules.pipeline import WildlifePipeline
from modules.webcam import WebcamManager
from modules.phone_stream import PhoneStreamManager

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(_BASE_DIR, "uploads")
app.config["OUTPUT_FOLDER"] = os.path.join(_BASE_DIR, "outputs")
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250 MB max upload

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# Initialize pipeline once on startup
active_yolo = (
    config.YOLO_WORLD_WEIGHTS
    if getattr(config, "YOLO_MODE", "world") == "world"
    else config.YOLO_WILDLIFE_WEIGHTS
)

print("=" * 60)
print(f"Initializing Wildlife-COD Pipeline (Detector Mode: {getattr(config, 'YOLO_MODE', 'world')})...")
print("=" * 60)
pipeline = WildlifePipeline(
    checkpoint_path=config.BEST_MODEL_PATH,
    yolo_weights=active_yolo,
)
print("=" * 60)
print("Pipeline ready! Initializing Stream Managers...")
print("=" * 60)
webcam_manager = WebcamManager(pipeline=pipeline)
phone_stream_manager = PhoneStreamManager(pipeline=pipeline)
print("Starting Flask Server...")
print("=" * 60)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/outputs/<path:filename>")
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


@app.route("/analyze/image", methods=["POST"])
def analyze_image():
    if webcam_manager.is_running or phone_stream_manager.is_active:
        return jsonify({"error": "Live camera/phone stream is currently active. Please stop live stream before analyzing images."}), 409

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    unique_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    upload_filename = f"img_{unique_id}{ext}"
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_filename)
    file.save(upload_path)

    # Convert formats OpenCV cannot read (avif, webp, heic, heif, tiff, bmp) to JPEG
    _OPENCV_UNSUPPORTED = {".avif", ".heic", ".heif"}
    if ext in _OPENCV_UNSUPPORTED:
        jpg_filename = f"img_{unique_id}.jpg"
        jpg_path = os.path.join(app.config["UPLOAD_FOLDER"], jpg_filename)
        try:
            with Image.open(upload_path) as pil_img:
                pil_img.convert("RGB").save(jpg_path, "JPEG", quality=95)
            upload_path = jpg_path
        except Exception as conv_err:
            return jsonify({"error": f"Unsupported image format '{ext}': {conv_err}"}), 400

    # Output directory for this image
    out_dir = os.path.join(app.config["OUTPUT_FOLDER"], "images", unique_id)
    os.makedirs(out_dir, exist_ok=True)

    try:
        results = pipeline.process_image(upload_path, save_dir=out_dir)

        threat = results["threat_result"]
        conf_display = f"{threat.confidence * 100:.1f}%" if threat.confidence > 0 else "LOW"
        return jsonify({
            "status": "success",
            "elapsed_ms": results["elapsed_ms"],
            "threat": {
                "threat_level": threat.threat_level,
                "threat_score": threat.threat_score,
                "animal_class": threat.animal_class,
                "confidence": threat.confidence,
                "confidence_display": conf_display,
                "motion_level": threat.motion_level,
                "relative_depth": threat.relative_depth,
                "reason": threat.reason,
                "depth_note": threat.depth_note,
            },
            "paths": {
                "original": f"/outputs/images/{unique_id}/original.jpg",
                "mask": f"/outputs/images/{unique_id}/unet_mask.png",
                "enhanced": f"/outputs/images/{unique_id}/enhanced.jpg",
                "depth": f"/outputs/images/{unique_id}/depth_map.jpg",
                "annotated": f"/outputs/images/{unique_id}/annotated_result.jpg",
                "comparison": f"/outputs/images/{unique_id}/comparison_grid.jpg",
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/video", methods=["POST"])
def analyze_video():
    if webcam_manager.is_running or phone_stream_manager.is_active:
        return jsonify({"error": "Live camera/phone stream is currently active. Please stop live stream before analyzing videos."}), 409

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    max_frames = int(request.form.get("max_frames", 60))

    unique_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    upload_filename = f"vid_{unique_id}{ext}"
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_filename)
    file.save(upload_path)

    out_video_dir = os.path.join(app.config["OUTPUT_FOLDER"], "videos")
    os.makedirs(out_video_dir, exist_ok=True)
    out_video_path = os.path.join(out_video_dir, f"enhanced_{unique_id}.mp4")

    try:
        results = pipeline.process_video(
            video_path=upload_path,
            output_video_path=out_video_path,
            max_frames=max_frames,
        )

        overall_threat = results["overall_threat"]

        # Save first sample comparison grid if available
        sample_url = None
        if results["sample_comparisons"]:
            sample_comp_path = os.path.join(out_video_dir, f"sample_{unique_id}.jpg")
            cv2.imwrite(sample_comp_path, results["sample_comparisons"][0])
            sample_url = f"/outputs/videos/sample_{unique_id}.jpg"

        threat_dict = None
        if overall_threat:
            threat_dict = {
                "threat_level": overall_threat.threat_level,
                "threat_score": overall_threat.threat_score,
                "animal_class": overall_threat.animal_class,
                "confidence": overall_threat.confidence,
                "motion_level": overall_threat.motion_level,
                "relative_depth": overall_threat.relative_depth,
                "reason": overall_threat.reason,
                "depth_note": overall_threat.depth_note,
            }

        return jsonify({
            "status": "success",
            "video_url": f"/outputs/videos/enhanced_{unique_id}.mp4",
            "sample_comparison_url": sample_url,
            "frames_processed": results["frames_processed"],
            "actual_fps": results["actual_fps"],
            "total_time_s": results["total_time_s"],
            "overall_threat": threat_dict,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ======================================================================
# Phase 3: Real-Time Laptop Webcam Streaming & Control Routes
# ======================================================================

@app.route("/webcam/devices", methods=["GET"])
def webcam_devices():
    """Probe system camera device indices and return available cameras."""
    devices = webcam_manager.detect_available_cameras()
    return jsonify({
        "status": "success",
        "devices": devices,
        "active_device": webcam_manager.camera_index if webcam_manager.is_running else None,
        "is_running": webcam_manager.is_running,
    })


@app.route("/webcam/start", methods=["POST"])
def webcam_start():
    """Start live webcam capture with camera device selection."""
    if phone_stream_manager.is_active:
        return jsonify({
            "status": "error",
            "message": "Phone camera stream is currently active. Please stop phone stream first."
        }), 409

    data = request.get_json(silent=True) or request.form
    camera_index = 0
    if "camera_index" in data:
        try:
            camera_index = int(data["camera_index"])
        except (ValueError, TypeError):
            camera_index = 0

    success, message = webcam_manager.start(camera_index=camera_index)
    status_code = 200 if success else 400
    return jsonify({
        "status": "success" if success else "error",
        "message": message,
        "camera_index": camera_index,
        "is_running": webcam_manager.is_running,
    }), status_code


@app.route("/webcam/stop", methods=["POST"])
def webcam_stop():
    """Stop live webcam capture and release hardware resources cleanly."""
    success, message = webcam_manager.stop()
    return jsonify({
        "status": "success" if success else "error",
        "message": message,
        "is_running": False,
    })


@app.route("/webcam/status", methods=["GET"])
def webcam_status():
    """Return real-time telemetry metrics and threat analysis for HUD."""
    status = webcam_manager.get_status()
    return jsonify(status)


@app.route("/webcam/stream")
def webcam_stream():
    """Multipart MJPEG stream endpoint for real-time video feed."""
    if not webcam_manager.is_running:
        return jsonify({
            "error": "Webcam is not active. Please start camera before connecting to stream."
        }), 400

    return Response(
        webcam_manager.generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ======================================================================
# Phase 5, 6, 7: Live Phone-Camera Streaming & Threat Alert Routes
# ======================================================================

@app.route("/phone")
def phone_page():
    """Mobile browser camera capture and real-time threat alert interface."""
    return render_template("phone.html")


@app.route("/phone/start", methods=["POST"])
def phone_start():
    """Start an active phone streaming session."""
    if webcam_manager.is_running:
        return jsonify({
            "status": "error",
            "message": "Laptop webcam is currently active. Please stop webcam before starting phone stream."
        }), 409

    success, message, session_id = phone_stream_manager.start_session(client_ip=request.remote_addr or "127.0.0.1")
    status_code = 200 if success else 400
    return jsonify({
        "status": "success" if success else "error",
        "message": message,
        "session_id": session_id,
        "is_active": phone_stream_manager.is_active,
    }), status_code


@app.route("/phone/stop", methods=["POST"])
def phone_stop():
    """Stop the active phone streaming session cleanly."""
    data = request.get_json(silent=True) or request.form
    session_id = data.get("session_id") if data else None
    success, message = phone_stream_manager.stop_session(session_id=session_id)
    return jsonify({
        "status": "success" if success else "error",
        "message": message,
        "is_active": False,
    })


@app.route("/phone/stream_frame", methods=["POST"])
def phone_stream_frame():
    """
    Ingest a single video frame sent by the mobile browser.
    Runs canonical process_frame() on GPU, triggers alerts, and returns structured data.
    """
    if webcam_manager.is_running:
        return jsonify({
            "status": "error",
            "message": "Laptop webcam is active. Stop webcam before streaming from phone."
        }), 409

    frame_data = request.data
    if not frame_data:
        # Check files/form fallback
        if "frame" in request.files:
            frame_data = request.files["frame"].read()
        else:
            return jsonify({"status": "error", "message": "No frame data provided"}), 400

    results = phone_stream_manager.process_incoming_frame(frame_bytes=frame_data)
    return jsonify(results)


@app.route("/phone/feed")
def phone_feed():
    """
    Multipart MJPEG live stream of phone camera frames with
    pipeline overlays for viewing on the Laptop Dashboard.
    """
    return Response(
        phone_stream_manager.generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.route("/phone/telemetry", methods=["GET"])
def phone_telemetry():
    """Return real-time telemetry dictionary for monitoring dashboard."""
    telemetry = phone_stream_manager.get_telemetry()
    return jsonify(telemetry)


@app.route("/api/network_info", methods=["GET"])
def network_info():
    """Return laptop IP address and direct mobile connection link for Wi-Fi pairing."""
    import socket
    hostname = socket.gethostname()
    try:
        ips = [ip for ip in socket.gethostbyname_ex(hostname)[2] if not ip.startswith("127.")]
    except Exception:
        ips = ["127.0.0.1"]

    primary_ip = ips[0] if ips else "127.0.0.1"
    # Select non-virtual LAN IP if multiple exist (prefer 192.168.0.x or 192.168.1.x)
    for ip in ips:
        if not ip.startswith("192.168.56."):
            primary_ip = ip
            break

    port = 5000
    req_scheme = request.scheme or "http"
    return jsonify({
        "hostname": hostname,
        "ip_addresses": ips,
        "primary_ip": primary_ip,
        "port": port,
        "phone_url": f"{req_scheme}://{primary_ip}:{port}/phone",
        "dashboard_url": f"{req_scheme}://{primary_ip}:{port}/",
    })


if __name__ == "__main__":
    import socket
    import argparse
    import ssl

    parser = argparse.ArgumentParser(description="Wildlife-COD Web Application")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--https", action="store_true", help="Run with local SSL certificate (certs/wildlife-local.pem)")
    parser.add_argument("--cert", default=os.path.join(_BASE_DIR, "certs", "wildlife-local.pem"), help="SSL certificate PEM file")
    parser.add_argument("--key", default=os.path.join(_BASE_DIR, "certs", "wildlife-local-key.pem"), help="SSL private key PEM file")
    args = parser.parse_args()

    hostname = socket.gethostname()
    try:
        ips = [ip for ip in socket.gethostbyname_ex(hostname)[2] if not ip.startswith("127.")]
    except Exception:
        ips = ["127.0.0.1"]
    primary_ip = ips[0] if ips else "127.0.0.1"
    for ip in ips:
        if not ip.startswith("192.168.56."):
            primary_ip = ip
            break

    ssl_ctx = None
    if args.https:
        cert_path = os.path.abspath(args.cert)
        key_path = os.path.abspath(args.key)

        missing = []
        if not os.path.isfile(cert_path):
            missing.append(f"Certificate file: {cert_path}")
        if not os.path.isfile(key_path):
            missing.append(f"Private key file: {key_path}")

        if missing:
            print("\n" + "=" * 70)
            print("HTTPS CONFIGURATION ERROR:")
            for m in missing:
                print(f"  Missing: {m}")
            print("=" * 70 + "\n")
            sys.exit(1)

        # Validate that certificate and private key form a valid pair
        try:
            test_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            test_ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
            ssl_ctx = (cert_path, key_path)
        except Exception as e:
            print("\n" + "=" * 70)
            print(f"SSL CERTIFICATE VALIDATION ERROR: {e}")
            print("=" * 70 + "\n")
            sys.exit(1)

    scheme = "https" if ssl_ctx else "http"

    print("\n" + "=" * 70)
    print("WILDLIFE-COD UNIFIED MONITORING & ALERT PLATFORM RUNNING!")
    print("=" * 70)
    print(f"  Protocol         : {scheme.upper()}")
    if ssl_ctx:
        print(f"  SSL Certificate  : {os.path.relpath(cert_path, _BASE_DIR)}")
        print(f"  SSL Private Key  : {os.path.relpath(key_path, _BASE_DIR)}")
    print(f"  Laptop Dashboard : {scheme}://localhost:{args.port}")
    print(f"  Local Network    : {scheme}://{primary_ip}:{args.port}")
    print(f"  Phone Stream URL : {scheme}://{primary_ip}:{args.port}/phone")
    print("=" * 70)
    print("Open the Phone Stream URL on your mobile phone to begin live capture.\n")

    app.run(host=args.host, port=args.port, debug=False, ssl_context=ssl_ctx)
