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
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2

# Ensure project root in python path
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

import config
from modules.pipeline import WildlifePipeline

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(_BASE_DIR, "uploads")
app.config["OUTPUT_FOLDER"] = os.path.join(_BASE_DIR, "outputs")
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250 MB max upload

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# Initialize pipeline once on startup
print("=" * 60)
print("Initializing Wildlife-COD End-to-End Pipeline...")
print("=" * 60)
pipeline = WildlifePipeline(
    checkpoint_path=config.BEST_MODEL_PATH,
    yolo_weights="yolov8n.pt",
)
print("=" * 60)
print("Pipeline ready! Starting Flask Server...")
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

    # Output directory for this image
    out_dir = os.path.join(app.config["OUTPUT_FOLDER"], "images", unique_id)
    os.makedirs(out_dir, exist_ok=True)

    try:
        results = pipeline.process_image(upload_path, save_dir=out_dir)

        threat = results["threat_result"]
        return jsonify({
            "status": "success",
            "elapsed_ms": results["elapsed_ms"],
            "threat": {
                "threat_level": threat.threat_level,
                "threat_score": threat.threat_score,
                "animal_class": threat.animal_class,
                "confidence": threat.confidence,
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


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Wildlife-COD Web Interface running!")
    print("Open your browser at: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
