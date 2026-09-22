# wildlife-COD

**Real-Time Camouflage-Aware Wildlife Detection with Rule-Based Threat Analysis**

Final-Year Engineering Project — Complete Working End-to-End Prototype

---

## 1. Project Objective

Wildlife camouflage poses a significant challenge to standard computer vision models. Animals naturally blend into foliage, rocky terrain, and low-contrast environments. This project delivers a unified multi-stage pipeline designed to expose camouflaged wildlife through learned binary segmentation and optical motion analysis, detect animal classes via YOLOv8, track them persistently with DeepSORT, generate spatial depth context using ZoeDepth, and assess security risk with a deterministic rule-based threat analysis system.

---

## 2. System Architecture

The paper and prototype architecture is strictly fixed to 7 specialized stages:

```
                    INPUT
                 IMAGE / VIDEO
                      │
                      ▼
              ┌───────────────┐
              │     U-NET     │  Stage 1: Camouflage Feature Segmentation
              │  Probability  │  (Trained on MoCA dataset, checkpoints/unet_best.pth)
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ OPTICAL FLOW  │  Stage 2: Farneback Dense Motion Analysis
              │  Motion Map   │  (Inter-frame pixel velocity)
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │    FUSION     │  Stage 3: Gated Attention + LAB CLAHE
              │  Enhancement  │  (Exposes camouflage structure without blow-out)
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │    YOLOv8     │  Stage 4: Animal & Object Detection
              │ Bounding Box  │  (Class identification + Confidence)
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │   DeepSORT    │  Stage 5: Multi-Object Tracking
              │   Track IDs   │  (Persistent ID assignment & frame trajectory)
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │   ZoeDepth    │  Stage 6: Monocular Metric/Relative Depth Map
              │   Depth Map   │  (Spatial scene structure)
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │THREAT ANALYSIS│  Stage 7: Deterministic Rule-Based Reasoning
              │  Rule Engine  │  (Species threat, motion magnitude, persistence)
              └───────────────┘
                      │
                      ▼
            SAFE / WARNING / CRITICAL
```

---

## 3. Pipeline Components

| Stage | Component | Technical Role |
|---|---|---|
| **1** | **U-Net** | 4-level encoder-decoder network with CBAM attention module at bottleneck (`checkpoints/unet_best.pth`). Outputs per-pixel camouflage probability map. Includes morphological opening/closing and component filtering to eliminate background noise. |
| **2** | **Optical Flow** | OpenCV Farneback dense optical flow algorithm computing inter-frame displacement vectors and motion magnitude. |
| **3** | **Fusion** | U-Net-gated motion fusion applying CLAHE contrast enhancement on the LAB L-channel and subtle HSV saturation boost in the animal region. |
| **4** | **YOLOv8** | PyTorch YOLOv8 detection model running inference on the enhanced frames to extract bounding boxes, animal labels, and detection confidence. |
| **5** | **DeepSORT** | Deep cosine metric learning and Kalman filter tracker (`deep-sort-realtime`) binding detections across video frames with unique, persistent Track IDs. |
| **6** | **ZoeDepth** | Relative monocular depth estimation (`ZoeD_N`) providing scene topology and depth categorization (`Near`, `Medium`, `Far`). |
| **7** | **Threat Analysis** | Transparent, rule-based classification combining animal danger category, detection confidence, movement level, and track persistence into `SAFE`, `WARNING`, or `CRITICAL`. |

---

## 4. Current Review Prototype Limitations

1. **Distance Metric Honesty**:
   - Monocular depth maps from ZoeDepth are computed and displayed for spatial awareness.
   - For this prototype, **calibrated metric distance (meters) is not used to force an unvalidated threat decision**.
   - Output explicitly notes: *"Depth available (Relative: Near/Medium/Far) — distance estimation reserved for calibrated camera deployment"*.
2. **Actual Measured FPS**:
   - No fake or fabricated FPS counters are used.
   - Offline batch video processing reports genuine hardware throughput on the active GPU (NVIDIA RTX 4050 Laptop GPU).

---

## 5. Directory Structure

```
wildlife_COD/
├── app.py                      # Interactive Flask web application
├── config.py                   # Central parameters and thresholds
├── requirements.txt            # Python dependencies
├── README.md                   # Complete architectural documentation
│
├── checkpoints/
│   └── unet_best.pth           # MoCA-trained U-Net weights
│
├── models/
│   └── unet/
│       └── unet.py             # U-Net architecture definition (DoubleConv / CBAM)
│
├── modules/
│   ├── camouflage_processor.py # Integrated U-Net + Farneback + Enhancement
│   ├── optical_flow.py         # Farneback Optical Flow wrapper
│   ├── fusion.py               # LAB CLAHE & Saturation natural fusion
│   ├── detection.py            # YOLOv8 detector wrapper
│   ├── tracking.py             # DeepSORT tracker wrapper
│   ├── depth.py                # ZoeDepth depth estimator wrapper
│   ├── threat_analysis.py      # Rule-based threat engine (SAFE/WARNING/CRITICAL)
│   └── pipeline.py             # Unified 7-stage orchestrator
│
├── scripts/
│   ├── test_image.py           # Standalone CLI image test runner
│   ├── test_video.py           # Standalone CLI video test runner
│   └── run_pipeline.py         # Diagnostic pipeline script
│
├── templates/
│   └── index.html              # Responsive dark-mode dashboard
│
├── data/                       # Sample inputs (test_image.jpg, testvideo.mp4)
├── uploads/                    # Uploaded user media
└── outputs/                    # Processed video and image outputs
```

---

## 6. Setup & Execution

### Prerequisites

```bash
# Verify GPU acceleration (NVIDIA RTX 4050 Laptop GPU):
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Interactive Web Interface

```bash
python app.py
```

Then open your browser at:
👉 **`http://127.0.0.1:5000`**

1. **Image Mode**:
   - Upload any camouflaged animal image.
   - View synchronized 4-stage panel: Original, U-Net Mask, Enhanced + YOLO Detection, and ZoeDepth Depth Map.
   - Inspect authoritative Threat Level badge.
2. **Video Mode**:
   - Upload any video sequence (`.mp4`, `.avi`, `.mkv`).
   - Full processing: U-Net + Optical Flow + Fusion + YOLOv8 + DeepSORT + ZoeDepth + Threat Analysis.
   - Play or download the annotated video with persistent Track IDs and Threat HUD.

---

## 7. Standalone CLI Verification

You can also run headless tests directly from the terminal:

### Image Mode Test:
```bash
python scripts/test_image.py --image data/test_image.jpg
```

### Video Mode Test:
```bash
python scripts/test_video.py --video data/testvideo.mp4 --max-frames 30
```

---

## 8. Future Real-Time Deployment Roadmap

The codebase is engineered modularly (`process_frame()`, `WildlifePipeline`) so that webcam/RTSP streaming can be plugged directly into the exact same 7-stage pipeline:
```
Current:  Uploaded Video/Image -> process_frame() -> Processed Output
Future:   Live Webcam/RTSP     -> process_frame() -> Real-Time Alert HUD
```
