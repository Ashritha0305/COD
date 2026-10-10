"""
scripts/benchmark_optical_flow_resolutions.py
=============================================
Phase A requirement:
Benchmark Farneback optical flow at:
  1. 960x540 (Current processing resolution)
  2. 640x360 (Intermediate resolution)
  3. 320x180 (Low resolution)
Measure latency (Mean, Median, P95, Std) and motion evidence fidelity.
"""

import os
import sys
import time
import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from modules.optical_flow import FarnebackOpticalFlow, _flow_to_magnitude

def run_benchmark(video_path: str = "data/testvideo.mp4", num_frames: int = 30):
    print("=" * 80)
    print(" FARNEBACK OPTICAL FLOW RESOLUTION BENCHMARK")
    print("=" * 80)
    print(f"Target Video : {video_path}")
    print(f"Frames       : {num_frames}")
    print("=" * 80)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames_bgr = []
    while len(frames_bgr) < num_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        # Native resize to 960x540 base
        frame_540 = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        frames_bgr.append(frame_540)
    cap.release()

    print(f"Loaded {len(frames_bgr)} frames at 960x540 for deterministic benchmarking.\n")

    resolutions = [
        ("960x540 (Current)", 960, 540),
        ("640x360 (Medium)",  640, 360),
        ("320x180 (Fast)",    320, 180),
    ]

    results = {}
    reference_maps = []

    for name, w, h in resolutions:
        latencies = []
        flow_maps = []
        prev_gray = None

        # Pre-warm
        g0 = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
        if (w, h) != (960, 540):
            g0 = cv2.resize(g0, (w, h), interpolation=cv2.INTER_AREA)
        g1 = cv2.cvtColor(frames_bgr[1], cv2.COLOR_BGR2GRAY)
        if (w, h) != (960, 540):
            g1 = cv2.resize(g1, (w, h), interpolation=cv2.INTER_AREA)
        _ = cv2.calcOpticalFlowFarneback(
            g0, g1, None,
            config.OF_PYR_SCALE, config.OF_LEVELS, config.OF_WINSIZE,
            config.OF_ITERATIONS, config.OF_POLY_N, config.OF_POLY_SIGMA,
            config.OF_FLAGS
        )

        # Benchmark run
        for idx in range(len(frames_bgr)):
            curr_frame = frames_bgr[idx]
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            if (w, h) != (960, 540):
                curr_gray_calc = cv2.resize(curr_gray, (w, h), interpolation=cv2.INTER_AREA)
            else:
                curr_gray_calc = curr_gray

            if prev_gray is None:
                prev_gray = curr_gray_calc
                continue

            t0 = time.perf_counter()
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray_calc, None,
                config.OF_PYR_SCALE, config.OF_LEVELS, config.OF_WINSIZE,
                config.OF_ITERATIONS, config.OF_POLY_N, config.OF_POLY_SIGMA,
                config.OF_FLAGS
            )
            # Magnitude calculation
            mag = _flow_to_magnitude(flow)
            # Upscale back to 960x540 if needed
            if (w, h) != (960, 540):
                mag_540 = cv2.resize(mag, (960, 540), interpolation=cv2.INTER_LINEAR)
            else:
                mag_540 = mag
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)
            flow_maps.append(mag_540)
            prev_gray = curr_gray_calc

        arr = np.array(latencies)
        results[name] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "std": float(np.std(arr)),
            "maps": flow_maps,
        }
        if name == "960x540 (Current)":
            reference_maps = flow_maps

    # Display comparison table
    print("-" * 85)
    print(f"{'RESOLUTION':<22} | {'MEAN (ms)':>10} | {'MEDIAN (ms)':>12} | {'P95 (ms)':>10} | {'SPEEDUP':>9} | {'MAE vs 960p':>12}")
    print("-" * 85)

    base_mean = results["960x540 (Current)"]["mean"]
    for name, _, _ in resolutions:
        r = results[name]
        speedup = base_mean / r["mean"]
        # Compute mean absolute error against reference
        maes = []
        for m_test, m_ref in zip(r["maps"], reference_maps):
            maes.append(float(np.mean(np.abs(m_test - m_ref))))
        mae_val = float(np.mean(maes)) if maes else 0.0

        print(f"{name:<22} | {r['mean']:>10.2f} | {r['median']:>12.2f} | {r['p95']:>10.2f} | {speedup:>8.2f}x | {mae_val:>12.4f}")
    print("-" * 85)


if __name__ == "__main__":
    run_benchmark()
