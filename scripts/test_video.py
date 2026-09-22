"""
scripts/test_video.py
=====================
Standalone CLI test runner for Video Mode.

Usage:
    python scripts/test_video.py
    python scripts/test_video.py --video path/to/video.mp4 --max-frames 60
"""

import os
import sys
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from modules.pipeline import WildlifePipeline


def main():
    parser = argparse.ArgumentParser(description="Run Wildlife-COD complete pipeline on a video")
    parser.add_argument("--video", type=str, default="data/testvideo.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/cli_video_test/enhanced_video.mp4", help="Output video path")
    parser.add_argument("--max-frames", type=int, default=45, help="Maximum frames to process (None for all)")
    args = parser.parse_args()

    print(f"Running Wildlife-COD Video Pipeline on: {args.video} (max_frames={args.max_frames})")
    pipeline = WildlifePipeline()

    def on_progress(cur, total, pct):
        if cur % 5 == 0 or cur == total:
            print(f"  Frame {cur}/{total} ({pct * 100:.0f}%) processed...")

    results = pipeline.process_video(
        video_path=args.video,
        output_video_path=args.output,
        max_frames=args.max_frames,
        progress_callback=on_progress,
    )

    ot = results["overall_threat"]
    print("\n" + "=" * 55)
    if ot:
        print(f"       OVERALL THREAT LEVEL: {ot.threat_level}")
        print("=" * 55)
        print(f"Target:       {ot.animal_class}")
        print(f"Confidence:   {ot.confidence * 100:.1f}%")
        print(f"Motion:       {ot.motion_level}")
        print(f"Depth Status: {ot.relative_depth}")
        print(f"Reasoning:    {ot.reason}")
    else:
        print("       OVERALL THREAT LEVEL: SAFE")
        print("=" * 55)
        print("No active wildlife targets detected.")

    print(f"Frames:       {results['frames_processed']}")
    print(f"Offline FPS:  {results['actual_fps']:.2f}")
    print(f"Total Time:   {results['total_time_s']:.1f} s")
    print("=" * 55)
    print(f"Output video saved to: {results['output_video_path']}\n")


if __name__ == "__main__":
    main()
