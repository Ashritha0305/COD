"""
scripts/test_image.py
=====================
Standalone CLI test runner for Image Mode.

Usage:
    python scripts/test_image.py
    python scripts/test_image.py --image path/to/image.jpg
"""

import os
import sys
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from modules.pipeline import WildlifePipeline


def main():
    parser = argparse.ArgumentParser(description="Run Wildlife-COD complete pipeline on an image")
    parser.add_argument("--image", type=str, default="data/test_image.jpg", help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="outputs/cli_image_test", help="Output directory")
    args = parser.parse_args()

    print(f"Running Wildlife-COD Image Pipeline on: {args.image}")
    pipeline = WildlifePipeline()
    results = pipeline.process_image(args.image, save_dir=args.output_dir)

    threat = results["threat_result"]
    print("\n" + "=" * 55)
    print(f"       THREAT LEVEL: {threat.threat_level}")
    print("=" * 55)
    print(f"Target:       {threat.animal_class}")
    print(f"Confidence:   {threat.confidence * 100:.1f}%")
    print(f"Motion:       {threat.motion_level}")
    print(f"Depth Status: {threat.relative_depth}")
    print(f"Note:         {threat.depth_note}")
    print(f"Reasoning:    {threat.reason}")
    print(f"Latency:      {results['elapsed_ms']:.0f} ms")
    print("=" * 55)
    print(f"Saved outputs to: {args.output_dir}/\n")


if __name__ == "__main__":
    main()
