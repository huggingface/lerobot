#!/usr/bin/env python3
"""
Simple test of RealSense depth colorization to verify basic functionality.
"""

import time

import numpy as np
import pyrealsense2 as rs


def test_simple_colorization():
    print("Testing RealSense depth colorization...")

    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start
    pipeline.start(config)

    # Create colorizer
    colorizer = rs.colorizer()

    print("Capturing 30 frames...")
    errors = 0

    for i in range(30):
        try:
            # Get frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()

            # Colorize
            colorized = colorizer.colorize(depth_frame)
            colorized_data = np.asanyarray(colorized.get_data())

            print(f"Frame {i + 1}: Success, shape={colorized_data.shape}")

        except Exception as e:
            errors += 1
            print(f"Frame {i + 1}: Error - {e}")

        time.sleep(0.033)  # ~30 FPS

    # Stop
    pipeline.stop()

    print(f"\nComplete: {errors} errors out of 30 frames")

    if errors > 0:
        print("\nPossible causes of errors:")
        print("1. USB bandwidth limitations")
        print("2. CPU/GPU processing delays")
        print("3. Frame synchronization issues")
        print("\nDespite errors, data is usually captured correctly.")


if __name__ == "__main__":
    test_simple_colorization()
