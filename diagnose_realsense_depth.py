#!/usr/bin/env python3
"""
Diagnostic script for RealSense depth colorization issues.
This script helps identify why "Error occurred during execution of the processing block!" happens.
"""

import logging
import time

import numpy as np
import pyrealsense2 as rs

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def diagnose_realsense_depth(serial_number: str = None):
    """
    Run diagnostic tests on RealSense depth colorization.
    """
    logger.info("Starting RealSense depth diagnostics...")

    # Create pipeline and config
    pipeline = rs.pipeline()
    config = rs.config()

    # Find device
    context = rs.context()
    devices = context.query_devices()

    if len(devices) == 0:
        logger.error("No RealSense devices found!")
        return

    # Use first device if no serial specified
    if serial_number:
        config.enable_device(serial_number)
        logger.info(f"Using device with serial: {serial_number}")
    else:
        device = devices[0]
        serial = device.get_info(rs.camera_info.serial_number)
        config.enable_device(serial)
        logger.info(f"Using first device found: {serial}")

    # Enable streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start pipeline
    logger.info("Starting pipeline...")
    profile = pipeline.start(config)

    # Get device info
    device = profile.get_device()
    logger.info(f"Device: {device.get_info(rs.camera_info.name)}")
    logger.info(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")
    logger.info(f"USB Type: {device.get_info(rs.camera_info.usb_type_descriptor)}")

    # Create colorizer with different settings
    colorizers = {
        "default": rs.colorizer(),
        "histogram_equalization": rs.colorizer(),
        "custom_range": rs.colorizer(),
    }

    # Configure colorizers
    try:
        # Test histogram equalization
        colorizers["histogram_equalization"].set_option(rs.option.histogram_equalization_enabled, 1.0)
        logger.info("Histogram equalization enabled")
    except Exception as e:
        logger.warning(f"Could not enable histogram equalization: {e}")

    try:
        # Test custom range
        colorizers["custom_range"].set_option(rs.option.min_distance, 0.3)
        colorizers["custom_range"].set_option(rs.option.max_distance, 3.0)
        logger.info("Custom range set: 0.3m - 3.0m")
    except Exception as e:
        logger.warning(f"Could not set custom range: {e}")

    # Test different scenarios
    test_scenarios = [("normal", 5, False), ("rapid_capture", 0.1, False), ("with_align", 5, True)]

    for scenario_name, delay, use_align in test_scenarios:
        logger.info(f"\n=== Testing scenario: {scenario_name} ===")

        align = rs.align(rs.stream.color) if use_align else None
        error_count = 0
        success_count = 0

        for i in range(10):
            try:
                # Wait for frames
                frames = pipeline.wait_for_frames()

                if use_align and align:
                    frames = align.process(frames)

                # Get depth frame
                depth_frame = frames.get_depth_frame()

                if not depth_frame:
                    logger.warning(f"Frame {i}: No depth frame received")
                    continue

                # Log frame info
                frame_number = depth_frame.get_frame_number()
                timestamp = depth_frame.get_timestamp()
                logger.debug(f"Frame {i}: number={frame_number}, timestamp={timestamp}")

                # Test each colorizer
                for colorizer_name, colorizer in colorizers.items():
                    try:
                        # Apply colorizer
                        colorized = colorizer.colorize(depth_frame)

                        # Verify output
                        colorized_data = np.asanyarray(colorized.get_data())

                        if colorized_data.shape[2] != 3:
                            logger.error(
                                f"Frame {i}, {colorizer_name}: Invalid colorized shape: {colorized_data.shape}"
                            )
                        else:
                            success_count += 1
                            logger.debug(
                                f"Frame {i}, {colorizer_name}: Success, shape={colorized_data.shape}"
                            )

                    except RuntimeError as e:
                        error_count += 1
                        logger.error(f"Frame {i}, {colorizer_name}: RuntimeError: {e}")
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Frame {i}, {colorizer_name}: {type(e).__name__}: {e}")

                # Delay between captures
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Frame capture error: {e}")

        logger.info(f"Scenario {scenario_name} complete: {success_count} successes, {error_count} errors")

    # Test rapid fire colorization
    logger.info("\n=== Testing rapid colorization ===")
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    if depth_frame:
        colorizer = rs.colorizer()
        start_time = time.time()
        rapid_errors = 0

        for i in range(100):
            try:
                colorized = colorizer.colorize(depth_frame)
                _ = np.asanyarray(colorized.get_data())
            except Exception as e:
                rapid_errors += 1
                logger.debug(f"Rapid test error {i}: {e}")

        elapsed = time.time() - start_time
        logger.info(f"Rapid test: 100 colorizations in {elapsed:.2f}s, {rapid_errors} errors")

    # Cleanup
    pipeline.stop()
    logger.info("\nDiagnostics complete!")

    # Summary
    logger.info("\nSummary:")
    logger.info("- Processing block errors may occur when:")
    logger.info("  1. Depth frame is corrupted or incomplete")
    logger.info("  2. Colorizer is accessed from multiple threads")
    logger.info("  3. Rapid colorization without proper synchronization")
    logger.info("  4. USB bandwidth issues cause frame drops")
    logger.info("\nRecommendation: The errors during recording are likely due to")
    logger.info("concurrent access or frame timing issues. The data is still")
    logger.info("captured correctly as you observed.")


if __name__ == "__main__":
    import sys

    serial = None
    if len(sys.argv) > 1:
        serial = sys.argv[1]

    diagnose_realsense_depth(serial)
