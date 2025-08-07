#!/usr/bin/env python3
"""
Performance testing script for optimized camera drivers.
Tests parallel reading performance with multiple cameras.
"""

import time
import logging
import argparse
from pathlib import Path
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_camera_performance(camera_configs, num_frames=100, with_depth=True):
    """
    Test camera performance with the specified configuration.
    
    Args:
        camera_configs: Dictionary of camera configurations
        num_frames: Number of frames to capture for testing
        with_depth: Whether to capture depth data
    """
    from lerobot.cameras.utils import make_cameras_from_configs
    from lerobot.cameras.parallel_camera_reader import ParallelCameraReader

    # Create cameras
    logger.info(f"Creating {len(camera_configs)} cameras...")
    cameras = make_cameras_from_configs(camera_configs)
    
    # Connect all cameras
    logger.info("Connecting cameras...")
    for name, cam in cameras.items():
        logger.info(f"  Connecting {name}: {cam}")
        cam.connect(warmup=True)

    # Create parallel reader
    reader = ParallelCameraReader(persistent_executor=True)

    # Warmup phase
    logger.info("Warming up cameras (10 frames)...")
    for i in range(10):
        frames = reader.read_cameras_parallel(cameras, with_depth=with_depth)
        time.sleep(0.033)  # ~30 FPS
    
    # Performance testing
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting performance test: {num_frames} frames")
    logger.info(f"{'='*60}")
    
    frame_times = []
    camera_times = {name: [] for name in cameras.keys()}
    
    for i in range(num_frames):
        start = time.perf_counter()
        
        # Read all cameras in parallel
        frames = reader.read_cameras_parallel(cameras, with_depth=with_depth)
        
        elapsed = (time.perf_counter() - start) * 1000
        frame_times.append(elapsed)
        
        # Log progress every 20 frames
        if (i + 1) % 20 == 0:
            recent_avg = np.mean(frame_times[-20:])
            logger.info(f"Frame {i+1}/{num_frames}: Last 20 frames avg: {recent_avg:.1f}ms")

    # Calculate statistics
    frame_times_np = np.array(frame_times)
    
    logger.info(f"\n{'='*60}")
    logger.info("PERFORMANCE RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total frames captured: {num_frames}")
    logger.info(f"Average frame time: {frame_times_np.mean():.2f}ms")
    logger.info(f"Median frame time: {np.median(frame_times_np):.2f}ms")
    logger.info(f"Min frame time: {frame_times_np.min():.2f}ms")
    logger.info(f"Max frame time: {frame_times_np.max():.2f}ms")
    logger.info(f"Std deviation: {frame_times_np.std():.2f}ms")
    
    # Calculate effective FPS
    avg_time_s = frame_times_np.mean() / 1000.0
    effective_fps = 1.0 / avg_time_s if avg_time_s > 0 else 0
    logger.info(f"Effective FPS: {effective_fps:.1f} Hz")
    
    # Show percentiles
    logger.info(f"\nPercentiles:")
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(frame_times_np, p)
        logger.info(f"  {p}th percentile: {val:.2f}ms")
    
    # Get final stats from reader
    reader_stats = reader.get_stats()
    logger.info(f"\nReader statistics:")
    logger.info(f"  Total reads: {reader_stats['total_reads']}")
    logger.info(f"  Failed reads: {reader_stats['failed_reads']}")
    logger.info(f"  Average time: {reader_stats['avg_time_ms']:.2f}ms")
    logger.info(f"  Max time: {reader_stats['max_time_ms']:.2f}ms")
    
    # Disconnect cameras
    logger.info("\nDisconnecting cameras...")
    for name, cam in cameras.items():
        cam.disconnect()
        logger.info(f"  Disconnected {name}")
    
    return frame_times_np, effective_fps


def main():
    parser = argparse.ArgumentParser(description='Test camera performance')
    parser.add_argument('--config', type=int, default=6,
                        help='Configuration number (1-8, matching record_simple.sh)')
    parser.add_argument('--frames', type=int, default=100,
                        help='Number of frames to capture')
    parser.add_argument('--no-depth', action='store_true',
                        help='Disable depth capture')
    args = parser.parse_args()
    
    # Camera configurations matching record_simple.sh
    from lerobot.cameras.kinect import KinectCameraConfig
    from lerobot.cameras.realsense import RealSenseCameraConfig
    from lerobot.cameras import ColorMode
    
    # RealSense serial numbers
    REALSENSE1_SERIAL = "218622270973"
    REALSENSE2_SERIAL = "218622278797"
    
    configs = {
        1: {  # Kinect RGB only
            "top": KinectCameraConfig(
                device_index=0,
                fps=30,
                color_mode=ColorMode.RGB,
                use_depth=False
            )
        },
        2: {  # Kinect RGB + Depth
            "top": KinectCameraConfig(
                device_index=0,
                fps=30,
                color_mode=ColorMode.RGB,
                use_depth=True,
                depth_colormap="jet"
            )
        },
        3: {  # 2 RealSense RGB only
            "left": RealSenseCameraConfig(
                serial_number_or_name=REALSENSE1_SERIAL,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=False
            ),
            "right": RealSenseCameraConfig(
                serial_number_or_name=REALSENSE2_SERIAL,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=False
            )
        },
        4: {  # 2 RealSense RGB + Depth
            "left": RealSenseCameraConfig(
                serial_number_or_name=REALSENSE1_SERIAL,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=True,
                depth_colormap="jet",
                depth_min_meters=0.07,  # D405 min range
                depth_max_meters=0.5    # D405 max range
            ),
            "right": RealSenseCameraConfig(
                serial_number_or_name=REALSENSE2_SERIAL,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=True,
                depth_colormap="jet",
                depth_min_meters=0.07,  # D405 min range
                depth_max_meters=0.5    # D405 max range
            )
        },
        5: {  # 2 RealSense RGB + Kinect RGB
            "left": RealSenseCameraConfig(
                serial_number_or_name=REALSENSE1_SERIAL,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=False
            ),
            "right": RealSenseCameraConfig(
                serial_number_or_name=REALSENSE2_SERIAL,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=False
            ),
            "top": KinectCameraConfig(
                device_index=0,
                fps=30,
                color_mode=ColorMode.RGB,
                use_depth=False
            )
        },
        6: {  # 2 RealSense RGB + Depth + Kinect RGB + Depth
            "left": RealSenseCameraConfig(
                serial_number_or_name=REALSENSE1_SERIAL,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=True,
                depth_colormap="jet",
                depth_min_meters=0.07,  # D405 min range
                depth_max_meters=0.5    # D405 max range
            ),
            "right": RealSenseCameraConfig(
                serial_number_or_name=REALSENSE2_SERIAL,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=True,
                depth_colormap="jet",
                depth_min_meters=0.07,  # D405 min range
                depth_max_meters=0.5    # D405 max range
            ),
            "top": KinectCameraConfig(
                device_index=0,
                fps=30,
                color_mode=ColorMode.RGB,
                use_depth=True,
                depth_colormap="jet"
            )
        },
        8: {  # 2 RealSense RGB + Depth + Kinect RGB only
            "left": RealSenseCameraConfig(
                serial_number_or_name=REALSENSE1_SERIAL,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=True,
                depth_colormap="jet",
                depth_min_meters=0.07,  # D405 min range
                depth_max_meters=0.5    # D405 max range
            ),
            "right": RealSenseCameraConfig(
                serial_number_or_name=REALSENSE2_SERIAL,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                use_depth=True,
                depth_colormap="jet",
                depth_min_meters=0.07,  # D405 min range
                depth_max_meters=0.5    # D405 max range
            ),
            "top": KinectCameraConfig(
                device_index=0,
                fps=30,
                color_mode=ColorMode.RGB,
                use_depth=False
            )
        }
    }
    
    if args.config not in configs:
        logger.error(f"Invalid configuration {args.config}. Choose from: {list(configs.keys())}")
        return 1
    
    camera_configs = configs[args.config]
    with_depth = not args.no_depth
    
    logger.info(f"Testing configuration {args.config}:")
    for name, cfg in camera_configs.items():
        logger.info(f"  {name}: {cfg.type} (depth={'enabled' if cfg.use_depth else 'disabled'})")
    
    try:
        frame_times, fps = test_camera_performance(
            camera_configs, 
            num_frames=args.frames,
            with_depth=with_depth
        )
        
        # Success message
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST COMPLETED SUCCESSFULLY")
        logger.info(f"Achieved {fps:.1f} FPS with {len(camera_configs)} cameras")
        logger.info(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())