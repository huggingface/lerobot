#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Depth Camera Performance Benchmark

This script benchmarks depth camera performance across different configurations
to identify optimal settings for robotics applications.

See the provided README.md or run `python benchmarks/depth_cameras/run_depth_benchmark.py --help` for usage info.
"""

import argparse
import datetime as dt
import logging
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import depth camera utilities
from lerobot.cameras.camera_manager import CameraManager, create_camera_system
from lerobot.cameras.depth_utils import colorize_depth_frame, is_raw_depth
from lerobot.cameras.realsense import RealSenseCameraConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def detect_realsense_cameras() -> list[str]:
    """Detect available RealSense cameras and return their serial numbers."""
    try:
        import pyrealsense2 as rs

        ctx = rs.context()
        devices = ctx.query_devices()
        serial_numbers = []

        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            serial_numbers.append(serial)
            logger.info(f"Detected RealSense camera: {name} (Serial: {serial})")

        if not serial_numbers:
            logger.warning("No RealSense cameras detected")

        return serial_numbers

    except ImportError:
        logger.error("pyrealsense2 not available. Install with: pip install pyrealsense2")
        return []
    except Exception as e:
        logger.error(f"Error detecting cameras: {e}")
        return []


def create_camera_configs(
    serial_numbers: list[str], width: int, height: int, fps: int, use_depth: bool
) -> dict[str, RealSenseCameraConfig]:
    """Create camera configurations from serial numbers."""
    configs = {}

    for i, serial in enumerate(serial_numbers):
        cam_name = f"camera_{i}"
        configs[cam_name] = RealSenseCameraConfig(
            serial_number_or_name=serial, width=width, height=height, fps=fps, use_depth=use_depth
        )

    return configs


class DepthCameraBenchmark:
    """Comprehensive depth camera performance benchmark."""

    def __init__(self, args):
        self.args = args
        self.results = []
        self.available_cameras = detect_realsense_cameras()

        if not self.available_cameras:
            raise RuntimeError("No RealSense cameras detected")

        logger.info(f"Available cameras: {len(self.available_cameras)}")

    def setup_cameras(self, config_subset: dict, test_name: str) -> tuple[dict, CameraManager]:
        """Setup cameras for testing with proper error handling."""
        try:
            logger.info(f"Setting up cameras for: {test_name}")
            camera_system = create_camera_system(config_subset)

            # Connect all cameras
            for cam_name, camera in camera_system.cameras.items():
                camera.connect(warmup=False)
                logger.info(f"Connected {cam_name}")

            # Warmup period for background threads
            time.sleep(self.args.warmup_time)

            # Verify cameras are working
            for cam_name, camera in camera_system.cameras.items():
                if not camera.is_connected:
                    raise RuntimeError(f"Camera {cam_name} failed to connect")

                # Test read to ensure background thread is working
                try:
                    if camera_system.capabilities[cam_name]["has_depth"]:
                        camera.async_read_rgb_and_depth(timeout_ms=1000)
                    else:
                        camera.async_read(timeout_ms=1000)
                except Exception as e:
                    raise RuntimeError(f"Camera {cam_name} read test failed: {e}") from e

            logger.info(f"Successfully setup {len(camera_system.cameras)} cameras")
            return camera_system.cameras, camera_system

        except Exception as e:
            logger.error(f"Camera setup failed for {test_name}: {e}")
            raise

    def cleanup_cameras(self, camera_system: CameraManager | None):
        """Safely disconnect all cameras."""
        if camera_system:
            try:
                for camera in camera_system.cameras.values():
                    if camera.is_connected:
                        camera.disconnect()
                logger.info("Cameras disconnected successfully")
            except Exception as e:
                logger.warning(f"Error during camera cleanup: {e}")

    def measure_performance(
        self,
        camera_system: CameraManager,
        test_name: str,
        timeout_ms: int = 150,
        processing_mode: str = "standard",
    ) -> dict:
        """Measure camera system performance."""
        logger.info(f"Measuring performance: {test_name} (timeout: {timeout_ms}ms)")

        fps_measurements = []
        latency_measurements = []
        success_count = 0
        error_count = 0

        start_time = time.perf_counter()
        last_frame_time = start_time

        with tqdm(desc=f"Testing {test_name}", unit="frames") as pbar:
            while (time.perf_counter() - start_time) < self.args.test_duration:
                try:
                    frame_start = time.perf_counter()

                    if processing_mode == "raw_only":
                        # Test raw camera reads without CameraManager
                        observation = self._raw_camera_reads(camera_system, timeout_ms)
                    elif processing_mode == "with_colorization":
                        # Test with depth colorization
                        observation = self._read_with_colorization(camera_system, timeout_ms)
                    elif processing_mode == "with_rerun":
                        # Test with Rerun logging overhead
                        observation = self._read_with_rerun(camera_system, timeout_ms)
                    else:
                        # Standard CameraManager read
                        observation = camera_system.read_all(timeout_ms=timeout_ms)

                    frame_end = time.perf_counter()

                    # Validate observation has data
                    if observation and any(isinstance(v, np.ndarray) for v in observation.values()):
                        current_time = frame_end

                        # Calculate FPS and latency
                        fps = 1.0 / (current_time - last_frame_time)
                        latency_ms = (frame_end - frame_start) * 1000

                        # Filter realistic values
                        if 1.0 <= fps <= 100.0:
                            fps_measurements.append(fps)
                            latency_measurements.append(latency_ms)
                            success_count += 1

                        last_frame_time = current_time
                        pbar.update(1)
                    else:
                        error_count += 1

                except Exception:
                    error_count += 1
                    # Brief pause to prevent error spam
                    time.sleep(0.01)

        # Calculate statistics
        total_attempts = success_count + error_count
        success_rate = (success_count / total_attempts * 100) if total_attempts > 0 else 0

        result = {
            "test_name": test_name,
            "timeout_ms": timeout_ms,
            "processing_mode": processing_mode,
            "camera_count": len(camera_system.cameras),
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_rate,
            "avg_fps": statistics.mean(fps_measurements) if fps_measurements else 0,
            "fps_std_dev": statistics.stdev(fps_measurements) if len(fps_measurements) > 1 else 0,
            "min_fps": min(fps_measurements) if fps_measurements else 0,
            "max_fps": max(fps_measurements) if fps_measurements else 0,
            "avg_latency_ms": statistics.mean(latency_measurements) if latency_measurements else 0,
            "max_latency_ms": max(latency_measurements) if latency_measurements else 0,
        }

        logger.info(f"Results: {result['avg_fps']:.1f} Hz avg, {result['success_rate']:.1f}% success")
        return result

    def _raw_camera_reads(self, camera_system: CameraManager, timeout_ms: int) -> dict:
        """Direct camera reads bypassing CameraManager processing."""
        observation = {}

        for cam_name, camera in camera_system.cameras.items():
            if camera_system.capabilities[cam_name]["has_depth"]:
                rgb, depth = camera.async_read_rgb_and_depth(timeout_ms=timeout_ms)
                observation[cam_name] = rgb
                observation[f"{cam_name}_depth_raw"] = depth
            else:
                rgb = camera.async_read(timeout_ms=timeout_ms)
                observation[cam_name] = rgb

        return observation

    def _read_with_colorization(self, camera_system: CameraManager, timeout_ms: int) -> dict:
        """Camera reads with depth colorization processing."""
        observation = {}

        for cam_name, camera in camera_system.cameras.items():
            if camera_system.capabilities[cam_name]["has_depth"]:
                rgb, depth = camera.async_read_rgb_and_depth(timeout_ms=timeout_ms)
                colorized_depth = colorize_depth_frame(depth)  # Processing overhead

                observation[cam_name] = rgb
                observation[f"{cam_name}_depth"] = colorized_depth
                observation[f"{cam_name}_depth_raw"] = depth
            else:
                rgb = camera.async_read(timeout_ms=timeout_ms)
                observation[cam_name] = rgb

        return observation

    def _read_with_rerun(self, camera_system: CameraManager, timeout_ms: int) -> dict:
        """Camera reads with Rerun logging overhead."""
        try:
            import rerun as rr
        except ImportError:
            logger.warning("Rerun not available, falling back to standard read")
            return camera_system.read_all(timeout_ms=timeout_ms)

        # Use CameraManager read_all for full processing pipeline
        observation = camera_system.read_all(timeout_ms=timeout_ms)

        # Simulate Rerun logging overhead
        for obs_name, obs_value in observation.items():
            if isinstance(obs_value, np.ndarray):
                if obs_name.endswith("_depth_raw") and is_raw_depth(obs_value):
                    # Log depth data (this adds overhead)
                    rr.log(f"benchmark.{obs_name}", rr.DepthImage(obs_value, meter=1.0 / 1000.0))
                elif not obs_name.endswith("_depth"):  # Skip colorized depth
                    # Log RGB data
                    rr.log(f"benchmark.{obs_name}", rr.Image(obs_value))

        return observation

    def run_camera_count_test(self):
        """Test performance scaling with different camera counts."""
        logger.info("Running camera count scaling test")

        max_cameras = min(len(self.available_cameras), self.args.max_cameras)

        for camera_count in range(1, max_cameras + 1):
            test_name = f"camera_count_{camera_count}"

            # Use first N available cameras
            selected_serials = self.available_cameras[:camera_count]
            configs = create_camera_configs(
                selected_serials, self.args.width, self.args.height, self.args.fps, self.args.use_depth
            )

            try:
                cameras, camera_system = self.setup_cameras(configs, test_name)

                # Test with different timeout values
                for timeout_ms in self.args.timeout_ms:
                    result = self.measure_performance(
                        camera_system, f"{test_name}_timeout_{timeout_ms}", timeout_ms=timeout_ms
                    )
                    result.update(
                        {
                            "test_category": "camera_count",
                            "width": self.args.width,
                            "height": self.args.height,
                            "fps": self.args.fps,
                            "use_depth": self.args.use_depth,
                        }
                    )
                    self.results.append(result)

            except Exception as e:
                logger.error(f"Camera count test failed for {camera_count} cameras: {e}")
                continue
            finally:
                self.cleanup_cameras(camera_system)

    def run_processing_overhead_test(self):
        """Test processing overhead of different pipeline stages."""
        logger.info("Running processing overhead test")

        # Use single camera for overhead analysis
        selected_serials = [self.available_cameras[0]]
        configs = create_camera_configs(
            selected_serials,
            self.args.width,
            self.args.height,
            self.args.fps,
            True,  # Force depth for meaningful overhead test
        )

        processing_modes = [
            ("raw_only", "Raw camera reads only"),
            ("standard", "CameraManager with depth processing"),
            ("with_colorization", "With depth colorization"),
            ("with_rerun", "With Rerun logging overhead"),
        ]

        try:
            cameras, camera_system = self.setup_cameras(configs, "processing_overhead")

            for mode_key, mode_desc in processing_modes:
                logger.info(f"Testing processing mode: {mode_desc}")

                result = self.measure_performance(
                    camera_system,
                    f"processing_{mode_key}",
                    timeout_ms=150,  # Standard timeout
                    processing_mode=mode_key,
                )
                result.update(
                    {
                        "test_category": "processing_overhead",
                        "width": self.args.width,
                        "height": self.args.height,
                        "fps": self.args.fps,
                        "use_depth": True,
                    }
                )
                self.results.append(result)

        except Exception as e:
            logger.error(f"Processing overhead test failed: {e}")
        finally:
            self.cleanup_cameras(camera_system)

    def run_timeout_optimization_test(self):
        """Test optimal timeout values for reliability vs responsiveness."""
        logger.info("Running timeout optimization test")

        # Use dual cameras if available for realistic load
        camera_count = min(2, len(self.available_cameras))
        selected_serials = self.available_cameras[:camera_count]
        configs = create_camera_configs(
            selected_serials, self.args.width, self.args.height, self.args.fps, self.args.use_depth
        )

        timeout_values = [50, 100, 150, 200, 300, 500, 1000]  # milliseconds

        try:
            cameras, camera_system = self.setup_cameras(configs, "timeout_optimization")

            for timeout_ms in timeout_values:
                result = self.measure_performance(
                    camera_system, f"timeout_optimization_{timeout_ms}ms", timeout_ms=timeout_ms
                )
                result.update(
                    {
                        "test_category": "timeout_optimization",
                        "width": self.args.width,
                        "height": self.args.height,
                        "fps": self.args.fps,
                        "use_depth": self.args.use_depth,
                    }
                )
                self.results.append(result)

        except Exception as e:
            logger.error(f"Timeout optimization test failed: {e}")
        finally:
            self.cleanup_cameras(camera_system)

    def run_rgb_vs_depth_comparison(self):
        """Compare RGB-only vs RGB+Depth performance."""
        logger.info("Running RGB vs Depth comparison test")

        # Test up to 2 cameras for meaningful comparison
        camera_count = min(2, len(self.available_cameras))
        selected_serials = self.available_cameras[:camera_count]

        for use_depth in [False, True]:
            mode_name = "RGB+Depth" if use_depth else "RGB-only"
            test_name = f"rgb_depth_comparison_{mode_name.lower().replace('+', '_')}"

            configs = create_camera_configs(
                selected_serials, self.args.width, self.args.height, self.args.fps, use_depth
            )

            try:
                cameras, camera_system = self.setup_cameras(configs, test_name)

                result = self.measure_performance(camera_system, test_name, timeout_ms=150)
                result.update(
                    {
                        "test_category": "rgb_vs_depth",
                        "width": self.args.width,
                        "height": self.args.height,
                        "fps": self.args.fps,
                        "use_depth": use_depth,
                    }
                )
                self.results.append(result)

            except Exception as e:
                logger.error(f"RGB vs Depth test failed for {mode_name}: {e}")
                continue
            finally:
                self.cleanup_cameras(camera_system)

    def save_results(self):
        """Save benchmark results to CSV file."""
        if not self.results:
            logger.warning("No results to save")
            return

        # Create output directory
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create DataFrame and save
        df = pd.DataFrame(self.results)

        # Add timestamp and hardware info
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"depth_camera_benchmark_{timestamp}.csv"

        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")

        # Print summary
        self.print_summary(df)

    def print_summary(self, df: pd.DataFrame):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("  DEPTH CAMERA BENCHMARK SUMMARY")
        print("=" * 60)

        for category in df["test_category"].unique():
            category_df = df[df["test_category"] == category]
            print(f"\n📊 {category.replace('_', ' ').title()}:")

            for _, row in category_df.iterrows():
                status = "✅" if row["avg_fps"] >= 25 else "⚠️" if row["avg_fps"] >= 15 else "❌"
                print(
                    f"  {status} {row['test_name']}: {row['avg_fps']:.1f} Hz "
                    f"({row['success_rate']:.1f}% success, {row['fps_std_dev']:.1f} std)"
                )

        print("\n🎯 Performance Targets:")
        print("   ✅ 25+ Hz: Excellent for real-time robotics")
        print("   ⚠️ 15-25 Hz: Acceptable for most applications")
        print("   ❌ <15 Hz: Too slow for responsive robotics")

        # Best configurations
        best_overall = df.loc[df["avg_fps"].idxmax()]
        print("\n🏆 Best Overall Performance:")
        print(f"   {best_overall['test_name']}: {best_overall['avg_fps']:.1f} Hz")

    def run_benchmark(self):
        """Run the complete benchmark suite."""
        logger.info("Starting depth camera benchmark")
        logger.info(f"Test duration: {self.args.test_duration}s per test")
        logger.info(f"Available cameras: {len(self.available_cameras)}")

        try:
            if self.args.test_type in ["all", "camera_count"]:
                self.run_camera_count_test()

            if self.args.test_type in ["all", "processing"]:
                self.run_processing_overhead_test()

            if self.args.test_type in ["all", "timeout"]:
                self.run_timeout_optimization_test()

            if self.args.test_type in ["all", "rgb_depth"]:
                self.run_rgb_vs_depth_comparison()

            self.save_results()

        except KeyboardInterrupt:
            logger.info("Benchmark interrupted by user")
            self.save_results()
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="Depth Camera Performance Benchmark")

    # Camera configuration
    parser.add_argument("--width", type=int, default=848, help="Camera width (default: 848)")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS (default: 30)")
    parser.add_argument("--use-depth", action="store_true", help="Enable depth stream")
    parser.add_argument("--max-cameras", type=int, default=4, help="Maximum cameras to test (default: 4)")

    # Test configuration
    parser.add_argument(
        "--test-type",
        choices=["all", "camera_count", "processing", "timeout", "rgb_depth"],
        default="all",
        help="Type of test to run (default: all)",
    )
    parser.add_argument(
        "--test-duration", type=float, default=10.0, help="Test duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--warmup-time", type=float, default=2.0, help="Camera warmup time in seconds (default: 2)"
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        nargs="+",
        default=[100, 150, 200],
        help="Timeout values to test in milliseconds (default: 100 150 200)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/depth_benchmark",
        help="Output directory for results (default: outputs/depth_benchmark)",
    )

    # Quick test mode
    parser.add_argument(
        "--quick-test", action="store_true", help="Run abbreviated test for hardware validation"
    )

    args = parser.parse_args()

    # Adjust parameters for quick test
    if args.quick_test:
        args.test_duration = 3.0
        args.warmup_time = 1.0
        args.timeout_ms = [150]
        args.test_type = "camera_count"
        logger.info("Running in quick test mode")

    try:
        benchmark = DepthCameraBenchmark(args)
        benchmark.run_benchmark()

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
