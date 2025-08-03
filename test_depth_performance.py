#!/usr/bin/env python3

"""
Comprehensive Depth Camera Performance Testing Script

Tests every component of the depth camera implementation to identify bottlenecks:
1. Individual camera performance
2. Dual camera USB bandwidth limits  
3. Processing overhead (colorization, Rerun)
4. Threading efficiency
5. Timeout optimization
6. Sequential vs parallel strategies

Usage: python test_depth_performance.py
"""

import time
import threading
import statistics
from typing import Dict, List, Tuple
import numpy as np

# LeRobot imports
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras.camera_manager import CameraManager
from lerobot.cameras.depth_utils import colorize_depth_frame, is_raw_depth
from lerobot.cameras.realsense import RealSenseCameraConfig


def print_header(title: str):
    """Print formatted test section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_results(test_name: str, fps_list: List[float], extra_info: str = ""):
    """Print formatted test results."""
    if not fps_list:
        print(f"❌ {test_name}: NO DATA")
        return
        
    avg_fps = statistics.mean(fps_list)
    min_fps = min(fps_list)
    max_fps = max(fps_list)
    std_fps = statistics.stdev(fps_list) if len(fps_list) > 1 else 0
    
    status = "✅" if avg_fps >= 25 else "⚠️" if avg_fps >= 15 else "❌"
    print(f"{status} {test_name}:")
    print(f"   Average: {avg_fps:.1f} Hz")
    print(f"   Range: {min_fps:.1f} - {max_fps:.1f} Hz")
    print(f"   Std Dev: {std_fps:.1f}")
    if extra_info:
        print(f"   {extra_info}")


class PerformanceTester:
    """Comprehensive performance testing suite."""
    
    def __init__(self):
        # Camera configurations from teleop.sh
        self.camera_configs = {
            "left_cam": RealSenseCameraConfig(
                serial_number_or_name="218622270973",
                width=848, height=480, fps=30, use_depth=True
            ),
            "right_cam": RealSenseCameraConfig(
                serial_number_or_name="218622278797", 
                width=848, height=480, fps=30, use_depth=True
            )
        }
        
        self.test_duration = 3.0  # seconds per test (reduced for less spam)
        self.warmup_time = 1.5    # seconds
        
    def setup_cameras(self, config_subset: Dict) -> Tuple[Dict, CameraManager]:
        """Setup cameras and camera manager for testing with proper threading."""
        cameras = make_cameras_from_configs(config_subset)
        camera_manager = CameraManager(cameras, config_subset)
        
        # Connect cameras
        print(f"   Connecting cameras...")
        camera_manager.connect_all()
        
        # Extended warmup for background threads to stabilize
        print(f"   Warming up cameras and background threads for {self.warmup_time}s...")
        time.sleep(self.warmup_time)
        
        # Verify all cameras are working before testing  
        print(f"   Verifying cameras... ", end="", flush=True)
        verified_count = 0
        for cam_name, cam in cameras.items():
            if not cam.is_connected:
                raise RuntimeError(f"Camera {cam_name} failed to connect")
            
            # Test single read to ensure background thread is working
            try:
                if camera_manager.capabilities[cam_name]['has_depth']:
                    # Test unified depth camera method
                    cam.async_read_rgb_and_depth(timeout_ms=1000)
                else:
                    cam.async_read(timeout_ms=1000)
                verified_count += 1
            except Exception as e:
                print(f"❌ {cam_name} failed: {e}")
                raise
        print(f"✅ {verified_count} cameras ready")
        
        return cameras, camera_manager
    
    def test_individual_cameras(self):
        """Test each camera individually for baseline performance."""
        print_header("Individual Camera Performance")
        
        for cam_name, cam_config in self.camera_configs.items():
            print(f"\n🔍 Testing {cam_name} individually...")
            
            try:
                # Test RGB-only performance (same camera, depth disabled)
                cam_config_rgb = RealSenseCameraConfig(
                    serial_number_or_name=cam_config.serial_number_or_name,
                    width=cam_config.width, height=cam_config.height, 
                    fps=cam_config.fps, use_depth=False  # Disable depth for baseline
                )
                
                cameras_rgb, manager_rgb = self.setup_cameras({cam_name: cam_config_rgb})
                fps_rgb = self._measure_fps(manager_rgb, f"{cam_name} RGB-only")
                manager_rgb.disconnect_all()
                
                # Test RGB+Depth performance (original config with depth enabled)
                cameras_depth, manager_depth = self.setup_cameras({cam_name: cam_config})
                fps_depth = self._measure_fps(manager_depth, f"{cam_name} RGB+Depth")
                manager_depth.disconnect_all()
                
                print_results(f"{cam_name} RGB-only", fps_rgb)
                print_results(f"{cam_name} RGB+Depth", fps_depth)
                
                # Calculate overhead if both tests succeeded
                if fps_rgb and fps_depth:
                    avg_rgb = statistics.mean(fps_rgb)
                    avg_depth = statistics.mean(fps_depth)
                    overhead = avg_rgb - avg_depth
                    print(f"   Depth overhead: {overhead:.1f} Hz")
                
            except Exception as e:
                print(f"❌ {cam_name}: Error - {e}")
    
    def test_dual_cameras(self):
        """Test both cameras together to identify USB bandwidth issues."""
        print_header("Dual Camera Performance")
        
        # Test RGB-only dual cameras
        print("\n🔍 Testing dual cameras RGB-only...")
        rgb_configs = {}
        for name, config in self.camera_configs.items():
            rgb_configs[name] = RealSenseCameraConfig(
                serial_number_or_name=config.serial_number_or_name,
                width=config.width, height=config.height,
                fps=config.fps, use_depth=False  # Disable depth for baseline
            )
        
        try:
            cameras_rgb, manager_rgb = self.setup_cameras(rgb_configs)
            fps_rgb_dual = self._measure_fps(manager_rgb, "Dual RGB-only")
            print_results("Dual RGB-only", fps_rgb_dual)
            manager_rgb.disconnect_all()
        except Exception as e:
            print(f"❌ Dual RGB-only: Error - {e}")
        
        # Test RGB+Depth dual cameras (current implementation)
        print("\n🔍 Testing dual cameras RGB+Depth...")
        try:
            cameras_depth, manager_depth = self.setup_cameras(self.camera_configs)
            fps_depth_dual = self._measure_fps(manager_depth, "Dual RGB+Depth")
            print_results("Dual RGB+Depth", fps_depth_dual)
            manager_depth.disconnect_all()
        except Exception as e:
            print(f"❌ Dual RGB+Depth: Error - {e}")
    
    def test_processing_overhead(self):
        """Test processing overhead of depth colorization and Rerun logging."""
        print_header("Processing Overhead Analysis")
        
        print("\n🔍 Testing depth processing overhead...")
        
        try:
            cameras, camera_manager = self.setup_cameras(self.camera_configs)
            
            # Test different processing scenarios
            scenarios = [
                ("Raw reads only", self._test_raw_reads_only),
                ("Raw + Colorization", self._test_with_colorization),
                ("Raw + Colorization + Rerun", self._test_with_rerun),
            ]
            
            for scenario_name, test_func in scenarios:
                fps_list = test_func(camera_manager)
                print_results(scenario_name, fps_list)
            
            camera_manager.disconnect_all()
            
        except Exception as e:
            print(f"❌ Processing overhead test: Error - {e}")
    
    def test_timeout_optimization(self):
        """Test different timeout values to find optimal settings."""
        print_header("Timeout Optimization")
        
        print("\n🔍 Testing different timeout values...")
        
        # Use realistic timeout values for depth cameras (avoid too short timeouts)
        timeout_values = [100, 150, 200, 300, 500]  # milliseconds
        
        try:
            cameras, camera_manager = self.setup_cameras(self.camera_configs)
            
            for timeout_ms in timeout_values:
                print(f"\n   Testing {timeout_ms}ms timeout...")
                fps_list = self._measure_fps_with_timeout(camera_manager, timeout_ms)
                print_results(f"Timeout {timeout_ms}ms", fps_list)
            
            camera_manager.disconnect_all()
            
        except Exception as e:
            print(f"❌ Timeout optimization test: Error - {e}")
    
    def test_threading_analysis(self):
        """Analyze camera background threading efficiency."""
        print_header("Threading Analysis")
        
        print("\n🔍 Analyzing camera background threads...")
        
        try:
            cameras, camera_manager = self.setup_cameras(self.camera_configs)
            
            # Test direct camera access vs CameraManager
            fps_direct = self._test_direct_camera_access(cameras)
            fps_manager = self._measure_fps(camera_manager, "CameraManager")
            
            print_results("Direct camera access", fps_direct)
            print_results("CameraManager access", fps_manager)
            
            # Thread contention test
            fps_parallel = self._test_parallel_access(cameras)
            print_results("Parallel camera access", fps_parallel)
            
            camera_manager.disconnect_all()
            
        except Exception as e:
            print(f"❌ Threading analysis: Error - {e}")
    
    def _measure_fps(self, camera_manager: CameraManager, test_name: str) -> List[float]:
        """Measure FPS for a camera manager with minimal console output."""
        fps_measurements = []
        start_time = time.perf_counter()
        last_time = start_time
        frame_count = 0
        error_count = 0
        
        print(f"   Testing {test_name}... ", end="", flush=True)
        
        while time.perf_counter() - start_time < self.test_duration:
            try:
                # Read all cameras with appropriate timeout
                observation = camera_manager.read_all(timeout_ms=150)
                
                # Only count successful reads with actual data
                if observation and any(isinstance(v, np.ndarray) for v in observation.values()):
                    current_time = time.perf_counter()
                    fps = 1.0 / (current_time - last_time)
                    
                    # Filter out unrealistic FPS values
                    if 1.0 <= fps <= 100.0:
                        fps_measurements.append(fps)
                        
                    last_time = current_time
                    frame_count += 1
                else:
                    error_count += 1
                
            except Exception as e:
                error_count += 1
                time.sleep(0.01)
                continue
        
        # Single summary line
        success_rate = (frame_count / (frame_count + error_count) * 100) if (frame_count + error_count) > 0 else 0
        print(f"Done ({frame_count} frames, {success_rate:.0f}% success)")
        
        return fps_measurements
    
    def _measure_fps_with_timeout(self, camera_manager: CameraManager, timeout_ms: int) -> List[float]:
        """Measure FPS with specific timeout."""
        fps_measurements = []
        start_time = time.perf_counter()
        last_time = start_time
        
        while time.perf_counter() - start_time < self.test_duration:
            try:
                observation = camera_manager.read_all(timeout_ms=timeout_ms)
                
                current_time = time.perf_counter()
                fps = 1.0 / (current_time - last_time)
                fps_measurements.append(fps)
                last_time = current_time
                
            except Exception:
                continue
        
        return fps_measurements
    
    def _test_raw_reads_only(self, camera_manager: CameraManager) -> List[float]:
        """Test raw camera reads without processing."""
        fps_measurements = []
        start_time = time.perf_counter()
        last_time = start_time
        
        while time.perf_counter() - start_time < self.test_duration:
            try:
                # Directly read cameras without processing
                for cam_name, cam in camera_manager.cameras.items():
                    if camera_manager.capabilities[cam_name]['has_depth']:
                        rgb, depth = cam.async_read_rgb_and_depth(timeout_ms=50)
                    else:
                        rgb = cam.async_read(timeout_ms=50)
                
                current_time = time.perf_counter()
                fps = 1.0 / (current_time - last_time)
                fps_measurements.append(fps)
                last_time = current_time
                
            except Exception:
                continue
        
        return fps_measurements
    
    def _test_with_colorization(self, camera_manager: CameraManager) -> List[float]:
        """Test with depth colorization."""
        fps_measurements = []
        start_time = time.perf_counter()
        last_time = start_time
        
        while time.perf_counter() - start_time < self.test_duration:
            try:
                # Read and colorize
                for cam_name, cam in camera_manager.cameras.items():
                    if camera_manager.capabilities[cam_name]['has_depth']:
                        rgb, depth = cam.async_read_rgb_and_depth(timeout_ms=50)
                        colorized = colorize_depth_frame(depth)  # Processing overhead
                    else:
                        rgb = cam.async_read(timeout_ms=50)
                
                current_time = time.perf_counter()
                fps = 1.0 / (current_time - last_time)
                fps_measurements.append(fps)
                last_time = current_time
                
            except Exception:
                continue
        
        return fps_measurements
    
    def _test_with_rerun(self, camera_manager: CameraManager) -> List[float]:
        """Test with Rerun visualization overhead."""
        import rerun as rr
        
        # Initialize Rerun for testing
        rr.init("depth_performance_test")
        
        fps_measurements = []
        start_time = time.perf_counter()
        last_time = start_time
        
        while time.perf_counter() - start_time < self.test_duration:
            try:
                # Full processing pipeline
                observation = camera_manager.read_all(timeout_ms=50)
                
                # Simulate Rerun logging
                for obs_name, obs_value in observation.items():
                    if isinstance(obs_value, np.ndarray):
                        if obs_name.endswith("_depth_raw") and is_raw_depth(obs_value):
                            rr.log(f"test.{obs_name}", rr.DepthImage(obs_value, meter=1.0/1000.0))
                        elif obs_name.endswith("_depth"):
                            continue  # Skip colorized as per current implementation
                        else:
                            rr.log(f"test.{obs_name}", rr.Image(obs_value))
                
                current_time = time.perf_counter()
                fps = 1.0 / (current_time - last_time)
                fps_measurements.append(fps)
                last_time = current_time
                
            except Exception:
                continue
        
        return fps_measurements
    
    def _test_direct_camera_access(self, cameras: Dict) -> List[float]:
        """Test direct camera access bypassing CameraManager."""
        fps_measurements = []
        start_time = time.perf_counter()
        last_time = start_time
        
        while time.perf_counter() - start_time < self.test_duration:
            try:
                # Direct access to cameras
                for cam_name, cam in cameras.items():
                    if getattr(cam, 'use_depth', False):
                        rgb, depth = cam.async_read_rgb_and_depth(timeout_ms=50)
                    else:
                        rgb = cam.async_read(timeout_ms=50)
                
                current_time = time.perf_counter()
                fps = 1.0 / (current_time - last_time)
                fps_measurements.append(fps)
                last_time = current_time
                
            except Exception:
                continue
        
        return fps_measurements
    
    def _test_parallel_access(self, cameras: Dict) -> List[float]:
        """Test parallel camera access with threading."""
        fps_measurements = []
        start_time = time.perf_counter()
        last_time = start_time
        
        def read_camera(cam_name, cam, results, timeout_ms=50):
            try:
                if getattr(cam, 'use_depth', False):
                    rgb, depth = cam.async_read_rgb_and_depth(timeout_ms=timeout_ms)
                    results[cam_name] = (rgb, depth)
                else:
                    rgb = cam.async_read(timeout_ms=timeout_ms)
                    results[cam_name] = rgb
            except Exception as e:
                results[cam_name] = None
        
        while time.perf_counter() - start_time < self.test_duration:
            try:
                results = {}
                threads = []
                
                # Start parallel reads
                for cam_name, cam in cameras.items():
                    thread = threading.Thread(target=read_camera, args=(cam_name, cam, results))
                    thread.start()
                    threads.append(thread)
                
                # Wait for all threads
                for thread in threads:
                    thread.join(timeout=0.1)  # 100ms max wait
                
                current_time = time.perf_counter()
                fps = 1.0 / (current_time - last_time)
                fps_measurements.append(fps)
                last_time = current_time
                
            except Exception:
                continue
        
        return fps_measurements
    
    def run_all_tests(self):
        """Run comprehensive test suite."""
        print_header("Depth Camera Performance Test Suite")
        print(f"Testing Duration: {self.test_duration}s per test")
        print(f"Warmup Time: {self.warmup_time}s per setup")
        print(f"Target Performance: 25+ Hz")
        
        # Run all test categories
        self.test_individual_cameras()
        self.test_dual_cameras()
        self.test_processing_overhead()
        self.test_timeout_optimization()
        self.test_threading_analysis()
        
        print_header("Test Suite Complete")
        print("🎯 Performance Analysis:")
        print("   • 25+ Hz = ✅ Excellent")
        print("   • 15-25 Hz = ⚠️ Acceptable")
        print("   • <15 Hz = ❌ Needs optimization")
        print("\n📊 Review results above to identify bottlenecks.")


def main():
    """Run the comprehensive performance test."""
    try:
        tester = PerformanceTester()
        tester.run_all_tests()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()