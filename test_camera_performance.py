#!/usr/bin/env python3
"""
Camera Performance Testing Script
Tests various optimization approaches for multi-camera systems.
"""

import logging
import time

import numpy as np
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_current_performance():
    """Test baseline performance with current implementation."""
    from lerobot.cameras.kinect import KinectCamera, KinectCameraConfig
    from lerobot.cameras.parallel_camera_reader import ParallelCameraReader
    from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

    print("\n" + "=" * 60)
    print("BASELINE PERFORMANCE TEST")
    print("=" * 60)

    # Configure cameras
    configs = {
        "cam_low": RealSenseCameraConfig(
            serial_number_or_name="218622270973",
            width=640,
            height=480,
            fps=30,
            use_depth=True,
            depth_colormap="jet",
        ),
        "cam_high": RealSenseCameraConfig(
            serial_number_or_name="218622278797",
            width=640,
            height=480,
            fps=30,
            use_depth=True,
            depth_colormap="jet",
        ),
        "cam_kinect": KinectCameraConfig(
            device_index=0,
            fps=30,
            use_depth=False,  # RGB only for baseline
        ),
    }

    # Create cameras
    cameras = {}
    for name, config in configs.items():
        if "kinect" in name:
            cameras[name] = KinectCamera(config)
        else:
            cameras[name] = RealSenseCamera(config)

    # Connect cameras
    print("\nConnecting cameras...")
    for name, cam in cameras.items():
        print(f"  Connecting {name}...")
        cam.connect()

    # Test parallel reading
    print("\nTesting parallel camera reading...")
    reader = ParallelCameraReader(persistent_executor=True)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        reader.read_cameras_parallel(cameras, timeout_ms=1000, with_depth=True)

    # Actual test
    print("\nRunning performance test (30 seconds)...")
    start_time = time.perf_counter()
    frame_times = []
    cpu_samples = []
    process = psutil.Process()

    test_duration = 30.0  # 30 seconds
    while time.perf_counter() - start_time < test_duration:
        frame_start = time.perf_counter()

        # Read cameras
        frames = reader.read_cameras_parallel(cameras, timeout_ms=1000, with_depth=True)

        frame_time = (time.perf_counter() - frame_start) * 1000
        frame_times.append(frame_time)

        # Sample CPU periodically
        if len(frame_times) % 10 == 0:
            cpu_samples.append(process.cpu_percent(interval=None))

    # Calculate statistics
    frame_times = np.array(frame_times)
    fps = 1000.0 / frame_times

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Frame Time: {frame_times.mean():.1f}ms ± {frame_times.std():.1f}ms")
    print(f"Min/Max: {frame_times.min():.1f}ms / {frame_times.max():.1f}ms")
    print(f"FPS: {fps.mean():.1f} ± {fps.std():.1f}")
    print(f"Process CPU: {np.mean(cpu_samples):.1f}%")
    print(f"System CPU: {psutil.cpu_percent(interval=1):.1f}%")

    # Get reader stats
    stats = reader.get_stats()
    print("\nParallel Reader Stats:")
    print(f"  Total reads: {stats['total_reads']}")
    print(f"  Failed reads: {stats['failed_reads']}")
    print(f"  Avg time: {stats['avg_time_ms']:.1f}ms")
    print(f"  Max time: {stats['max_time_ms']:.1f}ms")

    # Cleanup
    print("\nDisconnecting cameras...")
    for cam in cameras.values():
        cam.disconnect()

    return frame_times.mean(), np.mean(cpu_samples)


def test_thread_pool_sizes():
    """Test different thread pool sizes."""
    print("\n" + "=" * 60)
    print("THREAD POOL SIZE OPTIMIZATION TEST")
    print("=" * 60)

    from lerobot.cameras.parallel_camera_reader import ParallelCameraReader

    thread_counts = [3, 6, 8, 12, 16]
    results = {}

    for thread_count in thread_counts:
        print(f"\nTesting with {thread_count} threads...")

        # Create reader with specific thread count
        reader = ParallelCameraReader(max_workers=thread_count, persistent_executor=True)

        # Run simplified test (mock cameras for speed)
        class MockCamera:
            def __init__(self, delay_ms=30):
                self.delay = delay_ms / 1000.0
                self.use_depth = True

            def async_read(self, timeout_ms=1000):
                time.sleep(self.delay)  # Simulate camera read time
                return np.zeros((480, 640, 3), dtype=np.uint8)

            def async_read_all(self, timeout_ms=1000):
                time.sleep(self.delay)
                return {
                    "color": np.zeros((480, 640, 3), dtype=np.uint8),
                    "depth_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
                }

        # Create mock cameras
        cameras = {
            "cam1": MockCamera(30),  # 30ms read time (30fps)
            "cam2": MockCamera(30),
            "cam3": MockCamera(30),
        }

        # Test
        times = []
        for _ in range(100):
            start = time.perf_counter()
            reader.read_cameras_parallel(cameras, timeout_ms=1000, with_depth=True)
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        results[thread_count] = avg_time
        print(f"  Average time: {avg_time:.1f}ms")

    # Find optimal thread count
    best_threads = min(results, key=results.get)
    print(f"\n✅ Optimal thread count: {best_threads} ({results[best_threads]:.1f}ms)")

    return results


def test_memory_optimization():
    """Test pre-allocated memory buffers."""
    print("\n" + "=" * 60)
    print("MEMORY OPTIMIZATION TEST")
    print("=" * 60)

    # Test memory allocation overhead
    sizes = [(480, 640, 3), (1080, 1920, 3), (424, 512, 1)]  # Common camera resolutions

    for size in sizes:
        # Test allocation time
        alloc_times = []
        for _ in range(1000):
            start = time.perf_counter()
            arr = np.zeros(size, dtype=np.uint8)
            alloc_times.append((time.perf_counter() - start) * 1000)

        print(f"Resolution {size}: {np.mean(alloc_times):.3f}ms per allocation")

    # Test with pre-allocated buffers
    print("\nTesting with pre-allocated buffers...")
    buffer_pool = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]

    copy_times = []
    for _ in range(1000):
        src = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dst = buffer_pool[0]

        start = time.perf_counter()
        np.copyto(dst, src)
        copy_times.append((time.perf_counter() - start) * 1000)

    print(f"Pre-allocated buffer copy: {np.mean(copy_times):.3f}ms")

    return np.mean(alloc_times), np.mean(copy_times)


def main():
    """Run all performance tests."""
    print("\n" + "=" * 60)
    print("CAMERA SYSTEM PERFORMANCE ANALYZER")
    print("=" * 60)
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"CPU Frequency: {psutil.cpu_freq().current:.0f}MHz")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")

    results = {}

    # Test 1: Baseline performance
    try:
        avg_time, avg_cpu = test_current_performance()
        results["baseline"] = {"time": avg_time, "cpu": avg_cpu}
    except Exception as e:
        print(f"❌ Baseline test failed: {e}")

    # Test 2: Thread pool sizes
    try:
        thread_results = test_thread_pool_sizes()
        results["threads"] = thread_results
    except Exception as e:
        print(f"❌ Thread pool test failed: {e}")

    # Test 3: Memory optimization
    try:
        alloc_time, copy_time = test_memory_optimization()
        results["memory"] = {"alloc": alloc_time, "copy": copy_time}
    except Exception as e:
        print(f"❌ Memory test failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    if "baseline" in results:
        print(f"Baseline: {results['baseline']['time']:.1f}ms @ {results['baseline']['cpu']:.1f}% CPU")

    if "threads" in results:
        best_threads = min(results["threads"], key=results["threads"].get)
        print(f"Best thread count: {best_threads} ({results['threads'][best_threads]:.1f}ms)")

    if "memory" in results:
        print(
            f"Memory overhead: {results['memory']['alloc']:.3f}ms alloc, {results['memory']['copy']:.3f}ms copy"
        )

    print("\n✅ Testing complete! Check camera_optimization_tracker.md for next steps.")


if __name__ == "__main__":
    main()
