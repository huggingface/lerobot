#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import cv2

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


def test_camera_performance(camera_id, width, height, fps):
    """Test a single camera's performance with specific settings."""
    print(f"\n🎥 Testing Camera {camera_id} at {width}x{height} @ {fps} FPS")

    # Test 1: Direct OpenCV
    print("  📹 Direct OpenCV test...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"    ❌ Failed to open camera {camera_id}")
        return None

    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Warm up
    for _ in range(5):
        ret, frame = cap.read()

    # Test direct read speed
    times = []
    for i in range(20):
        start = time.perf_counter()
        ret, frame = cap.read()
        if ret:
            times.append((time.perf_counter() - start) * 1000)
        else:
            print(f"    ❌ Read failed on iteration {i}")

    cap.release()

    if times:
        avg_direct = sum(times) / len(times)
        max_direct = max(times)
        min_direct = min(times)
        print(f"    ✅ Direct OpenCV: {avg_direct:.1f}ms avg ({min_direct:.1f}-{max_direct:.1f}ms)")
    else:
        print("    ❌ All direct reads failed")
        return None

    # Test 2: LeRobot OpenCV wrapper
    print("  🤖 LeRobot wrapper test...")
    try:
        config = OpenCVCameraConfig(
            index_or_path=camera_id,
            fps=fps,
            width=width,
            height=height,
            color_mode=ColorMode.RGB,
        )
        camera = OpenCVCamera(config)
        camera.connect()

        # Warm up
        for _ in range(5):
            try:
                camera.read()
            except Exception as e:
                print(f"    ⚠️  Warmup read failed: {e}")

        # Test read speed
        times = []
        for i in range(20):
            start = time.perf_counter()
            try:
                frame = camera.read()
                times.append((time.perf_counter() - start) * 1000)
            except Exception as e:
                print(f"    ❌ Read failed on iteration {i}: {e}")

        camera.disconnect()

        if times:
            avg_wrapper = sum(times) / len(times)
            max_wrapper = max(times)
            min_wrapper = min(times)
            print(f"    ✅ LeRobot wrapper: {avg_wrapper:.1f}ms avg ({min_wrapper:.1f}-{max_wrapper:.1f}ms)")
        else:
            print("    ❌ All wrapper reads failed")
            return None

        # Test 3: Async read
        print("  ⚡ Async read test...")
        camera.connect()

        # Let async thread start
        time.sleep(0.5)

        times = []
        for i in range(20):
            start = time.perf_counter()
            try:
                frame = camera.async_read(timeout_ms=200)
                times.append((time.perf_counter() - start) * 1000)
            except Exception as e:
                print(f"    ❌ Async read failed on iteration {i}: {e}")

        camera.disconnect()

        if times:
            avg_async = sum(times) / len(times)
            max_async = max(times)
            min_async = min(times)
            print(f"    ✅ Async read: {avg_async:.1f}ms avg ({min_async:.1f}-{max_async:.1f}ms)")
            return {"camera_id": camera_id, "direct": avg_direct, "wrapper": avg_wrapper, "async": avg_async}
        else:
            print("    ❌ All async reads failed")
            return None

    except Exception as e:
        print(f"    ❌ LeRobot wrapper failed: {e}")
        return None


def main():
    print("🔍 CAMERA PERFORMANCE DIAGNOSIS")
    print("=" * 50)

    # Test both cameras with current settings
    results = []

    # Test Camera 0 (gripper)
    result0 = test_camera_performance(0, 1280, 720, 30)
    if result0:
        results.append(result0)

    # Test Camera 1 (top)
    result1 = test_camera_performance(1, 1280, 720, 30)
    if result1:
        results.append(result1)

    # Test with reduced settings
    print("\n🔧 Testing with REDUCED settings (640x480 @ 25 FPS)...")

    result0_low = test_camera_performance(0, 640, 480, 25)
    if result0_low:
        results.append(result0_low)

    result1_low = test_camera_performance(1, 640, 480, 25)
    if result1_low:
        results.append(result1_low)

    # Summary
    print("\n📊 PERFORMANCE SUMMARY:")
    print("=" * 60)
    if results:
        for result in results:
            print(
                f"Camera {result['camera_id']:1d}: Direct={result['direct']:5.1f}ms, "
                f"Wrapper={result['wrapper']:5.1f}ms, Async={result['async']:5.1f}ms"
            )

        # Find the problematic camera
        slow_cameras = [r for r in results if r["async"] > 20]  # More than 20ms is slow
        if slow_cameras:
            print("\n⚠️  SLOW CAMERAS DETECTED:")
            for cam in slow_cameras:
                print(f"  - Camera {cam['camera_id']}: {cam['async']:.1f}ms (should be <10ms)")

        fast_cameras = [r for r in results if r["async"] <= 10]
        if fast_cameras:
            print("\n✅ GOOD CAMERAS:")
            for cam in fast_cameras:
                print(f"  - Camera {cam['camera_id']}: {cam['async']:.1f}ms")
    else:
        print("❌ No successful camera tests!")


if __name__ == "__main__":
    main()
