#!/usr/bin/env python3
"""
Comprehensive test script for ZMQ camera integration.
Tests all camera functionalities: find_cameras, connect, read, async_read, disconnect.
"""

import time
import numpy as np
from pathlib import Path

from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig
from lerobot.cameras.configs import ColorMode


def test_find_cameras():
    """Test 1: Camera Discovery"""
    print("\n" + "="*60)
    print("TEST 1: Camera Discovery (find_cameras)")
    print("="*60)
    
    cameras = ZMQCamera.find_cameras()
    
    if not cameras:
        print("❌ No ZMQ cameras found!")
        print("   Make sure you have configured cameras in ~/.lerobot/zmq_cameras.json")
        print("   or set LEROBOT_ZMQ_CAMERAS environment variable")
        return None
    
    print(f"✅ Found {len(cameras)} ZMQ camera(s):")
    for i, cam in enumerate(cameras):
        print(f"\n  Camera {i}:")
        for key, value in cam.items():
            if key == "default_stream_profile":
                print(f"    {key}:")
                for sub_key, sub_value in value.items():
                    print(f"      {sub_key}: {sub_value}")
            else:
                print(f"    {key}: {value}")
    
    return cameras[0] if cameras else None


def test_connect_disconnect(cam_info):
    """Test 2: Connect and Disconnect"""
    print("\n" + "="*60)
    print("TEST 2: Connect and Disconnect")
    print("="*60)
    
    config = ZMQCameraConfig(
        server_address=cam_info["server_address"],
        port=cam_info["port"],
        camera_name=cam_info["camera_name"],
        color_mode=ColorMode.RGB,
    )
    
    camera = ZMQCamera(config)
    
    # Test is_connected before connection
    print(f"Before connect - is_connected: {camera.is_connected}")
    assert not camera.is_connected, "❌ Camera should not be connected initially"
    
    # Test connect
    print("Connecting to camera...")
    start = time.time()
    camera.connect(warmup=True)
    connect_time = time.time() - start
    print(f"✅ Connected in {connect_time:.2f}s")
    print(f"   Camera resolution: {camera.width}x{camera.height}")
    assert camera.is_connected, "❌ Camera should be connected after connect()"
    
    # Test disconnect
    print("Disconnecting camera...")
    camera.disconnect()
    print("✅ Disconnected")
    assert not camera.is_connected, "❌ Camera should not be connected after disconnect()"
    
    return config


def test_synchronous_read(config):
    """Test 3: Synchronous Read"""
    print("\n" + "="*60)
    print("TEST 3: Synchronous Read (read)")
    print("="*60)
    
    camera = ZMQCamera(config)
    camera.connect(warmup=False)
    
    print("Reading 10 frames synchronously...")
    read_times = []
    
    for i in range(10):
        start = time.time()
        frame = camera.read()
        read_time = (time.time() - start) * 1000  # ms
        read_times.append(read_time)
        
        # Validate frame
        assert isinstance(frame, np.ndarray), "❌ Frame should be numpy array"
        assert frame.ndim == 3, "❌ Frame should be 3D (H, W, C)"
        assert frame.shape[2] == 3, "❌ Frame should have 3 channels"
        
        if i == 0:
            print(f"   Frame shape: {frame.shape}")
            print(f"   Frame dtype: {frame.dtype}")
            print(f"   Frame range: [{frame.min()}, {frame.max()}]")
    
    avg_time = np.mean(read_times)
    std_time = np.std(read_times)
    fps = 1000 / avg_time if avg_time > 0 else 0
    
    print(f"✅ Read 10 frames successfully")
    print(f"   Average read time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   Estimated FPS: {fps:.1f}")
    
    camera.disconnect()


def test_color_mode_conversion(config):
    """Test 4: Color Mode Conversion"""
    print("\n" + "="*60)
    print("TEST 4: Color Mode Conversion")
    print("="*60)
    
    # Test RGB mode
    config_rgb = ZMQCameraConfig(
        server_address=config.server_address,
        port=config.port,
        camera_name=config.camera_name,
        color_mode=ColorMode.RGB,
    )
    camera_rgb = ZMQCamera(config_rgb)
    camera_rgb.connect(warmup=False)
    frame_rgb = camera_rgb.read()
    camera_rgb.disconnect()
    
    # Test BGR mode
    config_bgr = ZMQCameraConfig(
        server_address=config.server_address,
        port=config.port,
        camera_name=config.camera_name,
        color_mode=ColorMode.BGR,
    )
    camera_bgr = ZMQCamera(config_bgr)
    camera_bgr.connect(warmup=False)
    frame_bgr = camera_bgr.read()
    camera_bgr.disconnect()
    
    # Frames should be different (color channels swapped)
    assert not np.array_equal(frame_rgb, frame_bgr), "❌ RGB and BGR frames should differ"
    # But shapes should be the same
    assert frame_rgb.shape == frame_bgr.shape, "❌ RGB and BGR frames should have same shape"
    
    print("✅ RGB mode works correctly")
    print("✅ BGR mode works correctly")
    print(f"   Frame shapes match: {frame_rgb.shape}")


def test_asynchronous_read(config):
    """Test 5: Asynchronous Read"""
    print("\n" + "="*60)
    print("TEST 5: Asynchronous Read (async_read)")
    print("="*60)
    
    camera = ZMQCamera(config)
    camera.connect(warmup=False)
    
    print("Reading 10 frames asynchronously...")
    read_times = []
    
    for i in range(10):
        start = time.time()
        frame = camera.async_read(timeout_ms=1000)
        read_time = (time.time() - start) * 1000  # ms
        read_times.append(read_time)
        
        # Validate frame
        assert isinstance(frame, np.ndarray), "❌ Frame should be numpy array"
        assert frame.ndim == 3, "❌ Frame should be 3D (H, W, C)"
        
        if i == 0:
            print(f"   Frame shape: {frame.shape}")
            print(f"   Background thread alive: {camera.thread.is_alive()}")
    
    avg_time = np.mean(read_times)
    std_time = np.std(read_times)
    fps = 1000 / avg_time if avg_time > 0 else 0
    
    print(f"✅ Read 10 frames asynchronously")
    print(f"   Average read time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   Estimated FPS: {fps:.1f}")
    
    camera.disconnect()


def test_reconnection(config):
    """Test 6: Reconnection"""
    print("\n" + "="*60)
    print("TEST 6: Reconnection")
    print("="*60)
    
    camera = ZMQCamera(config)
    
    # Connect, read, disconnect cycle 1
    print("Connection cycle 1...")
    camera.connect(warmup=False)
    frame1 = camera.read()
    camera.disconnect()
    
    # Wait a bit
    time.sleep(0.5)
    
    # Connect, read, disconnect cycle 2
    print("Connection cycle 2...")
    camera.connect(warmup=False)
    frame2 = camera.read()
    camera.disconnect()
    
    # Frames should be valid
    assert frame1.shape == frame2.shape, "❌ Frames should have consistent shape"
    
    print("✅ Reconnection works correctly")
    print(f"   Both frames have shape: {frame1.shape}")


def test_timeout_handling(config):
    """Test 7: Timeout Handling"""
    print("\n" + "="*60)
    print("TEST 7: Timeout Handling")
    print("="*60)
    
    # Create config with very short timeout
    config_short = ZMQCameraConfig(
        server_address=config.server_address,
        port=config.port,
        camera_name=config.camera_name,
        timeout_ms=50,  # Very short timeout
    )
    
    camera = ZMQCamera(config_short)
    camera.connect(warmup=False)
    
    try:
        # This should work if server is fast enough
        frame = camera.read()
        print(f"✅ Read succeeded even with {config_short.timeout_ms}ms timeout")
    except TimeoutError:
        print(f"⚠️  Timeout occurred (expected with {config_short.timeout_ms}ms timeout)")
    
    camera.disconnect()


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ZMQ CAMERA COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Discovery
        cam_info = test_find_cameras()
        if not cam_info:
            print("\n❌ Cannot proceed without configured cameras")
            return
        
        # Test 2: Connect/Disconnect
        config = test_connect_disconnect(cam_info)
        
        # Test 3: Synchronous Read
        test_synchronous_read(config)
        
        # Test 4: Color Mode
        test_color_mode_conversion(config)
        
        # Test 5: Asynchronous Read
        test_asynchronous_read(config)
        
        # Test 6: Reconnection
        test_reconnection(config)
        
        # Test 7: Timeout
        test_timeout_handling(config)
        
        # Summary
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nZMQ Camera integration is working correctly! 🎉")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

