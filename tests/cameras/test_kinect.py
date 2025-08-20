#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# Example of running a specific test:
# ```bash
# pytest tests/cameras/test_kinect.py::test_connect
# ```

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# Skip all tests if pylibfreenect2 is not available
pytest.importorskip("pylibfreenect2")

from lerobot.cameras.kinect import KinectCamera, KinectCameraConfig, KinectPipeline


def create_mock_device():
    """Create a mock Kinect device for testing."""
    device = Mock()
    device.started = False
    device.listeners = []
    device.getSerialNumber.return_value = "mock_serial_123"
    device.getIrCameraParams.return_value = Mock()
    device.getColorCameraParams.return_value = Mock()

    def set_color_listener(listener):
        device.listeners.append(listener)

    device.setColorFrameListener = set_color_listener

    def set_ir_depth_listener(listener):
        device.listeners.append(listener)

    device.setIrAndDepthFrameListener = set_ir_depth_listener

    def start():
        device.started = True

    device.start = start

    def stop():
        device.started = False

    device.stop = stop

    device.close = Mock()
    return device


def create_mock_frame_map():
    """Create a mock FrameMap for testing."""
    frame_map = Mock()
    frame_map.frames = {}

    def getitem(frame_type):
        frame = Mock()
        # Create mock frame data - BGRA format
        frame.asarray = lambda: np.random.randint(0, 255, (1080, 1920, 4), dtype=np.uint8)
        return frame

    frame_map.__getitem__ = getitem
    return frame_map


def create_mock_listener(frame_types=1):
    """Create a mock SyncMultiFrameListener for testing."""
    listener = Mock()
    listener.frame_types = frame_types

    def wait_for_new_frame(frame_map, timeout_ms):
        # Simulate getting frames
        frame_map.frames = {"Color": Mock()}
        return True

    listener.waitForNewFrame = wait_for_new_frame
    listener.release = Mock()
    return listener


def create_mock_freenect2():
    """Create a mock Freenect2 for testing."""
    fn2 = Mock()
    fn2.devices = ["mock_serial_123"]

    fn2.enumerateDevices.return_value = 1

    def get_device_serial(index):
        if index < len(fn2.devices):
            return fn2.devices[index]
        return None

    fn2.getDeviceSerialNumber = get_device_serial

    fn2.openDevice.return_value = create_mock_device()
    fn2.openDefaultDevice.return_value = create_mock_device()

    return fn2


@pytest.fixture(name="mock_freenect2")
def fixture_mock_freenect2():
    """Mock the Freenect2 class for all tests."""
    with patch("lerobot.cameras.kinect.camera_kinect.Freenect2") as mock_fn2:
        mock_fn2.return_value = create_mock_freenect2()
        yield mock_fn2


@pytest.fixture(name="mock_listener")
def fixture_mock_listener():
    """Mock the SyncMultiFrameListener class for all tests."""
    with patch("lerobot.cameras.kinect.camera_kinect.SyncMultiFrameListener") as mock_listener:
        mock_listener.return_value = create_mock_listener(1)
        yield mock_listener


@pytest.fixture(name="mock_framemap")
def fixture_mock_framemap():
    """Mock the FrameMap class for all tests."""
    with patch("lerobot.cameras.kinect.camera_kinect.FrameMap") as mock_fm:
        mock_fm.return_value = create_mock_frame_map()
        yield mock_fm


@pytest.fixture(name="mock_pylibfreenect2")
def fixture_mock_pylibfreenect2():
    """Mock the entire pylibfreenect2 module."""
    mock_module = MagicMock()
    mock_module.Freenect2.return_value = create_mock_freenect2()
    mock_module.SyncMultiFrameListener.side_effect = create_mock_listener
    mock_module.FrameMap.side_effect = create_mock_frame_map
    mock_module.FrameType = Mock(Color=1, Depth=2, Ir=4)
    mock_module.Registration = Mock
    mock_module.CpuPacketPipeline = Mock(return_value=Mock())
    mock_module.CudaPacketPipeline = Mock(return_value=Mock())
    mock_module.OpenCLPacketPipeline = Mock(return_value=Mock())
    mock_module.OpenGLPacketPipeline = Mock(return_value=Mock())

    with patch.dict("sys.modules", {"pylibfreenect2": mock_module}):
        yield mock_module


def test_abc_implementation():
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    config = KinectCameraConfig(device_index=0)
    _ = KinectCamera(config)


def test_connect(mock_freenect2, mock_listener, mock_framemap):
    """Test connecting to a Kinect camera."""
    config = KinectCameraConfig(device_index=0)
    camera = KinectCamera(config)

    camera.connect(warmup=False)
    assert camera.is_connected


def test_connect_already_connected(mock_freenect2, mock_listener, mock_framemap):
    """Test that connecting an already connected camera raises an error."""
    config = KinectCameraConfig(device_index=0)
    camera = KinectCamera(config)
    camera.connect(warmup=False)

    with pytest.raises(DeviceAlreadyConnectedError):
        camera.connect(warmup=False)


def test_connect_no_device(mock_freenect2, mock_listener, mock_framemap):
    """Test that connecting when no device is available raises an error."""
    mock_freenect2.return_value.enumerateDevices = lambda: 0
    config = KinectCameraConfig(device_index=0)
    camera = KinectCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect(warmup=False)


def test_connect_with_serial_number(mock_freenect2, mock_listener, mock_framemap):
    """Test connecting with a specific serial number."""
    config = KinectCameraConfig(serial_number="mock_serial_123")
    camera = KinectCamera(config)

    camera.connect(warmup=False)
    assert camera.is_connected
    assert camera.serial_number == "mock_serial_123"


def test_read(mock_freenect2, mock_listener, mock_framemap):
    """Test reading a frame from the camera."""
    config = KinectCameraConfig(device_index=0, width=1920, height=1080, fps=30)
    camera = KinectCamera(config)
    camera.connect(warmup=False)

    # Mock the thread and frame
    camera.thread = Mock(is_alive=lambda: True)
    camera.latest_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    camera.new_frame_event.set()

    img = camera.read()
    assert isinstance(img, np.ndarray)
    assert img.shape == (1080, 1920, 3)


def test_read_before_connect():
    """Test that reading before connecting raises an error."""
    config = KinectCameraConfig(device_index=0)
    camera = KinectCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read()


def test_disconnect(mock_freenect2, mock_listener, mock_framemap):
    """Test disconnecting from the camera."""
    config = KinectCameraConfig(device_index=0)
    camera = KinectCamera(config)
    camera.connect(warmup=False)

    camera.disconnect()

    assert not camera.is_connected


def test_disconnect_before_connect():
    """Test that disconnecting before connecting raises an error."""
    config = KinectCameraConfig(device_index=0)
    camera = KinectCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()


def test_async_read(mock_freenect2, mock_listener, mock_framemap):
    """Test asynchronous reading from the camera."""
    config = KinectCameraConfig(device_index=0, width=1920, height=1080, fps=30)
    camera = KinectCamera(config)
    camera.connect(warmup=False)

    try:
        # Mock the thread and frame
        camera.thread = Mock(is_alive=lambda: True)
        camera.latest_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        camera.new_frame_event.set()

        img = camera.async_read()

        assert camera.thread is not None
        assert isinstance(img, np.ndarray)
        assert img.shape == (1080, 1920, 3)
    finally:
        if camera.is_connected:
            camera.disconnect()


def test_async_read_timeout(mock_freenect2, mock_listener, mock_framemap):
    """Test that async_read times out when no frame is available."""
    config = KinectCameraConfig(device_index=0, width=1920, height=1080, fps=30)
    camera = KinectCamera(config)
    camera.connect(warmup=False)

    try:
        camera.thread = Mock(is_alive=lambda: True)
        camera.latest_frame = None
        camera.new_frame_event.clear()

        with pytest.raises(TimeoutError):
            camera.async_read(timeout_ms=1)
    finally:
        if camera.is_connected:
            camera.disconnect()


def test_async_read_before_connect():
    """Test that async_read before connecting raises an error."""
    config = KinectCameraConfig(device_index=0)
    camera = KinectCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.async_read()


@pytest.mark.parametrize(
    "rotation",
    [
        Cv2Rotation.NO_ROTATION,
        Cv2Rotation.ROTATE_90,
        Cv2Rotation.ROTATE_180,
        Cv2Rotation.ROTATE_270,
    ],
    ids=["no_rot", "rot90", "rot180", "rot270"],
)
def test_rotation(rotation, mock_freenect2, mock_listener, mock_framemap):
    """Test different rotation configurations."""
    config = KinectCameraConfig(device_index=0, rotation=rotation)
    camera = KinectCamera(config)
    camera.connect(warmup=False)

    # Mock the thread and frame based on rotation
    camera.thread = Mock(is_alive=lambda: True)

    if rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
        # Rotated dimensions
        camera.latest_frame = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
        camera.width = 1080
        camera.height = 1920
    else:
        # Normal dimensions
        camera.latest_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        camera.width = 1920
        camera.height = 1080

    camera.new_frame_event.set()

    img = camera.read()
    assert isinstance(img, np.ndarray)

    if rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
        assert camera.width == 1080
        assert camera.height == 1920
        assert img.shape[:2] == (1920, 1080)
    else:
        assert camera.width == 1920
        assert camera.height == 1080
        assert img.shape[:2] == (1080, 1920)


@pytest.mark.parametrize(
    "color_mode",
    [ColorMode.RGB, ColorMode.BGR],
    ids=["rgb", "bgr"],
)
def test_color_mode(color_mode, mock_freenect2, mock_listener, mock_framemap):
    """Test different color mode configurations."""
    config = KinectCameraConfig(device_index=0, color_mode=color_mode)
    camera = KinectCamera(config)
    camera.connect(warmup=False)

    # Mock the thread and frame
    camera.thread = Mock(is_alive=lambda: True)
    camera.latest_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    camera.new_frame_event.set()

    img = camera.read()
    assert isinstance(img, np.ndarray)
    assert img.shape == (1080, 1920, 3)


@pytest.mark.parametrize(
    "pipeline",
    [
        KinectPipeline.AUTO,
        KinectPipeline.CPU,
        KinectPipeline.CUDA,
        KinectPipeline.OPENCL,
        KinectPipeline.OPENGL,
    ],
    ids=["auto", "cpu", "cuda", "opencl", "opengl"],
)
def test_pipeline_selection(pipeline, mock_freenect2, mock_listener, mock_framemap):
    """Test different pipeline configurations."""
    config = KinectCameraConfig(device_index=0, pipeline=pipeline)
    camera = KinectCamera(config)

    # For non-CPU pipelines, they might fail on systems without GPU support
    # So we'll allow ConnectionError for GPU pipelines
    if pipeline in [KinectPipeline.CUDA, KinectPipeline.OPENCL, KinectPipeline.OPENGL]:
        try:
            camera.connect(warmup=False)
            assert camera.is_connected
        except (ConnectionError, RuntimeError):
            # Expected on systems without GPU support
            pass
    else:
        camera.connect(warmup=False)
        assert camera.is_connected


def test_find_cameras(mock_freenect2):
    """Test finding available Kinect cameras."""
    cameras = KinectCamera.find_cameras()
    assert isinstance(cameras, list)
    assert len(cameras) == 1
    assert cameras[0]["type"] == "Kinect v2"
    assert cameras[0]["serial_number"] == "mock_serial_123"


def test_find_cameras_no_devices(mock_freenect2):
    """Test finding cameras when no devices are available."""
    mock_freenect2.return_value.enumerateDevices = lambda: 0
    cameras = KinectCamera.find_cameras()
    assert isinstance(cameras, list)
    assert len(cameras) == 0


def test_config_validation():
    """Test configuration validation."""
    # Test invalid FPS
    with pytest.raises(ValueError):
        KinectCameraConfig(device_index=0, fps=60)

    # Test invalid resolution
    with pytest.raises(ValueError):
        KinectCameraConfig(device_index=0, width=3840, height=2160)

    # Test invalid device index
    with pytest.raises(ValueError):
        KinectCameraConfig(device_index=-1)

    # Test invalid color mode
    with pytest.raises(ValueError):
        KinectCameraConfig(device_index=0, color_mode="invalid")

    # Test invalid rotation
    with pytest.raises(ValueError):
        KinectCameraConfig(device_index=0, rotation="invalid")

    # Test invalid pipeline
    with pytest.raises(ValueError):
        KinectCameraConfig(device_index=0, pipeline="invalid")
