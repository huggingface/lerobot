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

"""
Tests for CameraManager and factory functions.

Tests the camera management system that provides unified RGB and depth camera
handling for robots with parallel reading and automatic capability detection.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from lerobot.cameras.camera_manager import CameraManager, create_camera_system
from lerobot.cameras.opencv import OpenCVCameraConfig

# Test artifacts
TEST_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "cameras"
DEFAULT_PNG_FILE_PATH = TEST_ARTIFACTS_DIR / "image_160x120.png"


def test_create_camera_system_factory():
    """Test the factory function creates a CameraManager correctly."""
    camera_configs = {
        "test_cam": OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, width=160, height=120)
    }

    camera_system = create_camera_system(camera_configs)

    assert isinstance(camera_system, CameraManager)
    assert "test_cam" in camera_system.cameras
    assert "test_cam" in camera_system.camera_configs
    assert camera_system.camera_configs["test_cam"] == camera_configs["test_cam"]


def test_camera_manager_basic_initialization():
    """Test CameraManager initialization with single RGB camera."""
    # Create mock camera and config
    mock_camera = Mock()
    mock_camera.use_depth = False  # RGB only

    mock_config = Mock()
    mock_config.height = 480
    mock_config.width = 640

    cameras = {"rgb_cam": mock_camera}
    configs = {"rgb_cam": mock_config}

    manager = CameraManager(cameras, configs)

    assert manager.cameras == cameras
    assert manager.camera_configs == configs
    assert "rgb_cam" in manager.capabilities
    assert manager.capabilities["rgb_cam"]["type"] == "rgb"
    assert manager.capabilities["rgb_cam"]["has_depth"] is False


def test_camera_manager_depth_detection():
    """Test CameraManager detects depth capabilities correctly."""
    # Create mock depth camera
    mock_depth_camera = Mock()
    mock_depth_camera.use_depth = True  # Depth enabled

    mock_config = Mock()
    mock_config.height = 480
    mock_config.width = 640

    cameras = {"depth_cam": mock_depth_camera}
    configs = {"depth_cam": mock_config}

    manager = CameraManager(cameras, configs)

    assert manager.capabilities["depth_cam"]["type"] == "depth"
    assert manager.capabilities["depth_cam"]["has_depth"] is True
    assert "rgb" in manager.capabilities["depth_cam"]["streams"]
    assert "depth" in manager.capabilities["depth_cam"]["streams"]
    assert "depth_raw" in manager.capabilities["depth_cam"]["streams"]


def test_camera_manager_mixed_capabilities():
    """Test CameraManager with both RGB and depth cameras."""
    # RGB camera
    rgb_camera = Mock()
    rgb_camera.use_depth = False

    # Depth camera
    depth_camera = Mock()
    depth_camera.use_depth = True

    mock_config = Mock()
    mock_config.height = 480
    mock_config.width = 640

    cameras = {"rgb_cam": rgb_camera, "depth_cam": depth_camera}
    configs = {"rgb_cam": mock_config, "depth_cam": mock_config}

    manager = CameraManager(cameras, configs)

    # Check RGB camera
    assert manager.capabilities["rgb_cam"]["type"] == "rgb"
    assert manager.capabilities["rgb_cam"]["has_depth"] is False

    # Check depth camera
    assert manager.capabilities["depth_cam"]["type"] == "depth"
    assert manager.capabilities["depth_cam"]["has_depth"] is True


def test_get_features_rgb_only():
    """Test get_features returns correct features for RGB cameras."""
    mock_camera = Mock()
    mock_camera.use_depth = False

    mock_config = Mock()
    mock_config.height = 480
    mock_config.width = 640

    cameras = {"rgb_cam": mock_camera}
    configs = {"rgb_cam": mock_config}

    manager = CameraManager(cameras, configs)
    features = manager.get_features()

    assert "rgb_cam" in features
    assert features["rgb_cam"] == (480, 640, 3)
    assert "rgb_cam_depth" not in features


def test_get_features_with_depth():
    """Test get_features includes depth features for depth cameras."""
    mock_camera = Mock()
    mock_camera.use_depth = True

    mock_config = Mock()
    mock_config.height = 480
    mock_config.width = 640

    cameras = {"depth_cam": mock_camera}
    configs = {"depth_cam": mock_config}

    manager = CameraManager(cameras, configs)
    features = manager.get_features()

    assert "depth_cam" in features
    assert "depth_cam_depth" in features
    assert features["depth_cam"] == (480, 640, 3)  # RGB
    assert features["depth_cam_depth"] == (480, 640, 3)  # Colorized depth


@patch("lerobot.cameras.camera_manager.colorize_depth_frame")
def test_read_all_rgb_only(mock_colorize):
    """Test read_all with RGB-only cameras."""
    # Mock camera
    mock_camera = Mock()
    mock_camera.use_depth = False
    mock_camera.async_read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_config = Mock()
    mock_config.height = 480
    mock_config.width = 640

    cameras = {"rgb_cam": mock_camera}
    configs = {"rgb_cam": mock_config}

    manager = CameraManager(cameras, configs)

    with patch("threading.Thread") as mock_thread_class:
        # Setup thread mock
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        result = manager.read_all()

        # Should have RGB image
        assert "rgb_cam" in result
        assert isinstance(result["rgb_cam"], np.ndarray)

        # Should not have depth data
        assert "rgb_cam_depth" not in result
        assert "rgb_cam_depth_raw" not in result

        # Colorize should not be called for RGB-only cameras
        mock_colorize.assert_not_called()


@patch("lerobot.cameras.camera_manager.colorize_depth_frame")
def test_read_all_with_depth(mock_colorize):
    """Test read_all with depth cameras."""
    # Mock depth camera
    mock_camera = Mock()
    mock_camera.use_depth = True

    # Mock RGB and depth frames
    rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_frame = np.zeros((480, 640), dtype=np.uint16)
    mock_camera.async_read_rgb_and_depth.return_value = (rgb_frame, depth_frame)

    # Mock colorized depth
    colorized_depth = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_colorize.return_value = colorized_depth

    mock_config = Mock()
    mock_config.height = 480
    mock_config.width = 640

    cameras = {"depth_cam": mock_camera}
    configs = {"depth_cam": mock_config}

    manager = CameraManager(cameras, configs)

    with patch("threading.Thread") as mock_thread_class:
        # Setup thread mock
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        result = manager.read_all()

        # Should have RGB image
        assert "depth_cam" in result
        assert isinstance(result["depth_cam"], np.ndarray)

        # Should have colorized depth
        assert "depth_cam_depth" in result
        assert isinstance(result["depth_cam_depth"], np.ndarray)

        # Should have raw depth for Rerun
        assert "depth_cam_depth_raw" in result
        assert isinstance(result["depth_cam_depth_raw"], np.ndarray)

        # Verify depth colorization was called
        mock_colorize.assert_called_once_with(depth_frame)


def test_read_all_empty_cameras():
    """Test read_all with no cameras returns empty dict."""
    manager = CameraManager({}, {})
    result = manager.read_all()
    assert result == {}


def test_camera_manager_no_depth_attribute():
    """Test CameraManager handles cameras without use_depth attribute."""
    # Camera without use_depth attribute (defaults to False)
    mock_camera = Mock()
    del mock_camera.use_depth  # Remove the attribute

    mock_config = Mock()
    mock_config.height = 480
    mock_config.width = 640

    cameras = {"simple_cam": mock_camera}
    configs = {"simple_cam": mock_config}

    manager = CameraManager(cameras, configs)

    # Should default to RGB-only
    assert manager.capabilities["simple_cam"]["type"] == "rgb"
    assert manager.capabilities["simple_cam"]["has_depth"] is False


@pytest.mark.parametrize("timeout_ms", [50, 100, 200])
def test_read_all_timeout_parameter(timeout_ms):
    """Test read_all passes timeout parameter correctly."""
    mock_camera = Mock()
    mock_camera.use_depth = False
    mock_camera.async_read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_config = Mock()
    mock_config.height = 480
    mock_config.width = 640

    cameras = {"test_cam": mock_camera}
    configs = {"test_cam": mock_config}

    manager = CameraManager(cameras, configs)

    with patch("threading.Thread") as mock_thread_class:
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        manager.read_all(timeout_ms=timeout_ms)

        # Verify timeout was passed to camera
        mock_camera.async_read.assert_called_with(timeout_ms=timeout_ms)


def test_factory_function_integration():
    """Test the factory function creates a working system end-to-end."""
    camera_configs = {
        "test_cam": OpenCVCameraConfig(index_or_path=DEFAULT_PNG_FILE_PATH, width=160, height=120)
    }

    camera_system = create_camera_system(camera_configs)

    # Should be able to connect cameras
    for camera in camera_system.cameras.values():
        camera.connect(warmup=False)

    try:
        # Should be able to get features
        features = camera_system.get_features()
        assert "test_cam" in features
        assert features["test_cam"] == (120, 160, 3)

        # Should be able to read frames
        result = camera_system.read_all()
        assert "test_cam" in result
        assert isinstance(result["test_cam"], np.ndarray)

    finally:
        # Clean up
        for camera in camera_system.cameras.values():
            if camera.is_connected:
                camera.disconnect()
