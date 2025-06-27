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

import pytest

from lerobot.common.cameras.configs import ColorMode
from lerobot.common.cameras.depthai.camera_depthai import DepthAICamera
from lerobot.common.cameras.depthai.configuration_depthai import DepthAICameraConfig
from lerobot.common.errors import DeviceNotConnectedError

pytest.importorskip("depthai", reason="depthai is not installed")


@pytest.fixture
def mock_depthai_camera_config():
    return DepthAICameraConfig(
        mxid_or_name="14442C10D13EABF200",  # Example MxID
        color_mode=ColorMode.RGB,
        fps=30,
        width=640,
        height=480,
        enable_rgb=True,
        enable_depth=False,
    )


def test_depthai_camera_config():
    config = DepthAICameraConfig(
        mxid_or_name="14442C10D13EABF200",
        color_mode=ColorMode.RGB,
        fps=30,
        width=640,
        height=480,
        enable_rgb=True,
        enable_depth=True,
    )
    assert config.mxid_or_name == "14442C10D13EABF200"
    assert config.color_mode == ColorMode.RGB
    assert config.fps == 30
    assert config.width == 640
    assert config.height == 480
    assert config.enable_rgb is True
    assert config.enable_depth is True


def test_depthai_camera_config_validation():
    # Test that both enable_rgb and enable_depth cannot be False
    with pytest.raises(ValueError, match="At least one of enable_rgb or enable_depth must be True"):
        DepthAICameraConfig(
            mxid_or_name="14442C10D13EABF200",
            enable_rgb=False,
            enable_depth=False,
        )


def test_depthai_camera_config_resolution_consistency():
    # Test that fps, width, height must all be set or all be None
    with pytest.raises(ValueError, match="fps, width, and height must all be set or all be None"):
        DepthAICameraConfig(
            mxid_or_name="14442C10D13EABF200",
            fps=30,
            width=640,
            # height is None - should raise error
        )


def test_depthai_camera_init(mock_depthai_camera_config):
    # Test camera initialization (doesn't require actual hardware)
    camera = DepthAICamera(mock_depthai_camera_config)
    assert camera.config == mock_depthai_camera_config
    assert camera.fps == 30
    assert camera.width == 640
    assert camera.height == 480
    assert camera.enable_rgb is True
    assert camera.enable_depth is False


def test_depthai_camera_find_cameras():
    # Test camera discovery (should work without hardware, just returns empty list)
    cameras = DepthAICamera.find_cameras()
    assert isinstance(cameras, list)
    # We can't assert specific cameras since this depends on hardware
    # But we can check the structure if cameras are found
    for camera in cameras:
        assert "name" in camera
        assert "type" in camera
        assert "id" in camera
        assert camera["type"] == "DepthAI"


def test_depthai_camera_not_connected_operations(mock_depthai_camera_config):
    # Test operations that should fail when camera is not connected
    camera = DepthAICamera(mock_depthai_camera_config)

    with pytest.raises(DeviceNotConnectedError):
        camera.read()

    with pytest.raises(DeviceNotConnectedError):
        camera.async_read()

    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()


@pytest.mark.skipif(True, reason="Requires actual DepthAI hardware - integration test only")
def test_depthai_camera_integration():
    """
    Integration test that requires actual DepthAI hardware.

    This test is skipped by default but can be enabled manually
    for testing with real hardware.
    """
    # Find available cameras
    cameras = DepthAICamera.find_cameras()
    if not cameras:
        pytest.skip("No DepthAI cameras found")

    # Use first available camera
    camera_id = cameras[0]["id"]
    config = DepthAICameraConfig(
        mxid_or_name=camera_id,
        color_mode=ColorMode.RGB,
        fps=30,
        width=640,
        height=480,
        enable_rgb=True,
        enable_depth=False,
    )

    camera = DepthAICamera(config)

    try:
        # Test connection
        camera.connect()
        assert camera.is_connected

        # Test reading frames
        frame = camera.read()
        assert frame.shape == (480, 640, 3)

        # Test async reading
        async_frame = camera.async_read()
        assert async_frame.shape == (480, 640, 3)

    finally:
        # Ensure cleanup
        if camera.is_connected:
            camera.disconnect()
