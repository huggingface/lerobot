"""
Tests for physical cameras and their mocked versions.
If the physical camera is not connected to the computer, or not working,
the test will be skipped.

Example of running a specific test:
```bash
pytest -sx tests/test_cameras.py::test_camera
```

Example of running test on a real camera connected to the computer:
```bash
pytest -sx 'tests/test_cameras.py::test_camera[opencv-False]'
pytest -sx 'tests/test_cameras.py::test_camera[intelrealsense-False]'
```

Example of running test on a mocked version of the camera:
```bash
pytest -sx 'tests/test_cameras.py::test_camera[opencv-True]'
pytest -sx 'tests/test_cameras.py::test_camera[intelrealsense-True]'
```
"""

import numpy as np
import pytest

from lerobot import available_cameras
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from tests.utils import make_camera, require_camera, require_mock_camera

# Maximum absolute difference between two consecutive images recored by a camera.
# This value differs with respect to the camera.
MAX_PIXEL_DIFFERENCE = 25


def compute_max_pixel_difference(first_image, second_image):
    return np.abs(first_image.astype(float) - second_image.astype(float)).max()


def _test_camera(camera_type):
    """Test assumes that `camera.read()` returns the same image when called multiple times in a row.
    So the environment should not change (you shouldnt be in front of the camera) and the camera should not be moving.

    Warning: The tests worked for a macbookpro camera, but I am getting assertion error (`np.allclose(color_image, async_color_image)`)
    for my iphone camera and my LG monitor camera.
    """
    # TODO(rcadene): measure fps in nightly?
    # TODO(rcadene): test logs

    # Test instantiating
    camera = make_camera(camera_type)

    # Test reading, async reading, disconnecting before connecting raises an error
    with pytest.raises(RobotDeviceNotConnectedError):
        camera.read()
    with pytest.raises(RobotDeviceNotConnectedError):
        camera.async_read()
    with pytest.raises(RobotDeviceNotConnectedError):
        camera.disconnect()

    # Test deleting the object without connecting first
    del camera

    # Test connecting
    camera = make_camera(camera_type)
    camera.connect()
    assert camera.is_connected
    assert camera.fps is not None
    assert camera.width is not None
    assert camera.height is not None

    # Test connecting twice raises an error
    with pytest.raises(RobotDeviceAlreadyConnectedError):
        camera.connect()

    # Test reading from the camera
    color_image = camera.read()
    assert isinstance(color_image, np.ndarray)
    assert color_image.ndim == 3
    h, w, c = color_image.shape
    assert c == 3
    assert w > h

    # Test read and async_read outputs similar images
    # ...warming up as the first frames can be black
    for _ in range(30):
        camera.read()
    color_image = camera.read()
    async_color_image = camera.async_read()
    error_msg = (
        "max_pixel_difference between read() and async_read()",
        compute_max_pixel_difference(color_image, async_color_image),
    )
    # TODO(rcadene): properly set `rtol`
    np.testing.assert_allclose(
        color_image, async_color_image, rtol=1e-5, atol=MAX_PIXEL_DIFFERENCE, err_msg=error_msg
    )

    # Test disconnecting
    camera.disconnect()
    assert camera.camera is None
    assert camera.thread is None

    # Test disconnecting with `__del__`
    camera = make_camera(camera_type)
    camera.connect()
    del camera

    # Test acquiring a bgr image
    camera = make_camera(camera_type, color_mode="bgr")
    camera.connect()
    assert camera.color_mode == "bgr"
    bgr_color_image = camera.read()
    np.testing.assert_allclose(
        color_image, bgr_color_image[:, :, [2, 1, 0]], rtol=1e-5, atol=MAX_PIXEL_DIFFERENCE, err_msg=error_msg
    )
    del camera

    # TODO(rcadene): Add a test for a camera that doesnt support fps=60 and raises an OSError
    # TODO(rcadene): Add a test for a camera that supports fps=60

    # Test width and height can be set
    camera = make_camera(camera_type, fps=30, width=1280, height=720)
    camera.connect()
    assert camera.fps == 30
    assert camera.width == 1280
    assert camera.height == 720
    color_image = camera.read()
    h, w, c = color_image.shape
    assert h == 720
    assert w == 1280
    assert c == 3
    del camera

    # Test not supported width and height raise an error
    camera = make_camera(camera_type, fps=30, width=0, height=0)
    with pytest.raises(OSError):
        camera.connect()
    del camera


def _test_save_images_from_cameras(tmpdir, camera_type):
    # TODO(rcadene): refactor
    if camera_type == "opencv":
        from lerobot.common.robot_devices.cameras.opencv import save_images_from_cameras
    elif camera_type == "intelrealsense":
        from lerobot.common.robot_devices.cameras.intelrealsense import save_images_from_cameras

    save_images_from_cameras(tmpdir, record_time_s=1)


@pytest.mark.parametrize("camera_type", available_cameras)
@require_mock_camera
def test_camera_mock(monkeypatch, camera_type):
    _test_camera(camera_type)


@pytest.mark.parametrize("camera_type", available_cameras)
@require_camera
def test_camera(request, camera_type):
    _test_camera(camera_type)


@pytest.mark.parametrize("camera_type", available_cameras)
@require_mock_camera
def test_save_images_from_cameras_mock(tmpdir, monkeypatch, camera_type):
    _test_save_images_from_cameras(tmpdir, camera_type)


@pytest.mark.parametrize("camera_type", available_cameras)
@require_camera
def test_save_images_from_cameras(tmpdir, request, camera_type):
    _test_save_images_from_cameras(tmpdir, camera_type)
