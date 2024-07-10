import numpy as np
import pytest

from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera, save_images_from_cameras
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from tests.utils import require_koch

CAMERA_INDEX = 2
# Maximum absolute difference between two consecutive images recored by a camera.
# This value differs with respect to the camera.
MAX_PIXEL_DIFFERENCE = 25


def compute_max_pixel_difference(first_image, second_image):
    return np.abs(first_image.astype(float) - second_image.astype(float)).max()


@require_koch
def test_camera(request):
    """Test assumes that `camera.read()` returns the same image when called multiple times in a row.
    So the environment should not change (you shouldnt be in front of the camera) and the camera should not be moving.

    Warning: The tests worked for a macbookpro camera, but I am getting assertion error (`np.allclose(color_image, async_color_image)`)
    for my iphone camera and my LG monitor camera.
    """
    # TODO(rcadene): measure fps in nightly?
    # TODO(rcadene): test logs
    # TODO(rcadene): add compatibility with other camera APIs

    # Test instantiating
    camera = OpenCVCamera(CAMERA_INDEX)

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
    camera = OpenCVCamera(CAMERA_INDEX)
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
    print(
        "max_pixel_difference between read() and async_read()",
        compute_max_pixel_difference(color_image, async_color_image),
    )
    assert np.allclose(color_image, async_color_image, rtol=1e-5, atol=MAX_PIXEL_DIFFERENCE)

    # Test disconnecting
    camera.disconnect()
    assert camera.camera is None
    assert camera.thread is None

    # Test disconnecting with `__del__`
    camera = OpenCVCamera(CAMERA_INDEX)
    camera.connect()
    del camera

    # Test acquiring a bgr image
    camera = OpenCVCamera(CAMERA_INDEX, color_mode="bgr")
    camera.connect()
    assert camera.color_mode == "bgr"
    bgr_color_image = camera.read()
    assert np.allclose(color_image, bgr_color_image[:, :, [2, 1, 0]], rtol=1e-5, atol=MAX_PIXEL_DIFFERENCE)
    del camera

    # TODO(rcadene): Add a test for a camera that doesnt support fps=60 and raises an OSError
    # TODO(rcadene): Add a test for a camera that supports fps=60

    # Test fps=10 raises an OSError
    camera = OpenCVCamera(CAMERA_INDEX, fps=10)
    with pytest.raises(OSError):
        camera.connect()
    del camera

    # Test width and height can be set
    camera = OpenCVCamera(CAMERA_INDEX, fps=30, width=1280, height=720)
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
    camera = OpenCVCamera(CAMERA_INDEX, fps=30, width=0, height=0)
    with pytest.raises(OSError):
        camera.connect()
    del camera


@require_koch
def test_save_images_from_cameras(tmpdir, request):
    save_images_from_cameras(tmpdir, record_time_s=1)
