

import numpy as np
import pytest

from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.robot_devices.utils import RobotDeviceNotConnectedError, RobotDeviceAlreadyConnectedError


def test_camera():
    # Test instantiating with missing camera index raises an error
    with pytest.raises(ValueError):
        camera = OpenCVCamera()

    # Test instantiating with a wrong camera index raises an error
    with pytest.raises(ValueError):
        camera = OpenCVCamera(-1)

    # Test instantiating
    camera = OpenCVCamera(0)

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
    camera = OpenCVCamera(0)
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

    # Test reading asynchronously from the camera and image is similar
    async_color_image = camera.async_read()
    assert np.allclose(color_image, async_color_image)

    # Test disconnecting
    camera.disconnect()
    assert camera.camera is None
    assert camera.thread is None

    # Test disconnecting with `__del__`
    camera = OpenCVCamera(0)
    camera.connect()
    del camera

    # Test acquiring a bgr image
    camera = OpenCVCamera(0, color="bgr")
    camera.connect()
    assert camera.color == "bgr"
    bgr_color_image = camera.read()
    assert np.allclose(color_image, bgr_color_image[[2,1,0]])
    del camera

    # Test fps can be set
    camera = OpenCVCamera(0, fps=60)
    camera.connect()
    assert camera.fps == 60
    # TODO(rcadene): measure fps in nightly?
    del camera

    # Test width and height can be set
    camera = OpenCVCamera(0, fps=30, width=1280, height=720)
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





