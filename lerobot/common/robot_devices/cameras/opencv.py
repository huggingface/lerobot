"""
This file contains utilities for recording frames from cameras. For more info look at `OpenCVCamera` docstring.
"""

import math
import platform
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)
from lerobot.common.utils.utils import capture_timestamp_utc

# Use 1 thread to avoid blocking the main thread. Especially useful during data collection
# when other threads are used to save the images.
cv2.setNumThreads(1)

# The maximum opencv device index depends on your operating system. For instance,
# if you have 3 cameras, they should be associated to index 0, 1, and 2. This is the case
# on MacOS. However, on Ubuntu, the indices are different like 6, 16, 23.
# When you change the USB port or reboot the computer, the operating system might
# treat the same cameras as new devices. Thus we select a higher bound to search indices.
MAX_OPENCV_INDEX = 60


def find_camera_indices(raise_when_empty=False, max_index_search_range=MAX_OPENCV_INDEX):
    if platform.system() == "Linux":
        # Linux uses camera ports
        print("Linux detected. Finding available camera indices through scanning '/dev/video*' ports")
        possible_camera_ids = []
        for port in Path("/dev").glob("video*"):
            camera_idx = int(str(port).replace("/dev/video", ""))
            possible_camera_ids.append(camera_idx)
    else:
        print(
            "Mac or Windows detected. Finding available camera indices through "
            f"scanning all indices from 0 to {MAX_OPENCV_INDEX}"
        )
        possible_camera_ids = range(max_index_search_range)

    camera_ids = []
    for camera_idx in possible_camera_ids:
        camera = cv2.VideoCapture(camera_idx)
        is_open = camera.isOpened()
        camera.release()

        if is_open:
            print(f"Camera found at index {camera_idx}")
            camera_ids.append(camera_idx)

    if raise_when_empty and len(camera_ids) == 0:
        raise OSError(
            "Not a single camera was detected. Try re-plugging, or re-installing `opencv2`, "
            "or your camera driver, or make sure your camera is compatible with opencv2."
        )

    return camera_ids


@dataclass
class OpenCVCameraConfig:
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    OpenCVCameraConfig(30, 640, 480)
    OpenCVCameraConfig(60, 640, 480)
    OpenCVCameraConfig(90, 640, 480)
    OpenCVCameraConfig(30, 1280, 720)
    ```
    """

    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"

    def __post_init__(self):
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )


class OpenCVCamera:
    """
    The OpenCVCamera class allows to efficiently record images from cameras. It relies on opencv2 to communicate
    with the cameras. Most cameras are compatible. For more info, see the [Video I/O with OpenCV Overview](https://docs.opencv.org/4.x/d0/da7/videoio_overview.html).

    An OpenCVCamera instance requires a camera index (e.g. `OpenCVCamera(camera_index=0)`). When you only have one camera
    like a webcam of a laptop, the camera index is expected to be 0, but it might also be very different, and the camera index
    might change if you reboot your computer or re-plug your camera. This behavior depends on your operation system.

    To find the camera indices of your cameras, you can run our utility script that will be save a few frames for each camera:
    ```bash
    python lerobot/scripts/save_images_from_cameras.py \
    --driver opencv \
    --images-dir outputs/images_from_opencv_cameras
    ```

    When an OpenCVCamera is instantiated, if no specific config is provided, the default fps, width, height and color_mode
    of the given camera will be used.

    Example of usage:
    ```python
    camera = OpenCVCamera(camera_index=0)
    camera.connect()
    color_image = camera.read()
    # when done using the camera, consider disconnecting
    camera.disconnect()
    ```

    Example of changing default fps, width, height and color_mode:
    ```python
    camera = OpenCVCamera(0, fps=30, width=1280, height=720)
    camera = connect()  # applies the settings, might error out if these settings are not compatible with the camera

    camera = OpenCVCamera(0, fps=90, width=640, height=480)
    camera = connect()

    camera = OpenCVCamera(0, fps=90, width=640, height=480, color_mode="bgr")
    camera = connect()
    ```
    """

    def __init__(self, camera_index: int, config: OpenCVCameraConfig | None = None, **kwargs):
        if config is None:
            config = OpenCVCameraConfig()

        # Overwrite config arguments using kwargs
        config = replace(config, **kwargs)

        self.camera_index = camera_index
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.color_mode = config.color_mode

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"OpenCVCamera({self.camera_index}) is already connected.")

        # First create a temporary camera trying to access `camera_index`,
        # and verify it is a valid camera by calling `isOpened`.

        if platform.system() == "Linux":
            # Linux uses ports for connecting to cameras
            tmp_camera = cv2.VideoCapture(f"/dev/video{self.camera_index}")
        else:
            tmp_camera = cv2.VideoCapture(self.camera_index)

        is_camera_open = tmp_camera.isOpened()
        # Release camera to make it accessible for `find_camera_indices`
        del tmp_camera

        # If the camera doesn't work, display the camera indices corresponding to
        # valid cameras.
        if not is_camera_open:
            # Verify that the provided `camera_index` is valid before printing the traceback
            available_cam_ids = find_camera_indices()
            if self.camera_index not in available_cam_ids:
                raise ValueError(
                    f"`camera_index` is expected to be one of these available cameras {available_cam_ids}, but {self.camera_index} is provided instead. "
                    "To find the camera index you should use, run `python lerobot/scripts/save_images_from_cameras.py --driver opencv`."
                )

            raise OSError(f"Can't access OpenCVCamera({self.camera_index}).")

        # Secondly, create the camera that will be used downstream.
        # Note: For some unknown reason, calling `isOpened` blocks the camera which then
        # needs to be re-created.
        if platform.system() == "Linux":
            self.camera = cv2.VideoCapture(f"/dev/video{self.camera_index}")
        else:
            self.camera = cv2.VideoCapture(self.camera_index)

        if self.fps is not None:
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        if self.width is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if self.fps is not None and not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            raise OSError(
                f"Can't set {self.fps=} for OpenCVCamera({self.camera_index}). Actual value is {actual_fps}."
            )
        if self.width is not None and self.width != actual_width:
            raise OSError(
                f"Can't set {self.width=} for OpenCVCamera({self.camera_index}). Actual value is {actual_width}."
            )
        if self.height is not None and self.height != actual_height:
            raise OSError(
                f"Can't set {self.height=} for OpenCVCamera({self.camera_index}). Actual value is {actual_height}."
            )

        self.fps = actual_fps
        self.width = actual_width
        self.height = actual_height

        self.is_connected = True

    def read(self, temporary_color_mode: str | None = None) -> np.ndarray:
        """Read a frame from the camera returned in the format (height, width, channels)
        (e.g. 480 x 640 x 3), contrarily to the pytorch format which is channel first.

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is non blocking version of `camera.read()`.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        start_time = time.perf_counter()

        ret, color_image = self.camera.read()
        if not ret:
            raise OSError(f"Can't capture color image from camera {self.camera_index}.")

        requested_color_mode = self.color_mode if temporary_color_mode is None else temporary_color_mode

        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )

        # OpenCV uses BGR format as default (blue, green, red) for all operations, including displaying images.
        # However, Deep Learning framework such as LeRobot uses RGB format as default to train neural networks,
        # so we convert the image color from BGR to RGB.
        if requested_color_mode == "rgb":
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        h, w, _ = color_image.shape
        if h != self.height or w != self.width:
            raise OSError(
                f"Can't capture color image with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
            )

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        return color_image

    def read_loop(self):
        while self.stop_event is None or not self.stop_event.is_set():
            self.color_image = self.read()

    def async_read(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        num_tries = 0
        while self.color_image is None:
            num_tries += 1
            time.sleep(1 / self.fps)
            if num_tries > self.fps and (self.thread.ident is None or not self.thread.is_alive()):
                raise Exception(
                    "The thread responsible for `self.async_read()` took too much time to start. There might be an issue. Verify that `self.thread.start()` has been called."
                )

        return self.color_image

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            # wait for the thread to finish
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        self.camera.release()
        self.camera = None

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
