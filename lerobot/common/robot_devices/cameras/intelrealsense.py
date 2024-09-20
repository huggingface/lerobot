"""
This file contains utilities for recording frames from Intel Realsense cameras.
"""

import threading
import time
import traceback
from dataclasses import dataclass, replace
from threading import Thread

import cv2
import numpy as np
import pyrealsense2 as rs

from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)
from lerobot.common.utils.utils import capture_timestamp_utc

SERIAL_NUMBER_INDEX = 1


def find_camera_indices(raise_when_empty=True) -> list[int]:
    """
    Find the serial numbers of the Intel RealSense cameras
    connected to the computer.
    """
    camera_ids = []
    for device in rs.context().query_devices():
        serial_number = int(device.get_info(rs.camera_info(SERIAL_NUMBER_INDEX)))
        camera_ids.append(serial_number)

    if raise_when_empty and len(camera_ids) == 0:
        raise OSError(
            "Not a single camera was detected. Try re-plugging, or re-installing `librealsense` and its python wrapper `pyrealsense2`, or updating the firmware."
        )

    return camera_ids


@dataclass
class IntelRealSenseCameraConfig:
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    IntelRealSenseCameraConfig(30, 640, 480)
    IntelRealSenseCameraConfig(60, 640, 480)
    IntelRealSenseCameraConfig(90, 640, 480)
    IntelRealSenseCameraConfig(30, 1280, 720)
    IntelRealSenseCameraConfig(30, 640, 480, use_depth=True)
    ```
    """

    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    use_depth: bool = False
    force_hardware_reset: bool = True

    def __post_init__(self):
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        if (self.fps or self.width or self.height) and not (self.fps and self.width and self.height):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them, "
                f"but {self.fps=}, {self.width=}, {self.height=} were provided."
            )


class IntelRealSenseCamera:
    """
    The IntelRealSenseCamera class is similar to OpenCVCamera class but adds additional features for Intel Real Sense cameras:
    - camera_index corresponds to the serial number of the camera,
    - camera_index won't randomly change as it can be the case of OpenCVCamera for Linux,
    - read is more reliable than OpenCVCamera,
    - depth map can be returned.

    To find the camera indices of your cameras, you can run our utility script that will save a few frames for each camera:
    ```bash
    python lerobot/scripts/save_images_from_cameras.py \
    --driver intelrealsense \
    --images-dir outputs/images_from_intelrealsense_cameras
    ```

    When an IntelRealSenseCamera is instantiated, if no specific config is provided, the default fps, width, height and color_mode
    of the given camera will be used.

    Example of usage:
    ```python
    camera_index = 128422271347
    camera = IntelRealSenseCamera(camera_index)
    camera.connect()
    color_image = camera.read()
    # when done using the camera, consider disconnecting
    camera.disconnect()
    ```

    Example of changing default fps, width, height and color_mode:
    ```python
    camera = IntelRealSenseCamera(camera_index, fps=30, width=1280, height=720)
    camera = connect()  # applies the settings, might error out if these settings are not compatible with the camera

    camera = IntelRealSenseCamera(camera_index, fps=90, width=640, height=480)
    camera = connect()

    camera = IntelRealSenseCamera(camera_index, fps=90, width=640, height=480, color_mode="bgr")
    camera = connect()
    ```

    Example of returning depth:
    ```python
    camera = IntelRealSenseCamera(camera_index, use_depth=True)
    camera.connect()
    color_image, depth_map = camera.read()
    ```
    """

    def __init__(
        self,
        camera_index: int,
        config: IntelRealSenseCameraConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = IntelRealSenseCameraConfig()

        # Overwrite the config arguments using kwargs
        config = replace(config, **kwargs)

        self.camera_index = camera_index
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.force_hardware_reset = config.force_hardware_reset

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.depth_map = None
        self.logs = {}

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"IntelRealSenseCamera({self.camera_index}) is already connected."
            )

        config = rs.config()
        config.enable_device(str(self.camera_index))

        if self.fps and self.width and self.height:
            # TODO(rcadene): can we set rgb8 directly?
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        else:
            config.enable_stream(rs.stream.color)

        if self.use_depth:
            if self.fps and self.width and self.height:
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            else:
                config.enable_stream(rs.stream.depth)

        self.camera = rs.pipeline()
        try:
            self.camera.start(config)
            is_camera_open = True
        except RuntimeError:
            is_camera_open = False
            traceback.print_exc()

        # If the camera doesn't work, display the camera indices corresponding to
        # valid cameras.
        if not is_camera_open:
            # Verify that the provided `camera_index` is valid before printing the traceback
            available_cam_ids = find_camera_indices()
            if self.camera_index not in available_cam_ids:
                raise ValueError(
                    f"`camera_index` is expected to be one of these available cameras {available_cam_ids}, but {self.camera_index} is provided instead. "
                    "To find the camera index you should use, run `python lerobot/scripts/save_images_from_cameras.py --driver intelrealsense`."
                )

            raise OSError(f"Can't access IntelRealSenseCamera({self.camera_index}).")

        self.is_connected = True

    def read(self, temporary_color: str | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera returned in the format height x width x channels (e.g. 480 x 640 x 3)
        of type `np.uint8`, contrarily to the pytorch format which is float channel first.

        When `use_depth=True`, returns a tuple `(color_image, depth_map)` with a depth map in the format
        height x width (e.g. 480 x 640) of type np.uint16.

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is non blocking version of `camera.read()`.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealSenseCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        start_time = time.perf_counter()

        frame = self.camera.wait_for_frames(timeout_ms=5000)

        color_frame = frame.get_color_frame()

        if not color_frame:
            raise OSError(f"Can't capture color image from IntelRealSenseCamera({self.camera_index}).")

        color_image = np.asanyarray(color_frame.get_data())

        requested_color_mode = self.color_mode if temporary_color is None else temporary_color
        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )

        # IntelRealSense uses RGB format as default (red, green, blue).
        if requested_color_mode == "bgr":
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        h, w, _ = color_image.shape
        if h != self.height or w != self.width:
            raise OSError(
                f"Can't capture color image with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
            )

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        if self.use_depth:
            depth_frame = frame.get_depth_frame()
            if not depth_frame:
                raise OSError(f"Can't capture depth image from IntelRealSenseCamera({self.camera_index}).")

            depth_map = np.asanyarray(depth_frame.get_data())

            h, w = depth_map.shape
            if h != self.height or w != self.width:
                raise OSError(
                    f"Can't capture depth map with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
                )

            return color_image, depth_map
        else:
            return color_image

    def read_loop(self):
        while self.stop_event is None or not self.stop_event.is_set():
            if self.use_depth:
                self.color_image, self.depth_map = self.read()
            else:
                self.color_image = self.read()

    def async_read(self):
        """Access the latest color image"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealSenseCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
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

        if self.use_depth:
            return self.color_image, self.depth_map
        else:
            return self.color_image

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealSenseCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            # wait for the thread to finish
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        self.camera.stop()
        self.camera = None

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
