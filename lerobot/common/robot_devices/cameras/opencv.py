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

"""
This file contains utilities for recording frames from cameras. For more info look at `OpenCVCamera` docstring.
"""

import argparse
import concurrent.futures
import math
import os
import platform
import shutil
import threading
import time
from contextlib import suppress
from pathlib import Path
from threading import Thread

import numpy as np
from PIL import Image

from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
from lerobot.common.utils.utils import capture_timestamp_utc

# The maximum opencv device index depends on your operating system. For instance,
# if you have 3 cameras, they should be associated to index 0, 1, and 2. This is the case
# on MacOS. However, on Ubuntu, the indices are different like 6, 16, 23.
# When you change the USB port or reboot the computer, the operating system might
# treat the same cameras as new devices. Thus we select a higher bound to search indices.
MAX_OPENCV_INDEX = 60
_PROBE_TIMEOUT_SECONDS = 3.0  # Timeout for probing individual cameras


class _ProbeResult:
    """Helper class to store results from the camera probing thread."""

    def __init__(self):
        self.opened = False
        self.exception = None


def _attempt_open_camera_in_thread(camera_id_to_probe, result_obj, cv2_module_to_use):
    """
    Attempts to open and verify a camera in a separate thread.
    Modifies result_obj in place.
    Ensures the camera is released if opened.
    """
    temp_cam = None
    try:
        opened_successfully = False
        if platform.system() == "Linux" and isinstance(camera_id_to_probe, str):
            # First, try with explicit V4L2 backend on Linux for /dev/video* paths
            # print(f"      (Probe) Attempting {camera_id_to_probe} with V4L2 backend...") # Optional debug
            temp_cam = cv2_module_to_use.VideoCapture(camera_id_to_probe, cv2_module_to_use.CAP_V4L2)
            if temp_cam is not None and temp_cam.isOpened():
                opened_successfully = True
            else:
                # print(f"      (Probe) V4L2 attempt failed for {camera_id_to_probe}. Trying default backend...") # Optional debug
                if temp_cam is not None:  # Release the failed attempt before trying again
                    with suppress(Exception):
                        temp_cam.release()
                # Fallback to default backend if V4L2 failed
                temp_cam = cv2_module_to_use.VideoCapture(camera_id_to_probe)
                if temp_cam is not None and temp_cam.isOpened():
                    opened_successfully = True
        else:
            # For non-Linux or non-string IDs (e.g., integer indices on Windows/macOS)
            temp_cam = cv2_module_to_use.VideoCapture(camera_id_to_probe)
            if temp_cam is not None and temp_cam.isOpened():
                opened_successfully = True

        result_obj.opened = opened_successfully

    except Exception as e:
        result_obj.exception = e
    finally:
        if temp_cam is not None:
            with suppress(Exception):
                temp_cam.release()


def check_video_device_permissions(port: str) -> bool:
    """Check if the current user has read/write permissions for a video device.

    Args:
        port: Path to the video device (e.g. '/dev/video0')

    Returns:
        bool: True if user has proper permissions, False otherwise
    """
    try:
        # Check if file exists and is a character device
        if not Path(port).is_char_device():
            return False

        # Check if user has read/write permissions
        return os.access(port, os.R_OK | os.W_OK)
    except Exception:
        return False


def find_cameras(raise_when_empty=False, max_index_search_range=MAX_OPENCV_INDEX, mock=False) -> list[dict]:
    cameras = []
    if platform.system() == "Linux":
        print("Linux detected. Finding available camera indices through scanning '/dev/video*' ports")
        possible_ports = [str(port) for port in Path("/dev").glob("video*")]

        # First check permissions for all video devices
        inaccessible_ports = [port for port in possible_ports if not check_video_device_permissions(port)]
        if inaccessible_ports:
            error_msg = (
                "No permission to access video devices. The following devices are inaccessible:\n"
                + "\n".join(f"- {port}" for port in inaccessible_ports)
                + "\n\nTo fix this, either:\n"
                "1. Add your user to the 'video' group: sudo usermod -a -G video $USER\n"
                "2. Then log out and log back in for the changes to take effect\n"
                "3. Or run the script with sudo (not recommended)"
            )
            raise PermissionError(error_msg)

        ports = _find_cameras(possible_ports, mock=mock)
        for port in ports:
            cameras.append(
                {
                    "port": port,
                    "index": int(port.removeprefix("/dev/video")),
                }
            )
    else:
        print(
            "Mac or Windows detected. Finding available camera indices through "
            f"scanning all indices from 0 to {MAX_OPENCV_INDEX}"
        )
        possible_indices = range(max_index_search_range)
        indices = _find_cameras(possible_indices, mock=mock)
        for index in indices:
            cameras.append(
                {
                    "port": None,
                    "index": index,
                }
            )

    if raise_when_empty and not cameras:
        raise OSError(
            "Not a single camera was detected after probing. Try re-plugging, or re-installing `opencv2`, "
            "or your camera driver, or make sure your camera is compatible with opencv2."
        )
    return cameras


def _find_cameras(
    possible_camera_ids: list[int | str], raise_when_empty=False, mock=False
) -> list[int | str]:
    if mock:
        import tests.cameras.mock_cv2 as active_cv2_module
    else:
        import cv2 as active_cv2_module

    found_camera_ids = []
    if not possible_camera_ids:
        print("No possible camera IDs/ports found to check (e.g., no /dev/video* entries).")

    for camera_idx in possible_camera_ids:
        print(f"Probing camera: {camera_idx}...")

        probe_result_obj = _ProbeResult()

        probe_thread = threading.Thread(
            target=_attempt_open_camera_in_thread, args=(camera_idx, probe_result_obj, active_cv2_module)
        )
        probe_thread.daemon = True
        probe_thread.start()
        probe_thread.join(timeout=_PROBE_TIMEOUT_SECONDS)

        if probe_thread.is_alive():
            print(f"  Probe for camera {camera_idx} timed out after {_PROBE_TIMEOUT_SECONDS}s. Skipping.")
        else:
            if probe_result_obj.exception:
                print(f"  Exception while probing camera {camera_idx}: {probe_result_obj.exception}")
            elif probe_result_obj.opened:
                print(f"  Successfully opened and verified camera: {camera_idx}")
                found_camera_ids.append(camera_idx)
            else:
                print(f"  Failed to open or verify camera: {camera_idx}")

    if raise_when_empty and len(found_camera_ids) == 0:
        raise OSError(
            "Not a single camera was detected. Try re-plugging, or re-installing `opencv2`, "
            "or your camera driver, or make sure your camera is compatible with opencv2."
        )

    if not found_camera_ids:
        print("No working cameras found from the probed list.")
    else:
        print(f"Found working cameras at: {found_camera_ids}")

    return found_camera_ids


def is_valid_unix_path(path: str) -> bool:
    """Note: if 'path' points to a symlink, this will return True only if the target exists"""
    p = Path(path)
    return p.is_absolute() and p.exists()


def get_camera_index_from_unix_port(port: Path) -> int:
    return int(str(port.resolve()).removeprefix("/dev/video"))


def save_image(img_array, camera_index, frame_index, images_dir):
    img = Image.fromarray(img_array)
    path = images_dir / f"camera_{camera_index:02d}_frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def save_images_from_cameras(
    images_dir: Path,
    camera_ids: list | None = None,
    fps=None,
    width=None,
    height=None,
    record_time_s=2,
    mock=False,
):
    """
    Initializes all the cameras and saves images to the directory. Useful to visually identify the camera
    associated to a given camera index.
    """
    if camera_ids is None or len(camera_ids) == 0:
        camera_infos = find_cameras(mock=mock)
        if not camera_infos:
            print("No cameras found. Exiting save_images_from_cameras.")
            return
        camera_ids = [cam["index"] for cam in camera_infos]

    print("Connecting cameras")
    cameras = []
    for cam_idx in camera_ids:
        config = OpenCVCameraConfig(camera_index=cam_idx, fps=fps, width=width, height=height, mock=mock)
        camera = OpenCVCamera(config)
        camera.connect()
        print(
            f"OpenCVCamera({camera.camera_index}, fps={camera.fps}, width={camera.capture_width}, "
            f"height={camera.capture_height}, color_mode={camera.color_mode})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            now = time.perf_counter()

            for camera in cameras:
                # If we use async_read when fps is None, the loop will go full speed, and we will endup
                # saving the same images from the cameras multiple times until the RAM/disk is full.
                image = camera.read() if fps is None else camera.async_read()

                executor.submit(
                    save_image,
                    image,
                    camera.camera_index,
                    frame_index,
                    images_dir,
                )

            if fps is not None:
                dt_s = time.perf_counter() - now
                busy_wait(1 / fps - dt_s)

            print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

            if time.perf_counter() - start_time > record_time_s:
                break

            frame_index += 1

    # Disconnect cameras after use
    for camera in cameras:
        camera.disconnect()

    print(f"Images have been saved to {images_dir}")


class OpenCVCamera:
    """
    The OpenCVCamera class allows to efficiently record images from cameras. It relies on opencv2 to communicate
    with the cameras. Most cameras are compatible. For more info, see the [Video I/O with OpenCV Overview](https://docs.opencv.org/4.x/d0/da7/videoio_overview.html).

    An OpenCVCamera instance requires a camera index (e.g. `OpenCVCamera(camera_index=0)`). When you only have one camera
    like a webcam of a laptop, the camera index is expected to be 0, but it might also be very different, and the camera index
    might change if you reboot your computer or re-plug your camera. This behavior depends on your operation system.

    To find the camera indices of your cameras, you can run our utility script that will be save a few frames for each camera:
    ```bash
    python lerobot/common/robot_devices/cameras/opencv.py --images-dir outputs/images_from_opencv_cameras
    ```

    When an OpenCVCamera is instantiated, if no specific config is provided, the default fps, width, height and color_mode
    of the given camera will be used.

    Example of usage:
    ```python
    from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig

    config = OpenCVCameraConfig(camera_index=0)
    camera = OpenCVCamera(config)
    camera.connect()
    color_image = camera.read()
    # when done using the camera, consider disconnecting
    camera.disconnect()
    ```

    Example of changing default fps, width, height and color_mode:
    ```python
    config = OpenCVCameraConfig(camera_index=0, fps=30, width=1280, height=720)
    config = OpenCVCameraConfig(camera_index=0, fps=90, width=640, height=480)
    config = OpenCVCameraConfig(camera_index=0, fps=90, width=640, height=480, color_mode="bgr")
    # Note: might error out open `camera.connect()` if these settings are not compatible with the camera
    ```
    """

    def __init__(self, config: OpenCVCameraConfig):
        self.config = config
        self.camera_index = config.camera_index
        self.port = None

        # Linux uses ports for connecting to cameras
        if platform.system() == "Linux":
            if isinstance(self.camera_index, int):
                self.port = Path(f"/dev/video{self.camera_index}")
            elif isinstance(self.camera_index, str) and is_valid_unix_path(self.camera_index):
                self.port = Path(self.camera_index)
                # Retrieve the camera index from a potentially symlinked path
                self.camera_index = get_camera_index_from_unix_port(self.port)
            else:
                raise ValueError(f"Please check the provided camera_index: {self.camera_index}")

        # Store the raw (capture) resolution from the config.
        self.capture_width = config.width
        self.capture_height = config.height

        # If rotated by ±90, swap width and height.
        if config.rotation in [-90, 90]:
            self.width = config.height
            self.height = config.width
        else:
            self.width = config.width
            self.height = config.height

        self.fps = config.fps
        self.channels = config.channels
        self.color_mode = config.color_mode
        self.mock = config.mock

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}

        # cv2 module will be imported dynamically based on self.mock
        self.rotation = None
        # Actual cv2.ROTATE_* constants will be set in connect() after cv2 is imported.

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"OpenCVCamera({self.camera_index}) is already connected.")

        if self.mock:
            import tests.cameras.mock_cv2 as cv2
        else:
            import cv2

            # Use 1 thread to avoid blocking the main thread. Especially useful during data collection
            # when other threads are used to save the images.
            cv2.setNumThreads(1)

        # Set rotation constants after cv2 is imported
        if self.config.rotation == -90:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif self.config.rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif self.config.rotation == 180:
            self.rotation = cv2.ROTATE_180

        backend = (
            cv2.CAP_V4L2
            if platform.system() == "Linux"
            else cv2.CAP_DSHOW
            if platform.system() == "Windows"
            else cv2.CAP_AVFOUNDATION
            if platform.system() == "Darwin"
            else cv2.CAP_ANY
        )

        camera_idx = f"/dev/video{self.camera_index}" if platform.system() == "Linux" else self.camera_index
        # First create a temporary camera trying to access `camera_index`,
        # and verify it is a valid camera by calling `isOpened`.
        tmp_camera = cv2.VideoCapture(camera_idx, backend)
        is_camera_open = tmp_camera.isOpened()
        # Release camera to make it accessible for `find_camera_indices`
        tmp_camera.release()
        del tmp_camera

        # If the camera doesn't work, display the camera indices corresponding to
        # valid cameras.
        if not is_camera_open:
            # Verify that the provided `camera_index` is valid before printing the traceback
            cameras_info = find_cameras(mock=self.mock)  # Pass mock status
            available_cam_ids = [cam["index"] for cam in cameras_info]
            if self.camera_index not in available_cam_ids:
                raise ValueError(
                    f"`camera_index` is expected to be one of these available cameras {available_cam_ids}, but {self.camera_index} is provided instead. "
                    "To find the camera index you should use, run `python lerobot/common/robot_devices/cameras/opencv.py`."
                )

            raise OSError(f"Can't access OpenCVCamera({camera_idx}).")

        # Secondly, create the camera that will be used downstream.
        # Note: For some unknown reason, calling `isOpened` blocks the camera which then
        # needs to be re-created.
        self.camera = cv2.VideoCapture(camera_idx, backend)
        if not self.camera.isOpened():  # Add a final check
            raise OSError(
                f"Failed to open OpenCVCamera({camera_idx}) for use, even after initial checks passed."
            )

        if self.fps is not None:
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        if self.capture_width is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        if self.capture_height is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)

        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Using `math.isclose` since actual fps can be a float (e.g. 29.9 instead of 30)
        if self.fps is not None and not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            # Using `OSError` since it's a broad that encompasses issues related to device communication
            raise OSError(
                f"Can't set fps={self.fps} for OpenCVCamera({self.camera_index}). Actual value is {actual_fps}."
            )
        if self.capture_width is not None and not math.isclose(
            self.capture_width, actual_width, rel_tol=1e-3
        ):
            raise OSError(
                f"Can't set capture_width={self.capture_width} for OpenCVCamera({self.camera_index}). Actual value is {actual_width}."
            )
        if self.capture_height is not None and not math.isclose(
            self.capture_height, actual_height, rel_tol=1e-3
        ):
            raise OSError(
                f"Can't set capture_height={self.capture_height} for OpenCVCamera({self.camera_index}). Actual value is {actual_height}."
            )

        self.fps = round(actual_fps)
        self.capture_width = round(actual_width)
        self.capture_height = round(actual_height)

        # Re-evaluate self.width and self.height if capture dimensions were adjusted
        if self.config.rotation in [-90, 90]:
            self.width = self.capture_height
            self.height = self.capture_width
        else:
            self.width = self.capture_width
            self.height = self.capture_height

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
            if self.mock:
                import tests.cameras.mock_cv2 as cv2
            else:
                import cv2

            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        h, w, _ = color_image.shape
        # Check against capture_height/width before rotation
        if h != self.capture_height or w != self.capture_width:
            raise OSError(
                f"Can't capture color image with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
            )

        if self.rotation is not None:
            # cv2 module needs to be the one loaded in connect() or re-imported if not stored
            if self.mock:
                import tests.cameras.mock_cv2 as cv2_rot
            else:
                import cv2 as cv2_rot
            color_image = cv2_rot.rotate(color_image, self.rotation)

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        self.color_image = color_image

        return color_image

    def read_loop(self):
        while not self.stop_event.is_set():
            try:
                self.color_image = self.read()
            except Exception as e:
                print(f"Error reading in thread: {e}")
                # Add a small sleep to avoid tight loop on continuous error
                time.sleep(0.1)

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
        while self.color_image is None:  # Wait for self.color_image to be populated by read_loop
            time.sleep(1 / self.fps if self.fps > 0 else 0.01)  # Avoid division by zero if fps is 0 or None
            num_tries += 1
            if self.fps > 0 and num_tries > self.fps * 2:  # Max 2 seconds worth of frames
                raise TimeoutError(
                    f"OpenCVCamera({self.camera_index}): Timed out waiting for async_read() to provide the first frame."
                )
            elif (
                self.fps <= 0 and num_tries > 200
            ):  # Fallback for fps <= 0, approx 2 seconds if sleep is 0.01
                raise TimeoutError(
                    f"OpenCVCamera({self.camera_index}): Timed out waiting for async_read() (fps={self.fps})."
                )

        return self.color_image

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()  # wait for the thread to finish
            self.thread = None
            self.stop_event = None

        if self.camera is not None:  # Check if camera object exists
            self.camera.release()
            self.camera = None
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            try:
                self.disconnect()
            except (
                RobotDeviceNotConnectedError
            ):  # Can happen if disconnect called twice or state inconsistent
                pass
            except Exception as e:
                # Suppress other exceptions during __del__
                print(
                    f"Exception during OpenCVCamera({getattr(self, 'camera_index', 'N/A')}) __del__ disconnect: {e}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `OpenCVCamera` for all cameras connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--camera-ids",
        type=int,
        nargs="*",
        default=None,
        help="List of camera indices used to instantiate the `OpenCVCamera`. If not provided, find and use all available camera indices.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_opencv_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=4.0,
        help="Set the number of seconds used to record the frames. By default, 4 seconds.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))
