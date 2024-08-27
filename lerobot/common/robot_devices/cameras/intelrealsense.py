"""
This file contain ultilities for recording frames from Intel Realsense Cameras.
"""

from dataclasses import dataclass, replace
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)
from lerobot.common.utils.utils import capture_timestamp_utc
from lerobot.scripts.control_robot import busy_wait
from pathlib import Path
from PIL import Image
from threading import Thread
import argparse
import concurrent.futures
import cv2
import logging
import numpy as np
import pyrealsense2 as rs
import shutil
import threading
import time
import traceback


SERIAL_NUMBER_INDEX = 1


def find_camera_indices(raise_when_empty=True):
    """
    Identify and return the serial numbers of the Intel RealSense cameras
    connected to the laptop.

    Args:
        raise_when_empty (bool, optional): Whether to raise an OSError if no
        cameras are detected. Defaults to True.

    Raises:
        OSError: If no cameras are detected and `raise_when_empty` is True.

    Returns:
        list[int]: A list of serial numbers for the detected Intel RealSense
        cameras.
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


def save_image(img_array, camera_idx, frame_index, images_dir):
    """
    Save the image array as a `png` file in a given location.

    Args:
        img_array (np.ndarray): Image to be saved
        camera_idx (int): Serial number of the camera
        frame_index (int): Index associated with the frame
        images_dir (str): Location to save the image
    """
    try:
        img = Image.fromarray(img_array)
        path = images_dir / f"camera_{camera_idx}_frame_{frame_index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), quality=100)
        logging.info(f"Saved image: {path}")
    except Exception as e:
        logging.error(
            f"Failed to save image for camera {camera_idx} frame {frame_index}: {e}"
        )


def save_images_from_cameras(
    images_dir: Path,
    camera_ids: list[int] | None = None,
    fps=None,
    width=None,
    height=None,
    record_time_s=2,
):
    """
    Initializes all the cameras and saves images to the directory using
    asynchronous read.

    Args:
        images_dir (Path): Path to the local directory
        camera_ids (list[int] | None, optional): Serial numbers of the cameras. Defaults to None.
        fps (_type_, optional): FPS for image captures. Defaults to None.
        width (_type_, optional): Height of Image. Defaults to None.
        height (_type_, optional): Width of Image. Defaults to None.
        record_time_s (int, optional): Time for which the images will be captured. Defaults to 2.
    """
    if camera_ids is None:
        camera_ids = find_camera_indices()

    print("Connecting cameras")
    cameras = []
    for cam_idx in camera_ids:
        camera = IntelRealSenseCamera(cam_idx, fps=fps, width=width, height=height)
        camera.connect()
        print(
            f"IntelRealSense Camera({camera.camera_idx}, fps={camera.fps}, width={camera.width}, height={camera.height}, color_mode={camera.color})"
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
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                now = time.perf_counter()

                for camera in cameras:
                    # If we use async_read when fps is None, the loop will go full speed, and we will endup
                    # saving the same images from the cameras multiple times until the RAM/disk is full.
                    image = camera.read() if fps is None else camera.async_read()
                    if image is None:
                        print("No Frame")
                    bgr_converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    executor.submit(
                        save_image,
                        bgr_converted_image,
                        camera.camera_idx,
                        frame_index,
                        images_dir,
                    )

                if fps is not None:
                    dt_s = time.perf_counter() - now
                    busy_wait(1 / fps - dt_s)

                if time.perf_counter() - start_time > record_time_s:
                    break

                print(
                    f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}"
                )

                frame_index += 1
    finally:
        print(f"Images have been saved to {images_dir}")
        for camera in cameras:
            camera.disconnect()


@dataclass
class IntelRealSenseCameraConfig:
    """
    Configuration settings for an Intel RealSense camera.

    Attributes:
        fps (int | None): The frame rate (frames per second) for the camera. 
                          If set to None, the default fps will be used.
        width (int | None): The width of the camera's video stream. 
                            If set to None, the default width will be used.
        height (int | None): The height of the camera's video stream. 
                             If set to None, the default height will be used.
        color (str): The color format of the camera's video stream. 
                     Can be 'rgb' or 'bgr'. Defaults to 'rgb'.
        use_depth (bool): Whether to use the camera's depth sensor. 
                          Defaults to False.
        force_hardware_reset (bool): Whether to force a hardware reset of the camera. 
                                     Defaults to True.
    
    Raises:
        ValueError: If an invalid color format is provided.
        ValueError: If only some of fps, width, or height are set.
    """
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color: str = "rgb"
    use_depth: bool = False
    force_hardware_reset: bool = True

    def __post_init__(self):
        """
        Validate the configuration settings after initialization.

        Ensures that the color format is either 'rgb' or 'bgr', and that 
        if any of fps, width, or height are set, then all of them must be set.
        """
        if self.color not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {self.color} is provided."
            )

        if (self.fps or self.width or self.height) and not (
            self.fps and self.width and self.height
        ):
            raise ValueError(
                f"Expected all fps, width and height to be set, when one of them is set, but {self.fps=}, {self.width=}, {self.height=}."
            )


class IntelRealSenseCamera:
    """
    A class to manage and interface with an Intel RealSense camera.

    This class provides methods to connect to the camera, capture frames (both color and depth), 
    and manage the camera in an asynchronous manner.

    Attributes:
        AVAILABLE_CAMERA_INDICES (list[int]): A list of serial numbers for all available Intel RealSense cameras.
        camera_idx (str): The serial number of the camera being used.
        fps (int | None): The frame rate (frames per second) for the camera stream.
        width (int | None): The width of the camera's video stream.
        height (int | None): The height of the camera's video stream.
        color (str): The color format of the camera's video stream. Can be 'rgb' or 'bgr'.
        use_depth (bool): Whether the camera's depth sensor is enabled.
        force_hardware_reset (bool): Whether to force a hardware reset of the camera on initialization.
        thread (threading.Thread | None): A thread for managing the camera stream asynchronously.
        stop_event (threading.Event | None): An event to signal stopping of the camera stream.
        color_image (numpy.ndarray | None): The most recently captured color image from the camera.
        camera (pyrealsense2.pipeline | None): The RealSense camera pipeline object.
        is_connected (bool): Whether the camera is successfully connected.
        _color_image (numpy.ndarray | None): A private attribute holding the most recently captured color image.
        logs (dict): A dictionary to store logs related to the camera operations.
    """                                                                                                                                                                                                                                                                                    
    AVAILABLE_CAMERA_INDICES = find_camera_indices()

    def __init__(
        self,
        camera_index: int,
        config: IntelRealSenseCameraConfig | None = None,
        **kwargs,
    ):
        """
        Initialize the Intel RealSense Camera with the given configuration.

        Args:
            camera_index (int): The index or serial number of the camera to initialize.
            config (IntelRealSenseCameraConfig, optional): A configuration object containing settings for the camera.
                                                        Defaults to a new instance of IntelRealSenseCameraConfig.
            **kwargs: Additional configuration options to override those in the provided config.

        Raises:
            ValueError: If the provided camera_index is not valid or available.
        """

        if config is None:
            config = IntelRealSenseCameraConfig()

        # Overwrite the config arguments using kwargs
        config = replace(config, **kwargs)

        self.camera_idx = str(camera_index)
        self.fps = config.fps
        self.width = config.width
        self.height = config.height
        self.color = config.color
        self.use_depth = config.use_depth
        self.force_hardware_reset = config.force_hardware_reset

        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.camera = None
        self.is_connected = False
        self._color_image = None

        self.logs = {}

    def connect(self):
        """
        Connects and setups configuration for the camera.

        Raises:
            ValueError: If camera is already connected.
            ValueError: If camera is not detected by the device.
        """
        if self.is_connected:
            raise ValueError(f"Camera {self.camera_idx} is already connected.")

        config = rs.config()
        config.enable_device(str(self.camera_idx))

        if self.fps and self.width and self.height:
            # TODO(rcadene): can we set rgb8 directly?
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
            )
        else:
            config.enable_stream(rs.stream.color)

        if self.use_depth:
            if self.fps and self.width and self.height:
                config.enable_stream(
                    rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
                )
            else:
                config.enable_stream(rs.stream.depth)

        self.camera = rs.pipeline()
        try:
            self.camera.start(config)
        except RuntimeError:
            # Verify that the provided `camera_idx` is valid before printing the traceback
            if self.camera_idx not in IntelRealSenseCamera.AVAILABLE_CAMERA_INDICES:
                raise ValueError(
                    f"`camera_idx` is expected to be a serial number of one of these available cameras {IntelRealSenseCamera.AVAILABLE_CAMERA_INDICES}, but {self.camera_idx} is provided instead."
                )
            traceback.print_exc()

        self.is_connected = True

    def read(
        self, temporary_color: str | None = None
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Read a colored frame from the camera.

        Args:
            temporary_color (str | None, optional): Determine the color format( eg. RGB, BGR). Defaults to None.

        Raises:
            RobotDeviceNotConnectedError: If camera is not connected.
            OSError: If frame is not captured.
            ValueError: If the color format is not correct.
            OSError: If there is a height and width mismatch.
            OSError: If cannot read depth image.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]: Colored image.
        """

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealSense({self.camera_idx}) is not connected. Try running `camera.connect()` first."
            )

        start_time = time.perf_counter()

        frame = self.camera.wait_for_frames(timeout_ms=5000)

        color_frame = frame.get_color_frame()

        if not color_frame:
            raise OSError(f"Can't capture color image from camera {self.camera_idx}.")

        color_image = np.asanyarray(color_frame.get_data())

        if temporary_color is None:
            requested_color = self.color
        else:
            requested_color = temporary_color

        if requested_color not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color} is provided."
            )

        # OpenCV uses BGR format as default (blue, green red) for all operations, including displaying images.
        # However, Deep Learning framework such as LeRobot uses RGB format as default to train neural networks,
        # so we convert the image color from BGR to RGB.
        if requested_color == "rgb":
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

        if self.use_depth:
            depth_frame = frame.get_depth_frame()
            if not depth_frame:
                raise OSError(
                    f"Can't capture depth image from camera {self.camera_idx}."
                )
            depth_image = np.asanyarray(depth_frame.get_data())

            return color_image, depth_image
        else:
            return color_image

    def read_loop(self):
        """
        Loop to read the image invoked using thread.
        """
        while self.stop_event is None or not self.stop_event.is_set():
            self.color_image = self.read()

    def async_read(self):
        """
        Asynchronously read the images from the camera.

        Raises:
            RobotDeviceNotConnectedError: If camera is not connected.
            Exception: If takes more time to read the frame than the fps
                       specified.

        Returns:
            np.ndarray: Colored image that is captured
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IntelRealsense( {self.camera_idx}) is not connected. Try running `camera.connect()` first."
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
            if num_tries > self.fps and (
                self.thread.ident is None or not self.thread.is_alive()
            ):
                raise Exception(
                    "The thread responsible for `self.async_read()` took too much time to start. There might be an issue. Verify that `self.thread.start()` has been called."
                )
        return self.color_image

    def disconnect(self):
        """Disconnects the camera for graceful termination of program.

        Raises:
            RobotDeviceNotConnectedError: If disconnect invoked without connecting the camera.
        """

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"Intel ({self.camera_idx}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            # wait for the thread to finish
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        if getattr(self, "camera", None):
            try:
                self.camera.stop()
            except RuntimeError as e:
                if "stop() cannot be called before start()" in str(e):
                    # skip this runtime error
                    return
                traceback.print_exc()

        self.is_connected = False

    def __del__(self):
        if hasattr(self, "is_connected") and self.is_connected:
            self.disconnect()


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
    # parser.add_argument("--use-depth", type=int, default=0)
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=str,
        default=640,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=str,
        default=480,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/test_trossen",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=2.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )

    args = parser.parse_args()
    save_images_from_cameras(**vars(args))

    print("Program Ended")
