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
Provides the RealSenseCamera class for capturing frames from Intel RealSense cameras.
"""

import contextlib
import logging
import math
import queue
import time
from threading import Event, Thread
from typing import Any, Dict, List

import cv2
import numpy as np
import pyrealsense2 as rs

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_realsense import RealSenseCameraConfig

logger = logging.getLogger(__name__)


class RealSenseCamera(Camera):
    """
    Manages interactions with Intel RealSense cameras for frame and depth recording.

    This class provides an interface similar to `OpenCVCamera` but tailored for
    RealSense devices, leveraging the `pyrealsense2` library. It uses the camera's
    unique serial number for identification, offering more stability than device
    indices, especially on Linux. It also supports capturing depth maps alongside
    color frames.

    Use the provided utility script to find available camera indices and default profiles:
    ```bash
    python -m lerobot.find_cameras realsense
    ```

    A `RealSenseCamera` instance requires a configuration object specifying the
    camera's serial number or a unique device name. If using the name, ensure only
    one camera with that name is connected.

    The camera's default settings (FPS, resolution, color mode) from the stream
    profile are used unless overridden in the configuration.

    Example:
        ```python
        from lerobot.common.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
        from lerobot.common.cameras import ColorMode, Cv2Rotation

        # Basic usage with serial number
        config = RealSenseCameraConfig(serial_number_or_name="1234567890") # Replace with actual SN
        camera = RealSenseCamera(config)
        camera.connect()

        # Read 1 frame synchronously
        color_image = camera.read()
        print(color_image.shape)

        # Read 1 frame asynchronously
        async_image = camera.async_read()

        # When done, properly disconnect the camera using
        camera.disconnect()

        # Example with depth capture and custom settings
        custom_config = RealSenseCameraConfig(
            serial_number_or_name="1234567890", # Replace with actual SN
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR, # Request BGR output
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True
        )
        depth_camera = RealSenseCamera(custom_config)
        try:
            depth_camera.connect()
            depth_map = depth_camera.read_depth()
            print(f"Depth shape: {depth_map.shape}")
        finally:
            depth_camera.disconnect()

        # Example using a unique camera name
        name_config = RealSenseCameraConfig(serial_number_or_name="Intel RealSense D435") # If unique
        name_camera = RealSenseCamera(name_config)
        # ... connect, read, disconnect ...
        ```
    """

    def __init__(self, config: RealSenseCameraConfig):
        """
        Initializes the RealSenseCamera instance.

        Args:
            config: The configuration settings for the camera.
        """

        super().__init__(config)

        self.config = config

        if isinstance(config.serial_number_or_name, int):
            self.serial_number = str(config.serial_number_or_name)
        else:
            self.serial_number = self._find_serial_number_from_name(config.serial_number_or_name)

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth

        self.rs_pipeline: rs.pipeline | None = None
        self.rs_profile: rs.pipeline_profile | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width
            else:
                self.capture_width, self.capture_height = self.width, self.height

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera pipeline is started and streams are active."""
        return self.rs_pipeline is not None and self.rs_profile is not None

    def connect(self, warmup: bool = True):
        """
        Connects to the RealSense camera specified in the configuration.

        Initializes the RealSense pipeline, configures the required streams (color
        and optionally depth), starts the pipeline, and validates the actual stream settings.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ValueError: If the configuration is invalid (e.g., missing serial/name, name not unique).
            ConnectionError: If the camera is found but fails to start the pipeline or no RealSense devices are detected at all.
            RuntimeError: If the pipeline starts but fails to apply requested settings.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        self.rs_pipeline = rs.pipeline()
        rs_config = self._make_rs_pipeline_config()

        try:
            self.rs_profile = self.rs_pipeline.start(rs_config)
            logger.debug(f"Successfully started pipeline for camera {self.serial_number}.")
        except RuntimeError as e:
            self.rs_profile = None
            self.rs_pipeline = None
            raise ConnectionError(
                f"Failed to open {self} camera. Run 'python -m lerobot.find_cameras realsense' for details about the available cameras in your system."
            ) from e

        logger.debug(f"Validating stream configuration for {self}...")
        self._validate_capture_settings()

        if warmup:
            logger.debug(f"Reading a warm-up frame for {self}...")
            self.read()  # NOTE(Steven): For now we just read one frame, we could also loop for X frames/secs

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        """
        Detects available Intel RealSense cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'id' (serial number), 'name',
            firmware version, USB type, and other available specs, and the default profile properties (width, height, fps, format).

        Raises:
            OSError: If pyrealsense2 is not installed.
            ImportError: If pyrealsense2 is not installed.
        """
        found_cameras_info = []
        context = rs.context()
        devices = context.query_devices()

        for device in devices:
            camera_info = {
                "name": device.get_info(rs.camera_info.name),
                "type": "RealSense",
                "id": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "usb_type_descriptor": device.get_info(rs.camera_info.usb_type_descriptor),
                "physical_port": device.get_info(rs.camera_info.physical_port),
                "product_id": device.get_info(rs.camera_info.product_id),
                "product_line": device.get_info(rs.camera_info.product_line),
            }

            # Get stream profiles for each sensor
            sensors = device.query_sensors()
            for sensor in sensors:
                profiles = sensor.get_stream_profiles()

                for profile in profiles:
                    if profile.is_video_stream_profile() and profile.is_default():
                        vprofile = profile.as_video_stream_profile()
                        stream_info = {
                            "stream_type": vprofile.stream_name(),
                            "format": vprofile.format().name,
                            "width": vprofile.width(),
                            "height": vprofile.height(),
                            "fps": vprofile.fps(),
                        }
                        camera_info["default_stream_profile"] = stream_info

            found_cameras_info.append(camera_info)
            logger.debug(f"Found RealSense camera: {camera_info}")

        logger.info(f"Detected RealSense cameras: {[cam['id'] for cam in found_cameras_info]}")
        return found_cameras_info

    def _find_serial_number_from_name(self, name: str) -> str:
        """Finds the serial number for a given unique camera name."""
        camera_infos = self.find_cameras()
        found_devices = [cam for cam in camera_infos if str(cam["name"]) == name]

        if not found_devices:
            available_names = [cam["name"] for cam in camera_infos]
            raise ValueError(
                f"No RealSense camera found with name '{name}'. Available camera names: {available_names}"
            )

        if len(found_devices) > 1:
            serial_numbers = [dev["serial_number"] for dev in found_devices]
            raise ValueError(
                f"Multiple RealSense cameras found with name '{name}'. "
                f"Please use a unique serial number instead. Found SNs: {serial_numbers}"
            )

        serial_number = str(found_devices[0]["serial_number"])
        logger.info(f"Found serial number '{serial_number}' for camera name '{name}'.")
        return serial_number

    def _make_rs_pipeline_config(self) -> rs.config:
        """Creates and configures the RealSense pipeline configuration object."""
        rs_config = rs.config()
        rs.config.enable_device(rs_config, self.serial_number)

        if self.width and self.height and self.fps:
            logger.debug(
                f"Requesting Color Stream: {self.capture_width}x{self.capture_height} @ {self.fps} FPS, Format: {rs.format.rgb8}"
            )
            rs_config.enable_stream(
                rs.stream.color, self.capture_width, self.capture_height, rs.format.rgb8, self.fps
            )
            if self.use_depth:
                logger.debug(
                    f"Requesting Depth Stream: {self.capture_width}x{self.capture_height} @ {self.fps} FPS, Format: {rs.format.z16}"
                )
                rs_config.enable_stream(
                    rs.stream.depth, self.capture_width, self.capture_height, rs.format.z16, self.fps
                )
        else:
            logger.debug(f"Requesting Color Stream: Default settings, Format: {rs.stream.color}")
            rs_config.enable_stream(rs.stream.color)
            if self.use_depth:
                logger.debug(f"Requesting Depth Stream: Default settings, Format: {rs.stream.depth}")
                rs_config.enable_stream(rs.stream.depth)

        return rs_config

    def _validate_capture_settings(self) -> None:
        """
        Validates if the actual stream settings match the requested configuration.

        This method compares the requested FPS, width, and height against the
        actual settings obtained from the active RealSense profile after the
        pipeline has started.

        Raises:
            RuntimeError: If the actual camera settings significantly deviate
                          from the requested ones.
            DeviceNotConnectedError: If the camera is not connected when attempting
                                     to validate settings.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Cannot validate settings for {self} as it is not connected.")

        stream = self.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()

        if self.fps is None:
            self.fps = stream.fps()
        else:
            self._validate_fps(stream)

        if self.width is None or self.height is None:
            actual_width = int(round(stream.width()))
            actual_height = int(round(stream.height()))
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = actual_height, actual_width
                self.capture_width, self.capture_height = actual_width, actual_height
            else:
                self.width, self.height = actual_width, actual_height
                self.capture_width, self.capture_height = actual_width, actual_height
            logger.info(f"Capture width set to camera default: {self.width}.")
            logger.info(f"Capture height set to camera default: {self.height}.")
        else:
            self._validate_width_and_height(stream)

        if self.use_depth:
            stream = self.rs_profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self._validate_fps(stream)
            self._validate_width_and_height(stream)

    def _validate_fps(self, stream: rs.video_stream_profile) -> None:
        """Validates and sets the internal FPS based on actual stream FPS."""
        actual_fps = stream.fps()

        # Use math.isclose for robust float comparison
        if not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            raise RuntimeError(
                f"Failed to set requested FPS {self.fps} for {self}. Actual value reported: {actual_fps}."
            )
        logger.debug(f"FPS set to {actual_fps} for {self}.")

    def _validate_width_and_height(self, stream) -> None:
        """Validates and sets the internal capture width and height based on actual stream width."""
        actual_width = int(round(stream.width()))
        actual_height = int(round(stream.height()))

        if self.capture_width != actual_width:
            logger.warning(
                f"Requested capture width {self.capture_width} for {self}, but camera reported {actual_width}."
            )
            raise RuntimeError(
                f"Failed to set requested capture width {self.capture_width} for {self}. Actual value: {actual_width}."
            )
        logger.debug(f"Capture width set to {actual_width} for {self}.")

        if self.capture_height != actual_height:
            logger.warning(
                f"Requested capture height {self.capture_height} for {self}, but camera reported {actual_height}."
            )
            raise RuntimeError(
                f"Failed to set requested capture height {self.capture_height} for {self}. Actual value: {actual_height}."
            )
        logger.debug(f"Capture height set to {actual_height} for {self}.")

    def read_depth(self, timeout_ms: int = 100) -> np.ndarray:
        """
        Reads a single frame (depth) synchronously from the camera.

        This is a blocking call. It waits for a coherent set of frames (depth)
        from the camera hardware via the RealSense pipeline.

        Args:
            timeout_ms (int): Maximum time in milliseconds to wait for a frame. Defaults to 100ms.

        Returns:
            np.ndarray: The depth map as a NumPy array (height, width)
                  of type `np.uint16` (raw depth values in millimeters) and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the pipeline fails or frames are invalid.
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.use_depth:
            raise RuntimeError(
                f"Failed to capture depth frame '.read_depth()'. Depth stream is not enabled for {self}."
            )

        start_time = time.perf_counter()

        ret, frame = self.rs_pipeline.try_wait_for_frames(timeout_ms=timeout_ms)

        if not ret or frame is None:
            raise RuntimeError(f"{self} failed to capture frame. Returned status='{ret}'.")

        depth_frame = frame.get_depth_frame()
        depth_map = np.asanyarray(depth_frame.get_data())

        depth_map_processed = self._postprocess_image(depth_map, depth_frame=True)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} synchronous read took: {read_duration_ms:.1f}ms")

        return depth_map_processed

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 100) -> np.ndarray:
        """
        Reads a single frame (color) synchronously from the camera.

        This is a blocking call. It waits for a coherent set of frames (color)
        from the camera hardware via the RealSense pipeline.

        Args:
            timeout_ms (int): Maximum time in milliseconds to wait for a frame. Defaults to 100ms.

        Returns:
            np.ndarray: The captured color frame as a NumPy array
              (height, width, channels), processed according to `color_mode` and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the pipeline fails or frames are invalid.
            ValueError: If an invalid `color_mode` is requested.
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        ret, frame = self.rs_pipeline.try_wait_for_frames(
            timeout_ms=timeout_ms
        )  # NOTE(Steven): This read has a timeout while opencv doesn't

        if not ret or frame is None:
            raise RuntimeError(
                f"Failed to capture frame from {self}. '.read()' returned status={ret} and frame is None."
            )

        color_frame = frame.get_color_frame()
        color_image_raw = np.asanyarray(color_frame.get_data())

        color_image_processed = self._postprocess_image(color_image_raw, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} synchronous read took: {read_duration_ms:.1f}ms")

        return color_image_processed

    def _postprocess_image(
        self, image: np.ndarray, color_mode: ColorMode | None = None, depth_frame: bool = False
    ) -> np.ndarray:
        """
        Applies color conversion, dimension validation, and rotation to a raw color frame.

        Args:
            image (np.ndarray): The raw image frame (expected RGB format from RealSense).
            color_mode (Optional[ColorMode]): The target color mode (RGB or BGR). If None,
                                             uses the instance's default `self.color_mode`.

        Returns:
            np.ndarray: The processed image frame according to `self.color_mode` and `self.rotation`.

        Raises:
            ValueError: If the requested `color_mode` is invalid.
            RuntimeError: If the raw frame dimensions do not match the configured
                          `width` and `height`.
        """

        if color_mode and color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if depth_frame:
            h, w = image.shape
        else:
            h, w, _c = image.shape

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"Captured frame dimensions ({h}x{w}) do not match configured capture dimensions ({self.capture_height}x{self.capture_width}) for {self}."
            )

        processed_image = image
        if self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            logger.debug(f"Converted frame from RGB to BGR for {self}.")

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            processed_image = cv2.rotate(processed_image, self.rotation)
            logger.debug(f"Rotated frame by {self.config.rotation} degrees for {self}.")

        return processed_image

    def _read_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.

        Continuously reads frames (color and optional depth) using `read()`
        and places the latest result (single image or tuple) into the `frame_queue`.
        It overwrites any previous frame in the queue.
        """
        logger.debug(f"Starting read loop thread for {self}.")
        while not self.stop_event.is_set():
            try:
                frame_data = self.read(timeout_ms=500)

                with contextlib.suppress(queue.Empty):
                    _ = self.frame_queue.get_nowait()
                self.frame_queue.put(frame_data)
                logger.debug(f"Frame data placed in queue for {self}.")

            except DeviceNotConnectedError:
                logger.error(f"Read loop for {self} stopped: Camera disconnected.")
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

        logger.debug(f"Stopping read loop thread for {self}.")

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()
        logger.debug(f"Read thread started for {self}.")

    def _stop_read_thread(self):
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            logger.debug(f"Signaling stop event for read thread of {self}.")
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            logger.debug(f"Waiting for read thread of {self} to join...")
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning(f"Read thread for {self} did not terminate gracefully after 2 seconds.")
            else:
                logger.debug(f"Read thread for {self} joined successfully.")

        self.thread = None
        self.stop_event = None

    # NOTE(Steven): Missing implementation for depth for now
    def async_read(self, timeout_ms: float = 100) -> np.ndarray:
        """
        Reads the latest available frame data (color or color+depth) asynchronously.

        This method retrieves the most recent frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        only waits for a frame to appear in the internal queue up to the specified
        timeout.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available in the queue. Defaults to 100ms (0.1 seconds).

        Returns:
            np.ndarray:
            The latest captured frame data (color image), processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If the background thread died unexpectedly or another queue error occurs.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        try:
            return self.frame_queue.get(timeout=timeout_ms / 1000.0)
        except queue.Empty as e:
            thread_alive = self.thread is not None and self.thread.is_alive()
            logger.error(
                f"Timeout waiting for frame from {self} queue after {timeout_ms}ms. "
                f"(Read thread alive: {thread_alive})"
            )
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self.serial_number} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            ) from e
        except Exception as e:
            logger.exception(f"Unexpected error getting frame data from queue for {self}: {e}")
            raise RuntimeError(
                f"Error getting frame data from queue for camera {self.serial_number}: {e}"
            ) from e

    def disconnect(self):
        """
        Disconnects from the camera, stops the pipeline, and cleans up resources.

        Stops the background read thread (if running) and stops the RealSense pipeline.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected (pipeline not running).
        """

        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self.rs_pipeline is not None:
            logger.debug(f"Stopping RealSense pipeline object for {self}.")
            self.rs_pipeline.stop()
            self.rs_pipeline = None
            self.rs_profile = None

        logger.info(f"{self} disconnected.")
