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
Provides the OrbbecCamera class for capturing RGB frames from Orbbec cameras.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from lerobot.utils.import_utils import _pyorbbecsdk_available, require_package

if TYPE_CHECKING or _pyorbbecsdk_available:
    # NOTE: The package is distributed as `pyorbbecsdk2` on PyPI but imported as `pyorbbecsdk`.
    import pyorbbecsdk as ob
else:
    ob = None

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec import OrbbecCameraConfig

logger = logging.getLogger(__name__)

PACKAGE_NAME = "pyorbbecsdk2"
IMPORT_NAME = "pyorbbecsdk"


def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def _device_info_value(device_info: Any, method_name: str) -> Any:
    try:
        return getattr(device_info, method_name)()
    except Exception:
        return None


class OrbbecCamera(Camera):
    """
    Manages Orbbec RGB camera capture using the `pyorbbecsdk` library.

    This class provides an interface similar to `RealSenseCamera` but tailored for
    Orbbec devices. It uses the camera's unique serial number for identification,
    offering more stability than device indices, or a unique device name.

    This first version is color-only. Depth capture is intentionally not implemented
    so LeRobot dataset image features keep the usual RGB shape.

    Use the provided utility script to find available cameras and default profiles:
    ```bash
    lerobot-find-cameras orbbec
    ```

    The camera's default settings (FPS, resolution) from the stream profile are used
    unless overridden in the configuration.

    Example:
        ```python
        from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig
        from lerobot.cameras import ColorMode, Cv2Rotation

        # Basic usage with serial number
        config = OrbbecCameraConfig(serial_number_or_name="CP1234567890")  # Replace with actual SN
        camera = OrbbecCamera(config)
        camera.connect()

        # Read 1 frame synchronously (blocking)
        color_image = camera.read()

        # Read 1 frame asynchronously (waits for a new frame with a timeout)
        async_image = camera.async_read()

        camera.disconnect()

        # Example with custom settings
        custom_config = OrbbecCameraConfig(
            serial_number_or_name="CP1234567890",  # Replace with actual SN
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR,  # Request BGR output
            rotation=Cv2Rotation.NO_ROTATION,
        )
        camera = OrbbecCamera(custom_config)
        ```
    """

    def __init__(self, config: OrbbecCameraConfig):
        """
        Initializes the OrbbecCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        require_package(PACKAGE_NAME, extra="orbbec", import_name=IMPORT_NAME)
        super().__init__(config)

        self.config = config
        self.serial_number_or_name = config.serial_number_or_name
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s

        self.device: Any = None
        self.serial_number: str | None = None
        self.device_name: str | None = None

        self.pipeline: Any = None
        self.profile: Any = None
        self._is_connected = False

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self.capture_width: int | None = None
        self.capture_height: int | None = None
        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        identifier = self.serial_number or self.serial_number_or_name
        return f"{self.__class__.__name__}({identifier})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera pipeline is started and streaming."""
        return self._is_connected

    @staticmethod
    def _iter_devices() -> Any:
        require_package(PACKAGE_NAME, extra="orbbec", import_name=IMPORT_NAME)
        context = ob.Context()
        devices = context.query_devices()
        for index in range(devices.get_count()):
            yield devices.get_device_by_index(index)

    @staticmethod
    def _device_metadata(device: Any) -> dict[str, Any]:
        device_info = device.get_device_info()
        serial_number = _device_info_value(device_info, "get_serial_number")
        name = _device_info_value(device_info, "get_name")

        return {
            "name": name,
            "type": "Orbbec",
            "id": serial_number,
            "uid": _device_info_value(device_info, "get_uid"),
            "vid": _device_info_value(device_info, "get_vid"),
            "pid": _device_info_value(device_info, "get_pid"),
            "firmware_version": _device_info_value(device_info, "get_firmware_version"),
            "hardware_version": _device_info_value(device_info, "get_hardware_version"),
            "connection_type": _device_info_value(device_info, "get_connection_type"),
        }

    @staticmethod
    def _iter_video_profiles(profile_list: Any) -> Any:
        for index in range(profile_list.get_count()):
            profile = profile_list.get_stream_profile_by_index(index)
            if hasattr(profile, "is_video_stream_profile") and profile.is_video_stream_profile():
                profile = profile.as_video_stream_profile()
            if all(hasattr(profile, attr) for attr in ("get_format", "get_width", "get_height", "get_fps")):
                yield profile

    @staticmethod
    def _profile_to_dict(profile: Any) -> dict[str, Any]:
        return {
            "stream_type": _enum_name(profile.get_type()),
            "format": _enum_name(profile.get_format()),
            "width": profile.get_width(),
            "height": profile.get_height(),
            "fps": profile.get_fps(),
        }

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Orbbec cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            contains 'type', 'id' (serial number), 'name', and other available specs,
            along with the default color stream profile (width, height, fps, format).

        Raises:
            ImportError: If pyorbbecsdk is not installed.
        """
        found_cameras_info = []

        for device in OrbbecCamera._iter_devices():
            camera_info = OrbbecCamera._device_metadata(device)

            try:
                pipeline = ob.Pipeline(device)
                profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
                color_profile = profile_list.get_default_video_stream_profile()
                camera_info["default_stream_profile"] = OrbbecCamera._profile_to_dict(color_profile)
            except Exception as e:
                camera_info["default_stream_profile_error"] = str(e)

            found_cameras_info.append(camera_info)

        return found_cameras_info

    def _find_device(self) -> Any:
        """Finds the Orbbec device matching the configured serial number or name."""
        devices = list(self._iter_devices())

        for device in devices:
            metadata = self._device_metadata(device)
            if str(metadata.get("id")) == self.serial_number_or_name:
                self.serial_number = str(metadata.get("id"))
                self.device_name = str(metadata.get("name"))
                return device

        named_devices = []
        for device in devices:
            metadata = self._device_metadata(device)
            if str(metadata.get("name")) == self.serial_number_or_name:
                named_devices.append((device, metadata))

        if len(named_devices) == 1:
            device, metadata = named_devices[0]
            self.serial_number = str(metadata.get("id"))
            self.device_name = str(metadata.get("name"))
            return device

        if len(named_devices) > 1:
            serial_numbers = [str(metadata.get("id")) for _, metadata in named_devices]
            raise ValueError(
                f"Multiple Orbbec cameras found with name '{self.serial_number_or_name}'. "
                f"Please use a unique serial number instead. Found SNs: {serial_numbers}"
            )

        available = [
            {
                "serial_number": metadata.get("id"),
                "name": metadata.get("name"),
            }
            for metadata in [self._device_metadata(device) for device in devices]
        ]
        raise ValueError(
            f"No Orbbec camera found with serial/name '{self.serial_number_or_name}'. "
            f"Available cameras: {available}"
        )

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """
        Connects to the Orbbec camera specified in the configuration.

        Initializes the Orbbec pipeline, selects and enables the color stream, starts
        the pipeline, and validates the actual stream settings.

        Args:
            warmup: If True (default), reads frames for `warmup_s` seconds before returning.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ValueError: If the configuration is invalid (e.g. serial/name not found or not unique).
            ConnectionError: If the camera is found but fails to start the pipeline.
            RuntimeError: If no supported color profile matches the requested settings.
        """
        self.device = self._find_device()
        self.pipeline = ob.Pipeline(self.device)
        config = ob.Config()

        self.profile = self._select_color_profile(self.pipeline)
        config.enable_stream(self.profile)

        try:
            self.pipeline.start(config)
        except Exception as e:
            self.pipeline = None
            self.profile = None
            self.device = None
            self._is_connected = False
            raise ConnectionError(
                f"Failed to open {self}. Run `lerobot-find-cameras orbbec` to find available cameras."
            ) from e

        self._is_connected = True
        self._configure_capture_settings()

        if warmup:
            time.sleep(1)
            last_error = None
            warmed_up = False
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    self.read(timeout_ms=1000)
                    warmed_up = True
                except RuntimeError as e:
                    last_error = e
                time.sleep(0.1)
            if not warmed_up and last_error is not None:
                self.disconnect()
                raise RuntimeError(f"{self} failed to read a warmup frame.") from last_error

        logger.info(f"{self} connected.")

    def _select_color_profile(self, pipeline: Any) -> Any:
        """Selects a supported color stream profile matching the requested settings."""
        profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        profiles = list(self._iter_video_profiles(profile_list))

        if self.width and self.height and self.fps:
            matches = [
                profile
                for profile in profiles
                if profile.get_width() == self.capture_width
                and profile.get_height() == self.capture_height
                and profile.get_fps() == self.fps
                and self._is_supported_format(profile.get_format())
            ]
            if matches:
                return self._prefer_color_profile(matches)

            raise RuntimeError(
                f"{self} could not find a supported color profile matching "
                f"width={self.capture_width}, height={self.capture_height}, fps={self.fps}. "
                f"Available profiles: {self._format_available_profiles(profiles)}"
            )

        default_profile = profile_list.get_default_video_stream_profile()
        if not self._is_supported_format(default_profile.get_format()):
            raise RuntimeError(
                f"{self} default color profile uses unsupported format "
                f"{_enum_name(default_profile.get_format())}. "
                f"Available profiles: {self._format_available_profiles(profiles)}"
            )
        return default_profile

    @staticmethod
    def _supported_formats() -> set[Any]:
        return {
            ob.OBFormat.RGB,
            ob.OBFormat.BGR,
            ob.OBFormat.MJPG,
            ob.OBFormat.YUYV,
            ob.OBFormat.UYVY,
        }

    @staticmethod
    def _format_preference() -> dict[Any, int]:
        return {
            ob.OBFormat.RGB: 0,
            ob.OBFormat.BGR: 1,
            ob.OBFormat.MJPG: 2,
            ob.OBFormat.YUYV: 3,
            ob.OBFormat.UYVY: 4,
        }

    @classmethod
    def _is_supported_format(cls, color_format: Any) -> bool:
        return color_format in cls._supported_formats()

    @classmethod
    def _prefer_color_profile(cls, profiles: list[Any]) -> Any:
        return min(profiles, key=lambda profile: cls._format_preference().get(profile.get_format(), 100))

    @classmethod
    def _format_available_profiles(cls, profiles: list[Any]) -> list[dict[str, Any]]:
        return [cls._profile_to_dict(profile) for profile in profiles]

    @check_if_not_connected
    def _configure_capture_settings(self) -> None:
        """Sets fps, width, and height from the device stream profile if not already configured.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If no active color stream profile is available.
        """
        if self.profile is None:
            raise RuntimeError(f"{self} has no active color stream profile.")

        if self.fps is None:
            self.fps = self.profile.get_fps()

        if self.width is None or self.height is None:
            actual_width = int(round(self.profile.get_width()))
            actual_height = int(round(self.profile.get_height()))
            self.capture_width, self.capture_height = actual_width, actual_height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = actual_height, actual_width
            else:
                self.width, self.height = actual_width, actual_height

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> NDArray[Any]:
        """
        Reads a single color frame synchronously from the camera.

        This is a blocking call. It waits for a coherent set of frames from the camera
        hardware via the Orbbec pipeline.

        Args:
            color_mode: If provided, overrides the configured color mode for this read.
            timeout_ms: Maximum time in milliseconds to wait for a frame.

        Returns:
            np.ndarray: The captured color frame as a NumPy array (height, width, 3),
            processed according to `color_mode` and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the pipeline fails or frames are invalid.
            ValueError: If an invalid `color_mode` is requested.
        """
        start_time = time.perf_counter()

        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            raise RuntimeError(f"{self} read failed (no frames received).")

        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError(f"{self} read failed (no color frame received).")

        color_image_raw = self._frame_to_rgb_image(color_frame)
        color_image_processed = self._postprocess_image(color_image_raw, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return color_image_processed

    def _frame_to_rgb_image(self, frame: Any) -> NDArray[Any]:
        """Converts a raw Orbbec color frame to an RGB NumPy array."""
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())

        if color_format == ob.OBFormat.RGB:
            return data.reshape((height, width, 3)).copy()

        if color_format == ob.OBFormat.BGR:
            image = data.reshape((height, width, 3))
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if color_format == ob.OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"{self} failed to decode MJPG color frame.")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if color_format == ob.OBFormat.YUYV:
            image = data.reshape((height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUYV)

        if color_format == ob.OBFormat.UYVY:
            image = data.reshape((height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2RGB_UYVY)

        raise RuntimeError(
            f"{self} received unsupported color format {_enum_name(color_format)}. "
            f"Supported formats: {[format_.name for format_ in self._supported_formats()]}"
        )

    def _postprocess_image(self, image: NDArray[Any], color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        Applies color conversion, dimension validation, and rotation to a raw color frame.

        Args:
            image: The raw image frame (expected RGB format).
            color_mode: If provided, overrides the configured color mode for this frame.

        Returns:
            np.ndarray: The processed image frame according to `color_mode` and `self.rotation`.

        Raises:
            ValueError: If the requested `color_mode` is invalid.
            RuntimeError: If the raw frame dimensions do not match the configured width and height.
        """
        requested_color_mode = self.color_mode if color_mode is None else color_mode

        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured "
                f"width={self.capture_width} or height={self.capture_height}."
            )

        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        processed_image = image
        if requested_color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self) -> None:
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration it reads a color frame, stores it (thread-safe) and notifies
        listeners. Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        stop_event = self.stop_event
        if stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        while not stop_event.is_set():
            try:
                color_image = self.read(timeout_ms=500)

                with self.frame_lock:
                    self.latest_frame = color_image
                    self.latest_timestamp = time.perf_counter()
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

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

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        Reads the latest available color frame asynchronously.

        This method retrieves the most recent frame captured by the background read
        thread. It does not block waiting for the camera hardware directly, but may
        wait up to `timeout_ms` for the background thread to provide a new frame.

        Args:
            timeout_ms: Maximum time in milliseconds to wait for a new frame.

        Returns:
            np.ndarray: The latest captured color frame, processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no new frame becomes available within the specified timeout.
            RuntimeError: If the background thread set the event but no frame is available.
        """
        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent color frame captured immediately (peeking).

        This method is non-blocking and returns whatever is currently in the memory
        buffer. The frame may be stale, meaning it could have been captured a while
        ago (hanging camera scenario, e.g.).

        Args:
            max_age_ms: Maximum allowed age of the latest frame in milliseconds.

        Returns:
            NDArray[Any]: The frame image (numpy array).

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If the camera is connected but has not captured any frames yet.
            TimeoutError: If the latest frame is older than `max_age_ms`.
        """
        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    def disconnect(self) -> None:
        """
        Disconnects from the camera, stops the pipeline, and cleans up resources.

        Stops the background read thread (if running) and stops the Orbbec pipeline.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self.pipeline is not None:
            self.pipeline.stop()

        self.pipeline = None
        self.profile = None
        self.device = None
        self._is_connected = False

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
