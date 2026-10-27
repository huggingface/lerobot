# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
Provides the OrbbecCamera class for capturing frames from Orbbec cameras.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray

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


class OrbbecCamera(Camera):
    """
    Manages interactions with Orbbec cameras for frame and depth recording.

    This class provides an interface similar to `RealSenseCamera` but tailored for
    Orbbec devices, leveraging the `pyorbbecsdk` library. It uses the camera's
    unique serial number for identification, offering more stability than device
    indices. It also supports capturing depth maps alongside color frames; when both
    streams are enabled, depth is aligned to the color stream so both share the same
    resolution.

    Use the provided utility script to find available cameras and default profiles:
    ```bash
    lerobot-find-cameras orbbec
    ```

    An `OrbbecCamera` instance requires a configuration object specifying the
    camera's serial number or a unique device name. If using the name, ensure only
    one camera with that name is connected.

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

        # Read 1 frame asynchronously (waits for new frame with a timeout)
        async_image = camera.async_read()

        # Get the latest frame immediately (no wait)
        latest_image = camera.read_latest()

        camera.disconnect()

        # Example with depth capture and custom settings
        custom_config = OrbbecCameraConfig(
            serial_number_or_name="CP1234567890",  # Replace with actual SN
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR,  # Request BGR output
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True,
        )
        depth_camera = OrbbecCamera(custom_config)
        depth_camera.connect()

        # Read 1 depth frame
        depth_map = depth_camera.read_depth()
        ```
    """

    def __init__(self, config: OrbbecCameraConfig):
        """
        Initializes the OrbbecCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        require_package("pyorbbecsdk2", extra="orbbec", import_name="pyorbbecsdk")
        super().__init__(config)

        self.config = config
        # Devices and pipelines need the context alive, so it lives as long as the camera.
        self.context = ob.Context()
        self.serial_number = self._find_serial_number_from_name(config.serial_number_or_name)
        self.color_mode = config.color_mode
        self.use_rgb = config.use_rgb
        self.use_depth = config.use_depth

        self.pipeline: ob.Pipeline | None = None
        self.align_filter: ob.AlignFilter | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self.capture_width: int | None = None
        self.capture_height: int | None = None
        self._reset_connection_settings()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    def _reset_connection_settings(self) -> None:
        """Restore settings that may have been auto-detected during a previous connection."""
        self.fps = self.config.fps
        self.width = self.config.width
        self.height = self.config.height
        self.warmup_s = self.config.warmup_s
        self.capture_width, self.capture_height = self.width, self.height
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.capture_width, self.capture_height = self.height, self.width

    @property
    def is_connected(self) -> bool:
        """Checks if the camera pipeline is started."""
        return self.pipeline is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Orbbec cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            contains 'type', 'id' (serial number), 'name', firmware version, connection
            type, and the default color profile properties (width, height, fps, format).

        Raises:
            ImportError: If pyorbbecsdk is not installed.
        """
        require_package("pyorbbecsdk2", extra="orbbec", import_name="pyorbbecsdk")
        found_cameras_info = []
        # Keep the context alive: devices obtained from the list need its device manager.
        context = ob.Context()
        devices = context.query_devices()

        for index in range(devices.get_count()):
            device = devices.get_device_by_index(index)
            device_info = device.get_device_info()
            camera_info: dict[str, Any] = {
                "name": device_info.get_name(),
                "type": "Orbbec",
                "id": device_info.get_serial_number(),
                "firmware_version": device_info.get_firmware_version(),
                "connection_type": device_info.get_connection_type(),
                "product_id": device_info.get_pid(),
            }

            try:
                profile_list = ob.Pipeline(device).get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
                profile = profile_list.get_default_video_stream_profile()
                camera_info["default_stream_profile"] = {
                    "stream_type": "Color",
                    "format": profile.get_format().name,
                    "width": profile.get_width(),
                    "height": profile.get_height(),
                    "fps": profile.get_fps(),
                }
            except Exception as e:
                logger.warning(f"Could not query the default color profile of {camera_info['id']}: {e}")

            found_cameras_info.append(camera_info)

        return found_cameras_info

    def _find_serial_number_from_name(self, name: str) -> str:
        """Finds the serial number for a given serial number or unique camera name.

        Unlike RealSense serial numbers, Orbbec ones are alphanumeric and cannot be told
        apart from names, so serial numbers and names are matched in the same pass.
        """
        devices = self.context.query_devices()
        found_serial_numbers = []
        for index in range(devices.get_count()):
            serial_number = devices.get_device_serial_number_by_index(index)
            if serial_number == name:
                return serial_number
            if devices.get_device_name_by_index(index) == name:
                found_serial_numbers.append(serial_number)

        if not found_serial_numbers:
            raise ValueError(
                f"No Orbbec camera found with serial number or name '{name}'. "
                "Run `lerobot-find-cameras orbbec` to find available cameras."
            )

        if len(found_serial_numbers) > 1:
            raise ValueError(
                f"Multiple Orbbec cameras found with name '{name}'. "
                f"Please use a unique serial number instead. Found SNs: {found_serial_numbers}"
            )

        return found_serial_numbers[0]

    def _select_profiles(self, pipeline: "ob.Pipeline") -> list["ob.VideoStreamProfile"]:
        """Selects the color and/or depth stream profiles matching the requested settings.

        Color is requested directly in the output color mode so frames need no conversion.
        When both streams are enabled, depth is aligned to color afterwards, so any depth
        resolution can be used.
        """
        width, height, fps = self.capture_width or 0, self.capture_height or 0, self.fps or 0
        profiles = []

        if self.use_rgb:
            color_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            if not fps:
                default_profile = color_profiles.get_default_video_stream_profile()
                width, height = default_profile.get_width(), default_profile.get_height()
                fps = default_profile.get_fps()
            color_format = ob.OBFormat.BGR if self.color_mode == ColorMode.BGR else ob.OBFormat.RGB
            profiles.append(color_profiles.get_video_stream_profile(width, height, color_format, fps))
            width, height = 0, 0

        if self.use_depth:
            depth_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            profiles.append(depth_profiles.get_video_stream_profile(width, height, ob.OBFormat.Y16, fps))

        return profiles

    def _open_pipeline(self) -> None:
        """Opens the device, starts the pipeline and the background read thread.

        Raises:
            ConnectionError: If the camera fails to open or no profile matches the requested settings.
        """
        self._reset_connection_settings()

        try:
            device = self.context.query_devices().get_device_by_serial_number(self.serial_number)
            pipeline = ob.Pipeline(device)
            profiles = self._select_profiles(pipeline)
            ob_config = ob.Config()
            for profile in profiles:
                ob_config.enable_stream(profile)
            if self.use_rgb and self.use_depth:
                pipeline.enable_frame_sync()
            pipeline.start(ob_config)
        except Exception as e:
            raise ConnectionError(
                f"Failed to open {self} with fps={self.fps}, width={self.capture_width}, "
                f"height={self.capture_height}. Run `lerobot-find-cameras orbbec` to find available cameras."
            ) from e

        self.pipeline = pipeline
        if self.use_rgb and self.use_depth:
            self.align_filter = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)

        try:
            self._configure_capture_settings(profiles[0])
            self._start_read_thread()
        except BaseException:
            self._release_after_failed_setup()
            raise

    def _run_warmup(self) -> None:
        """Blocks until at least one valid frame has been captured by the background thread.

        Raises:
            ConnectionError: If no frame arrives before ``warmup_s`` elapses.
        """
        # Like RealSense, Orbbec cameras need a bit of time before delivering the first frames.
        self.warmup_s = max(self.warmup_s, 1)

        warmup_read = self.async_read if self.use_rgb else self.async_read_depth
        start_time = time.time()
        while time.time() - start_time < self.warmup_s:
            warmup_read(timeout_ms=self.warmup_s * 1000)
            time.sleep(0.1)

    def _release_after_failed_setup(self) -> None:
        """Releases the device handle after a failed connection attempt."""
        try:
            self._cleanup_resources()
        except Exception:
            logger.exception(f"Failed to fully clean up {self} after connect() failed.")

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """
        Connects to the Orbbec camera specified in the configuration.

        Opens the device, configures the required streams (color and optionally depth),
        starts the pipeline and the background read thread, and validates the actual
        stream settings.

        Args:
            warmup (bool): If True, waits at connect() time until at least one valid frame
                           has been captured by the background thread. Defaults to True.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ConnectionError: If the camera fails to open, no profile matches the requested
                settings, or no frame is captured during warmup.
        """
        self._open_pipeline()

        if warmup:
            try:
                self._run_warmup()
            except BaseException:
                self._release_after_failed_setup()
                raise

        logger.info(f"{self} connected.")

    def _configure_capture_settings(self, profile: "ob.VideoStreamProfile") -> None:
        """Sets fps, width, and height from the active stream profile if not already configured.

        Uses the color stream profile (or the depth stream profile when the color stream
        is disabled). Handles rotation by swapping width/height when needed. Original
        capture dimensions are always stored.
        """
        if self.fps is None:
            self.fps = profile.get_fps()

        if self.width is None or self.height is None:
            actual_width, actual_height = profile.get_width(), profile.get_height()
            self.capture_width, self.capture_height = actual_width, actual_height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = actual_height, actual_width
            else:
                self.width, self.height = actual_width, actual_height

    def _read(self, read_depth: bool = False) -> NDArray[Any]:
        """Shared helper for :meth:`read`/:meth:`read_depth`: wait for a fresh color or depth frame."""
        self.new_frame_event.clear()
        return self._async_read(timeout_ms=10000, read_depth=read_depth)

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        """
        Reads a single frame (color) synchronously from the camera.

        This is a blocking call. It waits for the next color frame captured by the
        background read thread.

        Returns:
            np.ndarray: The captured color frame as a NumPy array
              (height, width, channels), processed according to `color_mode` and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If the color stream is disabled or the read thread is not running.
        """
        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — camera was configured with use_rgb=False.")

        return self._read()

    @check_if_not_connected
    def read_depth(self) -> NDArray[Any]:
        """
        Reads a single depth frame synchronously from the camera.

        Returns:
            np.ndarray: The depth map as a NumPy array (height, width, 1)
                  of type `np.uint16` (depth values in millimeters).

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If the depth stream is disabled or the read thread is not running.
        """
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera was configured with use_depth=False.")

        return self._read(read_depth=True)

    def _postprocess_image(self, frame: Any, depth_frame: bool = False) -> NDArray[Any]:
        """
        Converts a raw Orbbec frame to a NumPy array, validates its dimensions and applies rotation.

        Color frames are already delivered in the requested color mode, and depth frames
        are converted to millimeters.

        Args:
            frame: The raw Orbbec color or depth frame.
            depth_frame: Whether the frame is a depth frame.

        Returns:
            np.ndarray: The processed image, rotated according to `self.rotation`.

        Raises:
            RuntimeError: If the frame dimensions do not match the configured `width` and `height`.
        """
        h, w = frame.get_height(), frame.get_width()
        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        if depth_frame:
            image = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(h, w)
            depth_scale = frame.get_depth_scale()
            if depth_scale != 1.0:
                image = (image * depth_scale).astype(np.uint16)
        else:
            image = np.asarray(frame.get_data()).reshape(h, w, 3)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            image = cv2.rotate(image, self.rotation)

        return image

    def _read_loop(self) -> None:
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a color/depth frame set (blocking call with 10s timeout)
        2. Stores result in latest_color_frame/latest_depth_frame and updates timestamp (thread-safe)
        3. Sets new_frame_event to notify listeners

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        stop_event = self.stop_event
        if stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not stop_event.is_set():
            try:
                if self.pipeline is None:
                    raise DeviceNotConnectedError(f"{self} pipeline is not running.")

                frames = self.pipeline.wait_for_frames(10000)
                if frames is None:
                    raise RuntimeError(f"{self} read failed (no frames received).")
                if self.align_filter is not None:
                    aligned_frames = self.align_filter.process(frames)
                    # Incomplete frame set (streams not yet in sync), wait for the next one.
                    if aligned_frames is None:
                        continue
                    frames = aligned_frames.as_frame_set()

                if self.use_rgb:
                    color_frame = frames.get_color_frame()
                    if color_frame is None:
                        raise RuntimeError(f"{self} read failed (no color frame received).")
                    processed_color_frame = self._postprocess_image(color_frame)

                if self.use_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame is None:
                        raise RuntimeError(f"{self} read failed (no depth frame received).")
                    # (H, W) -> (H, W, 1)
                    processed_depth_frame = self._postprocess_image(depth_frame, depth_frame=True)[
                        ..., np.newaxis
                    ]

                capture_time = time.perf_counter()

                with self.frame_lock:
                    # Under the lock, so a late frame cannot resurrect the buffer _stop_read_thread() cleared.
                    if stop_event.is_set():
                        break
                    if self.use_rgb:
                        self.latest_color_frame = processed_color_frame
                    if self.use_depth:
                        self.latest_depth_frame = processed_depth_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        self._stop_read_thread()

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
            if self.thread.is_alive():  # pragma: no cover
                logger.warning(f"{self} read thread did not terminate within timeout.")

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    def _cleanup_resources(self) -> None:
        """Stop background reads and stop the pipeline, including after partial setup."""
        read_thread = self.thread
        pipeline = self.pipeline

        try:
            self._stop_read_thread()
        finally:
            self.pipeline = None
            self.align_filter = None
            try:
                if pipeline is not None:
                    pipeline.stop()
            finally:
                # Stopping the pipeline may unblock a hardware read that outlived
                # the first bounded join in _stop_read_thread().
                if read_thread is not None and read_thread.is_alive():
                    read_thread.join(timeout=2.0)
                    if read_thread.is_alive():  # pragma: no cover
                        logger.warning(f"{self} read thread remained alive after stopping the pipeline.")

    def _async_read(self, timeout_ms: float, read_depth: bool = False) -> NDArray[Any]:
        """Shared helper for :meth:`async_read`/:meth:`async_read_depth`: return the latest buffered frame."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        Reads the latest available frame data (color) asynchronously.

        This method retrieves the most recent color frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray: The latest captured frame data (color image), processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If the background thread died unexpectedly or another error occurs.
        """
        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — camera was configured with use_rgb=False.")

        return self._async_read(timeout_ms=timeout_ms)

    def _read_latest(self, max_age_ms: int, read_depth: bool = False) -> NDArray[Any]:
        """Shared helper for :meth:`read_latest`/:meth:`read_latest_depth`: peek the latest buffered frame."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent (color) frame captured immediately (Peeking).

        This method is non-blocking and returns whatever is currently in the
        memory buffer. The frame may be stale,
        meaning it could have been captured a while ago (hanging camera scenario e.g.).

        Returns:
            NDArray[Any]: The frame image (numpy array).

        Raises:
            TimeoutError: If the latest frame is older than `max_age_ms`.
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If the camera is connected but has not captured any frames yet.
        """
        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — camera was configured with use_rgb=False.")

        return self._read_latest(max_age_ms=max_age_ms)

    @check_if_not_connected
    def async_read_depth(self, timeout_ms: float = 200) -> NDArray[np.uint16]:
        """Read the latest depth frame asynchronously, in millimeters.

        Mirrors :meth:`async_read` but returns the depth stream rather than the
        color stream. Output is ``np.uint16`` of shape ``(H, W, 1)``, where each
        pixel is the distance from the sensor in millimeters.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If ``use_depth`` is ``False`` for this camera, or if
                the background read thread is not running.
            TimeoutError: If no frame becomes available within ``timeout_ms``.
        """
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera was configured with use_depth=False.")

        return self._async_read(timeout_ms=timeout_ms, read_depth=True)

    @check_if_not_connected
    def read_latest_depth(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent depth frame in millimeters (peeking).

        Non-blocking counterpart of :meth:`read_latest` for the depth stream.
        Output is ``np.uint16`` of shape ``(H, W, 1)``, where each pixel is the
        distance from the sensor in millimeters.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If ``use_depth`` is ``False`` for this camera, or if
                no depth frame has been captured yet.
            TimeoutError: If the latest depth frame is older than ``max_age_ms``.
        """
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera was configured with use_depth=False.")

        return self._read_latest(max_age_ms=max_age_ms, read_depth=True)

    @check_if_not_connected
    def disconnect(self) -> None:
        """
        Disconnects from the camera, stops the pipeline, and cleans up resources.

        Stops the background read thread (if running) and stops the Orbbec pipeline.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        self._cleanup_resources()

        logger.info(f"{self} disconnected.")
