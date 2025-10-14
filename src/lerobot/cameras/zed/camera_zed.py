"""
Provides the ZedCamera class for capturing frames from ZED stereo cameras.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np

try:
    import pyzed.sl as sl
except Exception as e:
    logging.info(f"Could not import ZED SDK: {e}")

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_zed import ZedCameraConfig

logger = logging.getLogger(__name__)


class ZedCamera(Camera):
    """
    Manages interactions with ZED stereo cameras for frame and depth recording.

    This class provides an interface for ZED cameras, leveraging the `pyzed.sl` library.
    It uses the camera's unique serial number for identification. ZED cameras support
    high-quality depth sensing and various resolutions.

    Use the provided utility script to find available camera indices and default profiles:
    ```bash
    lerobot-find-cameras zed
    ```

    A `ZedCamera` instance requires a configuration object specifying the
    camera's serial number or a unique device name.

    Example:
        ```python
        from lerobot.cameras.zed import ZedCamera, ZedCameraConfig
        from lerobot.cameras import ColorMode, Cv2Rotation

        # Basic usage with serial number
        config = ZedCameraConfig(serial_number_or_name="0123456789") # Replace with actual SN
        camera = ZedCamera(config)
        camera.connect()

        # Read 1 frame synchronously
        color_image = camera.read()
        print(color_image.shape)

        # Read 1 depth frame
        depth_map = camera.read_depth()

        # When done, properly disconnect the camera.
        camera.disconnect()

        # Example with custom settings
        custom_config = ZedCameraConfig(
            serial_number_or_name="0123456789",
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR,
            rotation=Cv2Rotation.ROTATE_90,
            use_depth=True,
            depth_mode="NEURAL"
        )
        depth_camera = ZedCamera(custom_config)
        depth_camera.connect()
        ```
    """

    def __init__(self, config: ZedCameraConfig):
        """
        Initializes the ZedCamera instance.

        Args:
            config: The configuration settings for the camera.
        """

        super().__init__(config)

        self.config = config

        if config.serial_number_or_name.isdigit():
            self.serial_number = config.serial_number_or_name
        else:
            self.serial_number = self._find_serial_number_from_name(config.serial_number_or_name)

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s
        self.depth_mode = config.depth_mode

        self.zed_camera: sl.Camera | None = None
        self.runtime_params: sl.RuntimeParameters | None = None
        self.mat_resolution: sl.Resolution | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

        # ZED specific attributes
        self.image_mat = sl.Mat()
        self.depth_mat = sl.Mat()
        self.point_cloud_mat = sl.Mat()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        """Checks if the ZED camera is opened and streaming."""
        return self.zed_camera is not None and self.zed_camera.is_opened()

    def connect(self, warmup: bool = True):
        """
        Connects to the ZED camera specified in the configuration.

        Initializes the ZED camera, configures the required parameters,
        starts the camera, and validates the actual stream settings.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ValueError: If the configuration is invalid (e.g., missing serial/name, name not unique).
            ConnectionError: If the camera is found but fails to open or no ZED devices are detected.
            RuntimeError: If the camera starts but fails to apply requested settings.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Create ZED camera object
        self.zed_camera = sl.Camera()

        # Set initialization parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  # Default, can be overridden
        init_params.camera_fps = self.fps or 30
        init_params.depth_mode = self._get_zed_depth_mode()
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.set_from_serial_number(int(self.serial_number))

        # Set depth minimum and maximum range in meters:cite[4]
        init_params.depth_minimum_distance = 0.2
        init_params.depth_maximum_distance = 1.5

        # Open the camera
        err = self.zed_camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.zed_camera = None
            raise ConnectionError(
                f"Failed to open {self}. Error code: {err}. "
                f"Run `lerobot-find-cameras zed` to find available cameras."
            )

        # Configure runtime parameters
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD

        # Set mat resolution based on configuration
        self._configure_mat_resolution()

        if warmup:
            # ZED cameras need longer warmup time:cite[4]
            time.sleep(self.warmup_s)
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    def _get_zed_depth_mode(self) -> sl.DEPTH_MODE:
        """Converts depth mode string to ZED depth mode enum."""
        if not self.use_depth:
            return sl.DEPTH_MODE.NONE

        mode_map = {
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
            "NEURAL": sl.DEPTH_MODE.NEURAL
        }
        return mode_map.get(self.depth_mode, sl.DEPTH_MODE.QUALITY)

    def _configure_mat_resolution(self) -> None:
        """Configures the matrix resolution based on camera settings."""
        if self.width and self.height:
            # Use custom resolution
            self.mat_resolution = sl.Resolution(self.capture_width, self.capture_height)
        else:
            # Use camera's default resolution
            camera_info = self.zed_camera.get_camera_information()
            self.mat_resolution = camera_info.camera_configuration.resolution
            self.width = self.mat_resolution.width
            self.height = self.mat_resolution.height
            self.capture_width = self.width
            self.capture_height = self.height

            # Update fps if not set
            if self.fps is None:
                self.fps = camera_info.camera_configuration.fps

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available ZED cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'id' (serial number), 'name',
            model, firmware version, and other available specs.

        Raises:
            OSError: If pyzed.sl is not installed.
            ImportError: If pyzed.sl is not installed.
        """
        found_cameras_info = []

        # Get list of connected devices
        device_list = sl.Camera.get_device_list()

        for device in device_list:
            resolution = device.camera_configuration.resolution
            camera_info = {
                "name": "ZED Camera",
                "type": "ZED",
                "id": str(device.serial_number),
                "model": "ZED 2i" if device.camera_model == sl.MODEL.ZED2i else str(device.camera_model),
                "firmware_version": str(device.firmware_version),
                "usb_type": str(device.usb_type),
                "state": str(device.state),
                "camera_configuration": {
                    "resolution": f"{resolution.width}x{resolution.height}",
                    "fps": device.camera_configuration.fps,
                }
            }
            found_cameras_info.append(camera_info)

        return found_cameras_info

    def _find_serial_number_from_name(self, name: str) -> str:
        """Finds the serial number for a given unique camera name."""
        camera_infos = self.find_cameras()
        found_devices = [cam for cam in camera_infos if str(cam["name"]) == name]

        if not found_devices:
            available_names = [cam["name"] for cam in camera_infos]
            raise ValueError(
                f"No ZED camera found with name '{name}'. Available camera names: {available_names}"
            )

        if len(found_devices) > 1:
            serial_numbers = [dev["id"] for dev in found_devices]
            raise ValueError(
                f"Multiple ZED cameras found with name '{name}'. "
                f"Please use a unique serial number instead. Found SNs: {serial_numbers}"
            )

        serial_number = str(found_devices[0]["id"])
        return serial_number

    def read_depth(self, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single frame (depth) synchronously from the camera.

        This is a blocking call. It waits for a depth frame from the ZED camera.

        Args:
            timeout_ms (int): Maximum time in milliseconds to wait for a frame. Defaults to 200ms.

        Returns:
            np.ndarray: The depth map as a NumPy array (height, width)
                  of type `np.uint16` (raw depth values in millimeters) with rotation applied.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the camera fails.
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.use_depth:
            raise RuntimeError(
                f"Failed to capture depth frame '.read_depth()'. Depth stream is not enabled for {self}."
            )

        start_time = time.perf_counter()

        # Grab a frame with timeout
        if not self.zed_camera.grab(self.runtime_params):
            raise RuntimeError(f"{self} read_depth failed to grab frame.")

        # Retrieve depth map
        self.zed_camera.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH, self.mat_resolution)
        depth_map = self.depth_mat.get_data()

        depth_map_processed = self._postprocess_image(depth_map, depth_frame=True)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read_depth took: {read_duration_ms:.1f}ms")

        return depth_map_processed

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single frame (color) synchronously from the camera.

        This is a blocking call. It waits for a color frame from the ZED camera.

        Args:
            timeout_ms (int): Maximum time in milliseconds to wait for a frame. Defaults to 200ms.

        Returns:
            np.ndarray: The captured color frame as a NumPy array
              (height, width, channels), processed according to `color_mode` and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the camera fails.
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        # Grab a frame with timeout handling
        if not self.zed_camera.grab(self.runtime_params):
            raise RuntimeError(f"{self} read failed to grab frame.")

        # Retrieve left image (RGB by default in ZED SDK)
        self.zed_camera.retrieve_image(self.image_mat, sl.VIEW.LEFT, self.mat_resolution)
        color_image_raw = self.image_mat.get_data()

        color_image_processed = self._postprocess_image(color_image_raw, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return color_image_processed

    def _postprocess_image(
            self, image: np.ndarray, color_mode: ColorMode | None = None, depth_frame: bool = False
    ) -> np.ndarray:
        """
        Applies color conversion, dimension validation, and rotation to a raw frame.

        Args:
            image (np.ndarray): The raw image frame from ZED camera.
            color_mode (Optional[ColorMode]): The target color mode (RGB or BGR). If None,
                                             uses the instance's default `self.color_mode`.
            depth_frame (bool): Whether this is a depth frame.

        Returns:
            np.ndarray: The processed image frame according to configuration.

        Raises:
            ValueError: If the requested `color_mode` is invalid.
            RuntimeError: If the raw frame dimensions do not match expectations.
        """

        if color_mode and color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if depth_frame:
            h, w = image.shape
        else:
            h, w, c = image.shape

            if c != 3:
                raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        processed_image = image

        # ZED returns images in BGR format by default, convert if needed
        if not depth_frame and self.color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a color frame
        2. Stores result in latest_frame (thread-safe)
        3. Sets new_frame_event to notify listeners

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        while not self.stop_event.is_set():
            try:
                color_image = self.read(timeout_ms=500)

                with self.frame_lock:
                    self.latest_frame = color_image
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

    def _stop_read_thread(self):
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """
        Reads the latest available frame data (color) asynchronously.

        This method retrieves the most recent color frame captured by the background
        read thread.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms.

        Returns:
            np.ndarray: The latest captured frame data (color image).

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If the background thread died unexpectedly.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

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

    def disconnect(self):
        """
        Disconnects from the camera and cleans up resources.

        Stops the background read thread and closes the ZED camera.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """

        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self.zed_camera is not None:
            self.zed_camera.close()
            self.zed_camera = None
            self.runtime_params = None
            self.mat_resolution = None

        logger.info(f"{self} disconnected.")
