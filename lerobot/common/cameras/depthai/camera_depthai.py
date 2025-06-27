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
Provides the DepthAICamera class for capturing frames from Luxonis DepthAI cameras.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any, Dict, List

import cv2
import numpy as np

try:
    import depthai as dai
except Exception as e:
    logging.info(f"Could not import depthai: {e}")

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_depthai import DepthAICameraConfig

logger = logging.getLogger(__name__)


class DepthAICamera(Camera):
    """
    Manages interactions with Luxonis DepthAI cameras for RGB and depth recording.

    This class provides an interface similar to `OpenCVCamera` and `RealSenseCamera` but tailored for
    DepthAI devices, leveraging the `depthai` library. It uses the camera's unique MxID for
    identification, offering more stability than device indices. It supports capturing RGB frames
    from CAM_A and depth maps from stereo cameras CAM_B and CAM_C.

    Use the provided utility script to find available camera indices and default profiles:
    ```bash
    python -m lerobot.find_cameras depthai
    ```

    A `DepthAICamera` instance requires a configuration object specifying the
    camera's MxID or a unique device name. If using the name, ensure only
    one camera with that name is connected.

    The camera's default settings (FPS, resolution, color mode) from the stream
    profile are used unless overridden in the configuration.

    Example:
        ```python
        from lerobot.common.cameras.depthai import DepthAICamera, DepthAICameraConfig
        from lerobot.common.cameras import ColorMode, Cv2Rotation

        # RGB only with MxID
        config = DepthAICameraConfig(mxid_or_name="14442C10D13EABF200") # Replace with actual MxID
        camera = DepthAICamera(config)
        camera.connect()

        # Read 1 frame synchronously
        color_image = camera.read()
        print(color_image.shape)

        # Read 1 frame asynchronously
        async_image = camera.async_read()

        # When done, properly disconnect the camera using
        camera.disconnect()

        # Example with depth capture and custom settings
        custom_config = DepthAICameraConfig(
            mxid_or_name="14442C10D13EABF200", # Replace with actual MxID
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR, # Request BGR output
            rotation=Cv2Rotation.NO_ROTATION,
            enable_rgb=True,
            enable_depth=True
        )
        depth_camera = DepthAICamera(custom_config)
        depth_camera.connect()

        # Read RGB and depth frames
        color_image = depth_camera.read()
        depth_map = depth_camera.read_depth()

        # Example using a unique camera name for depth only
        depth_config = DepthAICameraConfig(
            mxid_or_name="OAK-D-LITE", # If unique
            enable_rgb=False,
            enable_depth=True
        )
        name_camera = DepthAICamera(name_config)
        # ... connect, read, disconnect ...
        ```
    """

    def __init__(self, config: DepthAICameraConfig):
        """
        Initializes the DepthAICamera instance.

        Args:
            config: The configuration settings for the camera.
        """

        super().__init__(config)

        self.config = config

        # Check if mxid_or_name is an MxID (hex string) or a name
        if len(config.mxid_or_name) == 18 and all(
            c in "0123456789ABCDEF" for c in config.mxid_or_name.upper()
        ):
            self.mxid = config.mxid_or_name
        else:
            self.mxid = self._find_mxid_from_name(config.mxid_or_name)

        # Set defaults and extract configuration
        self.fps = config.fps or 30
        self.width = config.width or 640
        self.height = config.height or 480
        self.color_mode = config.color_mode
        self.enable_rgb = config.enable_rgb
        self.enable_depth = config.enable_depth
        self.warmup_s = config.warmup_s

        self.device: dai.Device | None = None
        self.pipeline: dai.Pipeline | None = None
        self.q_rgb: dai.DataOutputQueue | None = None
        self.q_depth: dai.DataOutputQueue | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.latest_depth_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()
        self.new_depth_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        # Calculate capture dimensions considering rotation
        self.capture_width, self.capture_height = self.width, self.height
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.mxid})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera device is connected."""
        return self.device is not None

    @property
    def streams_available(self) -> Dict[str, bool]:
        """
        Returns which streams are enabled and available.

        Returns:
            Dict with 'rgb' and 'depth' keys indicating availability.
        """
        return {
            "rgb": self.enable_rgb and self.q_rgb is not None,
            "depth": self.enable_depth and self.q_depth is not None,
        }

    def connect(self, warmup: bool = True):
        """
        Connects to the DepthAI camera and starts the pipeline.

        Args:
            warmup: Whether to warm up the camera by reading a few frames.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ConnectionError: If the camera fails to connect or start.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        try:
            # Create pipeline using DepthAI v3 API
            self.pipeline = dai.Pipeline()

            # Configure RGB camera (CAM_A) if enabled
            if self.enable_rgb:
                cam_rgb = self.pipeline.create(dai.node.Camera).build()
                # Request output with desired resolution
                self.q_rgb = cam_rgb.requestOutput((self.width, self.height)).createOutputQueue()

            # Configure depth cameras (CAM_B and CAM_C) if enabled
            if self.enable_depth:
                # Create mono cameras using v3 API
                cam_left = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
                cam_right = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

                # Create stereo depth node
                stereo = self.pipeline.create(dai.node.StereoDepth)

                # Link mono camera outputs to stereo inputs
                mono_left_out = cam_left.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)
                mono_right_out = cam_right.requestFullResolutionOutput(type=dai.ImgFrame.Type.NV12)

                mono_left_out.link(stereo.left)
                mono_right_out.link(stereo.right)

                # Configure stereo depth
                stereo.setRectification(True)
                stereo.setExtendedDisparity(True)
                stereo.setLeftRightCheck(True)

                # Create depth output queue
                self.q_depth = stereo.disparity.createOutputQueue()

            # Start pipeline
            self.pipeline.start()

            # Store pipeline as device reference for v3 compatibility
            self.device = self.pipeline

        except Exception as e:
            self.device = None
            self.pipeline = None
            self.q_rgb = None
            self.q_depth = None
            raise ConnectionError(f"Failed to open {self}. Error: {str(e)}") from e

        if warmup:
            time.sleep(1)  # DepthAI cameras need time to warm up
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    if self.enable_rgb:
                        self.read()
                except Exception as e:
                    # Ignore warmup failures - camera may not be ready yet
                    logger.debug(f"Warmup read failed (expected during camera initialization): {e}")
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        """
        Detects available DepthAI cameras connected to the system.

        Returns:
            List of camera info dictionaries with id, name, type, and default profile.
        """
        found_cameras_info = []

        try:
            # Use the correct DepthAI v3 API for device discovery
            devices = dai.DeviceBase.getAllAvailableDevices()

            for device_info in devices:
                # In DepthAI v3, device info attributes are direct properties
                camera_info = {
                    "name": getattr(device_info, "name", "Unknown DepthAI Camera"),
                    "type": "DepthAI",
                    "id": device_info.deviceId,  # Use deviceId instead of getMxId()
                    "state": getattr(device_info, "state", "Unknown").name
                    if hasattr(getattr(device_info, "state", None), "name")
                    else str(getattr(device_info, "state", "Unknown")),
                    "protocol": getattr(device_info, "protocol", "Unknown").name
                    if hasattr(getattr(device_info, "protocol", None), "name")
                    else str(getattr(device_info, "protocol", "Unknown")),
                    "platform": getattr(device_info, "platform", "Unknown").name
                    if hasattr(getattr(device_info, "platform", None), "name")
                    else str(getattr(device_info, "platform", "Unknown")),
                    "product_name": getattr(device_info, "productName", "Unknown"),
                    "board_name": getattr(device_info, "boardName", "Unknown"),
                    "default_stream_profile": {
                        "format": "RGB888",
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                    },
                }
                found_cameras_info.append(camera_info)

        except Exception as e:
            logger.warning(f"Error finding DepthAI cameras: {e}")

        return found_cameras_info

    def _find_mxid_from_name(self, name: str) -> str:
        """Finds the MxID for a given unique camera name."""
        camera_infos = self.find_cameras()
        found_devices = [cam for cam in camera_infos if str(cam["name"]) == name]

        if not found_devices:
            available_names = [cam["name"] for cam in camera_infos]
            raise ValueError(
                f"No DepthAI camera found with name '{name}'. Available camera names: {available_names}"
            )

        if len(found_devices) > 1:
            mxids = [dev["id"] for dev in found_devices]
            raise ValueError(
                f"Multiple DepthAI cameras found with name '{name}'. "
                f"Please use a unique MxID instead. Found MxIDs: {mxids}"
            )

        mxid = str(found_devices[0]["id"])
        return mxid

    def read_depth(self, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single depth frame synchronously from the stereo cameras.

        Args:
            timeout_ms: Maximum time in milliseconds to wait for a frame.

        Returns:
            np.ndarray: The captured depth map as a NumPy array (height, width) with dtype uint16.
                       Values represent depth in millimeters.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If depth is not enabled or reading frames fails.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.enable_depth:
            raise RuntimeError(f"{self} depth stream is not enabled. Set enable_depth=True in config.")

        try:
            # Get depth frame from queue
            depth_data = self.q_depth.get()
            if depth_data is None:
                raise RuntimeError("No depth frame available from camera")

            # Get depth frame as NumPy array
            depth_frame = depth_data.getFrame()
            if depth_frame is None or depth_frame.size == 0:
                raise RuntimeError("Received empty depth frame")

            # Apply rotation if configured
            if self.rotation is not None:
                depth_frame = cv2.rotate(depth_frame, self.rotation)

            return depth_frame

        except Exception as e:
            raise RuntimeError(f"{self} depth read failed: {e}") from e

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single RGB frame synchronously from the camera.

        Args:
            color_mode: The color mode (RGB or BGR). If None, uses the default from config.
            timeout_ms: Maximum time in milliseconds to wait for a frame.

        Returns:
            np.ndarray: The captured RGB frame as a NumPy array (height, width, channels).

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If RGB is not enabled or reading frames fails.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.enable_rgb:
            raise RuntimeError(f"{self} RGB stream is not enabled. Set enable_rgb=True in config.")

        try:
            # Get frame from RGB queue
            frame_data = self.q_rgb.get()
            if frame_data is None:
                raise RuntimeError("No RGB frame available from camera")

            # Get OpenCV frame (RGB format from DepthAI v3)
            color_frame = frame_data.getCvFrame()
            if color_frame is None or color_frame.size == 0:
                raise RuntimeError("Received empty RGB frame")

            # Process the frame (handle color conversion and rotation)
            return self._postprocess_image(color_frame, color_mode)

        except Exception as e:
            raise RuntimeError(f"{self} RGB read failed: {e}") from e

    def _postprocess_image(
        self, image: np.ndarray, color_mode: ColorMode | None = None, depth_frame: bool = False
    ) -> np.ndarray:
        """
        Applies color conversion and rotation to a raw frame.

        Args:
            image: The raw image frame (BGR format from DepthAI v3 camera - default OpenCV format).
            color_mode: The target color mode (RGB or BGR). If None, uses instance default.
            depth_frame: Whether this is a depth frame (unused, for future compatibility).

        Returns:
            np.ndarray: The processed image frame.
        """
        # Handle color conversion for color frames
        if not depth_frame:
            target_mode = color_mode or self.color_mode
            if target_mode == ColorMode.RGB:
                # Convert from BGR (DepthAI v3 default) to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # For BGR mode, keep as-is since DepthAI provides BGR

        # Apply rotation if needed
        if self.rotation and self.rotation != 0:
            image = cv2.rotate(image, self.rotation)

        return image

    def _read_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a color frame with 500ms timeout
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
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray:
            The latest captured frame data (color image), processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If the background thread died unexpectedly or another error occurs.
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

    def async_read_depth(self, timeout_ms: float = 200) -> np.ndarray:
        """
        Reads the latest available depth frame asynchronously.

        This method retrieves the most recent depth frame from the stereo cameras.
        Unlike async_read which uses a background thread, this method directly
        accesses the depth queue for the latest depth data.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a depth frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray: The latest captured depth frame as a NumPy array (height, width)
                       with dtype uint16. Values represent disparity.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If depth stream is not enabled.
            TimeoutError: If no depth frame becomes available within the specified timeout.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.enable_depth:
            raise RuntimeError(f"{self} depth stream is not enabled. Set enable_depth=True in config.")

        try:
            # Get depth frame from queue with timeout
            depth_data = self.q_depth.get()
            if depth_data is None:
                raise TimeoutError(f"Timed out waiting for depth frame from {self} after {timeout_ms} ms.")

            # Get depth frame as NumPy array
            depth_frame = depth_data.getFrame()
            if depth_frame is None or depth_frame.size == 0:
                raise RuntimeError("Received empty depth frame")

            # Apply rotation if configured
            if self.rotation is not None:
                depth_frame = cv2.rotate(depth_frame, self.rotation)

            return depth_frame

        except Exception as e:
            if "Timed out" in str(e):
                raise
            raise RuntimeError(f"{self} async depth read failed: {e}") from e

    def disconnect(self):
        """
        Disconnects from the camera, closes the device, and cleans up resources.

        Stops the background read thread (if running) and closes the DepthAI device.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected (device not connected).
        """

        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self.device is not None:
            # In v3 API, pipeline.stop() handles cleanup
            if hasattr(self.device, "stop"):
                self.device.stop()
            self.device = None
            self.pipeline = None
            self.q_rgb = None
            self.q_depth = None

        logger.info(f"{self} disconnected.")
