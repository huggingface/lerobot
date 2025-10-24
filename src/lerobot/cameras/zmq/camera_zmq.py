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
Provides the ZMQCamera class for capturing frames from remote cameras via ZeroMQ.
"""

import json
import logging
import os
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
import zmq
from numpy.typing import NDArray

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from .configuration_zmq import ZMQCameraConfig

logger = logging.getLogger(__name__)


class ZMQCamera(Camera):
    """
    Manages camera interactions using ZeroMQ for remote frame streaming.

    This class provides a high-level interface to connect to remote cameras
    that stream JPEG-encoded images over ZeroMQ PUB/SUB sockets. It supports
    both synchronous and asynchronous frame reading.

    The camera server must be running and publishing JPEG images on the specified
    address and port. Use the provided utility script to find available ZMQ cameras:
    ```bash
    lerobot-find-cameras zmq
    ```

    Example:
        ```python
        from lerobot.cameras.zmq import ZMQCamera
        from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig, ColorMode

        # Basic usage
        config = ZMQCameraConfig(
            server_address="192.168.123.164",
            port=5554,
            camera_name="remote_cam"
        )
        camera = ZMQCamera(config)
        camera.connect()

        # Read 1 frame synchronously
        color_image = camera.read()
        print(color_image.shape)

        # Read 1 frame asynchronously
        async_image = camera.async_read()

        # When done, properly disconnect the camera
        camera.disconnect()
        ```
    """

    def __init__(self, config: ZMQCameraConfig):
        """
        Initializes the ZMQCamera instance.

        Args:
            config: The configuration settings for the ZMQ camera.
        """
        super().__init__(config)

        self.config = config
        self.server_address = config.server_address
        self.port = config.port
        self.camera_name = config.camera_name
        self.color_mode = config.color_mode
        self.timeout_ms = config.timeout_ms

        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self._connected = False

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.new_frame_event: Event = Event()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.camera_name}@{self.server_address}:{self.port})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected."""
        return self._connected and self.context is not None and self.socket is not None

    def connect(self, warmup: bool = True) -> None:
        """
        Connects to the ZMQ camera server and configures settings.

        Args:
            warmup: If True (default), captures a warmup frame before returning.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            RuntimeError: If connection to the ZMQ server fails.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        logger.info(f"Connecting to {self}...")

        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect(f"tcp://{self.server_address}:{self.port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            
            # Set receive timeout
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)

            self._connected = True

            # Try to receive one frame to validate connection
            try:
                test_frame = self.read()
                
                # Auto-detect resolution if not specified
                if self.width is None or self.height is None:
                    h, w = test_frame.shape[:2]
                    self.height = h
                    self.width = w
                    logger.info(f"{self} auto-detected resolution: {w}x{h}")

                logger.info(f"{self} connected successfully.")
                
                if warmup:
                    logger.debug(f"Warming up {self}...")
                    time.sleep(0.1)  # Brief warmup period
                    
            except Exception as e:
                self._connected = False
                if self.socket:
                    self.socket.close()
                if self.context:
                    self.context.term()
                self.socket = None
                self.context = None
                raise RuntimeError(f"Failed to receive initial frame from {self}: {e}")

        except Exception as e:
            self._connected = False
            if self.socket:
                self.socket.close()
            if self.context:
                self.context.term()
            self.socket = None
            self.context = None
            raise RuntimeError(f"Failed to connect to {self}: {e}")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available ZMQ cameras based on configuration.

        Reads camera configurations from:
        1. Environment variable LEROBOT_ZMQ_CAMERAS (JSON format)
        2. Config file at ~/.lerobot/zmq_cameras.json

        Example JSON format:
        ```json
        [
            {
                "name": "unitree_g1_head",
                "address": "192.168.123.164",
                "port": 5554
            },
            {
                "name": "lab_cam_1",
                "address": "192.168.1.100",
                "port": 5555
            }
        ]
        ```

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing ZMQ camera information.
        """
        found_cameras_info = []
        camera_configs = []

        # Try to load from environment variable first
        env_cameras = os.environ.get("LEROBOT_ZMQ_CAMERAS")
        if env_cameras:
            try:
                camera_configs = json.loads(env_cameras)
                logger.info(f"Loaded {len(camera_configs)} ZMQ camera configs from LEROBOT_ZMQ_CAMERAS")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LEROBOT_ZMQ_CAMERAS environment variable: {e}")
        #use unitree_g1_head as an example
        camera_configs = [
            {
                "name": "unitree_g1_head",
                "address": "192.168.123.164",
                "port": 5554
            }
        ]
        # Try to load from config file
        if not camera_configs:
            config_path = Path.home() / ".lerobot" / "zmq_cameras.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        camera_configs = json.load(f)
                    logger.info(f"Loaded {len(camera_configs)} ZMQ camera configs from {config_path}")
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load ZMQ camera config from {config_path}: {e}")

        if not camera_configs:
            logger.info(
                "No ZMQ cameras configured. Set LEROBOT_ZMQ_CAMERAS environment variable "
                f"or create {Path.home() / '.lerobot' / 'zmq_cameras.json'}"
            )
            return []

        # Test each configured camera
        for cam_config in camera_configs:
            try:
                name = cam_config.get("name", "unknown")
                address = cam_config.get("address")
                port = cam_config.get("port", 5554)

                if not address:
                    logger.warning(f"Skipping camera '{name}': missing address")
                    continue

                # Try to connect with a short timeout
                context = zmq.Context()
                socket = context.socket(zmq.SUB)
                socket.connect(f"tcp://{address}:{port}")
                socket.setsockopt_string(zmq.SUBSCRIBE, "")
                socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout for discovery

                try:
                    # Try to receive one frame to validate
                    message = socket.recv()
                    np_img = np.frombuffer(message, dtype=np.uint8)
                    test_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                    if test_image is not None:
                        height, width = test_image.shape[:2]
                        
                        camera_info = {
                            "name": f"ZMQ Camera: {name}",
                            "type": "ZMQ",
                            "id": f"{address}:{port}",
                            "server_address": address,
                            "port": port,
                            "camera_name": name,
                            "default_stream_profile": {
                                "width": width,
                                "height": height,
                                "format": "JPEG",
                            },
                        }
                        found_cameras_info.append(camera_info)
                        logger.info(f"Found ZMQ camera: {name} at {address}:{port}")
                    else:
                        logger.warning(f"Camera '{name}' at {address}:{port} returned invalid image")

                except zmq.Again:
                    logger.warning(f"Camera '{name}' at {address}:{port} timeout - not streaming")
                except Exception as e:
                    logger.warning(f"Error testing camera '{name}' at {address}:{port}: {e}")
                finally:
                    socket.close()
                    context.term()

            except Exception as e:
                logger.warning(f"Error processing camera config: {e}")

        return found_cameras_info

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        Reads a single frame synchronously from the ZMQ camera.

        This is a blocking call. It waits for the next available frame from the
        ZMQ socket.

        Args:
            color_mode: If specified, overrides the default color mode for this read.

        Returns:
            np.ndarray: The captured frame as a NumPy array in the format
                       (height, width, channels), using the specified or default
                       color mode.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame is received within the timeout period.
            RuntimeError: If reading the frame fails.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        if self.socket is None:
            raise DeviceNotConnectedError(f"{self} socket is not initialized")

        try:
            message = self.socket.recv()
        except zmq.Again:
            raise TimeoutError(f"{self} timeout waiting for frame after {self.timeout_ms}ms")
        except Exception as e:
            raise RuntimeError(f"{self} read failed: {e}")

        # Decode JPEG
        np_img = np.frombuffer(message, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            raise RuntimeError(f"{self} failed to decode image")

        processed_frame = self._postprocess_image(frame, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return processed_frame

    def _postprocess_image(self, image: NDArray[Any], color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        Applies color conversion to a raw frame.

        Args:
            image: The raw image frame (BGR format from cv2.imdecode).
            color_mode: The target color mode (RGB or BGR). If None, uses self.color_mode.

        Returns:
            np.ndarray: The processed image frame.

        Raises:
            ValueError: If the requested color_mode is invalid.
            RuntimeError: If the frame dimensions don't match expectations.
        """
        requested_color_mode = self.color_mode if color_mode is None else color_mode

        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape

        # Validate dimensions if they were specified
        if self.height is not None and self.width is not None:
            if h != self.height or w != self.width:
                logger.warning(
                    f"{self} frame dimensions ({w}x{h}) don't match configured ({self.width}x{self.height}). "
                    "This might be expected if the server sends different resolutions."
                )

        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        processed_image = image
        if requested_color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return processed_image

    def _read_loop(self) -> None:
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a frame from ZMQ
        2. Stores result in latest_frame (thread-safe)
        3. Sets new_frame_event to notify listeners

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        while not self.stop_event.is_set():
            try:
                frame = self.read()

                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except TimeoutError:
                # Timeout is expected occasionally, just continue
                logger.debug(f"{self} read timeout in background thread")
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

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        Reads the latest available frame asynchronously.

        This method retrieves the most recent frame captured by the background
        read thread. It does not block waiting for ZMQ directly, but may wait
        up to timeout_ms for the background thread to provide a frame.

        Args:
            timeout_ms: Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms.

        Returns:
            np.ndarray: The latest captured frame as a NumPy array in the format
                       (height, width, channels), processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame becomes available within the specified timeout.
            RuntimeError: If an unexpected error occurs.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def disconnect(self) -> None:
        """
        Disconnects from the ZMQ camera and cleans up resources.

        Stops the background read thread (if running) and closes the ZMQ socket.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.socket is not None:
            self.socket.close()
            self.socket = None

        if self.context is not None:
            self.context.term()
            self.context = None

        self._connected = False

        logger.info(f"{self} disconnected.")

