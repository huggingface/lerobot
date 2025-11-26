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
import threading
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any
import base64
import cv2
import numpy as np
import zmq
from numpy.typing import NDArray
import base64
import msgpack
import msgpack_numpy as m
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
        
        # Format type detected during connection (msgpack, json, or raw_jpeg)
        self._format_type: str | None = None

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

            # Try to receive one frame to validate connection and detect format
            try:
                # Try each format until one works
                test_frame = None
                for format_type in ["msgpack", "json", "raw_jpeg"]:
                    try:
                        test_frame = self.read(format=format_type)
                        self._format_type = format_type
                        logger.info(f"{self} detected format: {format_type}")
                        break
                    except Exception as e:
                        logger.debug(f"{self} format '{format_type}' failed: {e}")
                        continue
                
                if test_frame is None:
                    raise RuntimeError("Failed to decode frame with any supported format (msgpack, json, raw_jpeg)")
                
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
    def find_cameras(
        subnet: str | None = None,
        ports: list[int] | None = None,
        timeout_ms: int = 200,
    ) -> list[dict[str, Any]]:
        """
        Scans the local network for ZMQ cameras (fast parallel scan).

        Uses threading to scan multiple hosts simultaneously. Without parallelization,
        scanning 254 hosts would take 6+ minutes. With threads, takes ~10-15 seconds.

        Args:
            subnet: Network subnet to scan (e.g., "192.168.1.0/24"). If None, auto-detects.
            ports: List of ports to scan. Defaults to [5554, 5555, 5556].
            timeout_ms: Connection timeout per host in milliseconds. Default: 200ms.

        Returns:
            List of dicts containing camera info (address, port, format, resolution).
            
        Example:
            >>> cameras = ZMQCamera.find_cameras()
            >>> # Or specify: cameras = ZMQCamera.find_cameras(subnet="10.0.0.0/24", ports=[5554])
        """
        import socket
        import ipaddress
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if ports is None:
            ports = [5554, 5555, 5556]

        # Auto-detect local subnet
        if subnet is None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                subnet = ".".join(local_ip.split(".")[:-1]) + ".0/24"
                logger.info(f"Auto-detected subnet: {subnet}")
            except Exception as e:
                logger.error(f"Failed to auto-detect subnet: {e}")
                return []

        # Parse subnet
        try:
            network = ipaddress.ip_network(subnet, strict=False)
            hosts = list(network.hosts())
            # Always include localhost (for MuJoCo sim, local servers)
            hosts.insert(0, ipaddress.IPv4Address("127.0.0.1"))
        except Exception as e:
            logger.error(f"Invalid subnet '{subnet}': {e}")
            return []

        total = len(hosts) * len(ports)
        logger.info(f"Scanning {len(hosts)} hosts × {len(ports)} ports = {total} targets (this takes ~10-15s)...")
        
        def test_target(host_ip: str, port: int) -> dict | None:
            """Test one host:port for ZMQ camera."""
            ctx = zmq.Context()
            sock = ctx.socket(zmq.SUB)
            sock.connect(f"tcp://{host_ip}:{port}")
            sock.setsockopt_string(zmq.SUBSCRIBE, "")
            sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
            
            # Wait for subscription to establish (ZMQ "slow joiner" problem)
            time.sleep(0.1)
            
            # Try receiving a few times
            msg = None
            for _ in range(3):
                try:
                    msg = sock.recv()
                    break
                except zmq.Again:
                    time.sleep(0.05)
            
            if msg is None:
                sock.close()
                ctx.term()
                return None
            
            # Try formats: msgpack → json → raw_jpeg
            frame = fmt = None
            
            # Msgpack
            try:
                d = msgpack.unpackb(msg, object_hook=m.decode)
                if isinstance(d, dict) and "images" in d and len(d["images"]) > 0:
                    img = next(iter(d["images"].values()))
                    if isinstance(img, str):
                        frame = cv2.imdecode(np.frombuffer(base64.b64decode(img), np.uint8), cv2.IMREAD_COLOR)
                    elif isinstance(img, np.ndarray):
                        frame = img
                    if frame is not None:
                        fmt = "msgpack"
            except:
                pass
            
            # JSON
            if frame is None:
                try:
                    d = json.loads(msg.decode('utf-8'))
                    if isinstance(d, dict):
                        for v in d.values():
                            if isinstance(v, str) and len(v) > 100:
                                try:
                                    frame = cv2.imdecode(np.frombuffer(base64.b64decode(v), np.uint8), cv2.IMREAD_COLOR)
                                    if frame is not None:
                                        fmt = "json"
                                        break
                                except:
                                    pass
                except:
                    pass
            
            # Raw JPEG
            if frame is None:
                try:
                    frame = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        fmt = "raw_jpeg"
                except:
                    pass
            
            sock.close()
            ctx.term()
            
            if frame is not None:
                h, w = frame.shape[:2]
                return {
                    "name": f"ZMQ @ {host_ip}:{port}",
                    "type": "ZMQ",
                    "id": f"{host_ip}:{port}",
                    "server_address": host_ip,
                    "port": port,
                    "camera_name": f"cam_{host_ip.replace('.', '_')}_{port}",
                    "format": fmt,
                    "default_stream_profile": {"width": w, "height": h, "format": fmt.upper()},
                }
            return None

        # Parallel scan with thread pool
        found = []
        with ThreadPoolExecutor(max_workers=100) as ex:
            futures = [ex.submit(test_target, str(h), p) for h in hosts for p in ports]
            for i, fut in enumerate(as_completed(futures), 1):
                if i % 100 == 0:
                    logger.info(f"  Progress: {i}/{total} ({100*i//total}%)")
                res = fut.result()
                if res:
                    found.append(res)
                    logger.info(f"  ✓ {res['server_address']}:{res['port']} ({res['format']})")
        
        logger.info(f"Scan complete! Found {len(found)} camera(s).")
        return found

    def read(self, color_mode: ColorMode | None = None, format: str | None = None) -> NDArray[Any]:
        """
        Reads a single frame synchronously from the ZMQ camera.

        Supports three message formats:
        1. "msgpack": Msgpack with base64 JPEGs: {"timestamps": {...}, "images": {camera_name: "b64"}}
           (used by MuJoCo sim)
        2. "json": JSON with base64 JPEGs: {"state": 0.0, "camera_name": "b64jpeg"}
           (used by LeKiwi-style servers)
        3. "raw_jpeg": Raw JPEG bytes (used by Unitree G1 head camera)
        
        Args:
            color_mode: Target color mode (RGB or BGR). If None, uses self.color_mode.
            format: Message format to use. If None, uses auto-detected format from connect().
                   One of: "msgpack", "json", "raw_jpeg"
        
        Returns:
            np.ndarray: Decoded frame in shape (height, width, 3)
        
        Raises:
            DeviceNotConnectedError: If camera is not connected
            TimeoutError: If no frame received within timeout_ms
            RuntimeError: If frame decoding fails
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.socket is None:
            raise DeviceNotConnectedError(f"{self} socket is not initialized")

        # Use detected format if not specified
        if format is None:
            format = self._format_type
        
        if format is None:
            raise RuntimeError(f"{self} format not specified and not auto-detected during connect()")

        start_time = time.perf_counter()

        try:
            message = self.socket.recv()
        except zmq.Again:
            raise TimeoutError(f"{self} timeout waiting for frame after {self.timeout_ms}ms")
        except Exception as e:
            raise RuntimeError(f"{self} read failed: {e}")

        frame = None

        # Decode based on format
        if format == "msgpack":
            data = msgpack.unpackb(message, object_hook=m.decode)
            if not isinstance(data, dict) or "images" not in data:
                raise RuntimeError(f"{self} invalid msgpack format: expected dict with 'images' key")

            images_dict = data["images"]
            
            # Prefer named camera if present
            if self.camera_name in images_dict:
                img_data = images_dict[self.camera_name]
            elif len(images_dict) > 0:
                # Fallback: first available camera
                img_data = next(iter(images_dict.values()))
            else:
                raise RuntimeError(f"{self} no images found in msgpack message")

            # Decode the image data
            if isinstance(img_data, str):
                color_bytes = base64.b64decode(img_data)
                np_img = np.frombuffer(color_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            elif isinstance(img_data, np.ndarray):
                frame = img_data
            else:
                raise RuntimeError(f"{self} unknown image payload type: {type(img_data)}")
        
        elif format == "json":
            data = json.loads(message.decode('utf-8'))
            if not isinstance(data, dict) or self.camera_name not in data:
                raise RuntimeError(f"{self} invalid JSON format: expected dict with '{self.camera_name}' key")
            
            img_b64 = data[self.camera_name]
            if not isinstance(img_b64, str):
                raise RuntimeError(f"{self} expected base64 string in JSON, got {type(img_b64)}")
            
            color_bytes = base64.b64decode(img_b64)
            np_img = np.frombuffer(color_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        elif format == "raw_jpeg":
            np_img = np.frombuffer(message, dtype=np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        else:
            raise ValueError(f"{self} unsupported format: {format}. Use 'msgpack', 'json', or 'raw_jpeg'")

        if frame is None or not isinstance(frame, np.ndarray):
            raise RuntimeError(f"{self} failed to decode image using format '{format}'")

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

    def async_read(self, timeout_ms: float = 10000) -> NDArray[Any]:
        """
        Reads the latest available frame asynchronously.

        This method retrieves the most recent frame captured by the background
        read thread. It does not block waiting for ZMQ directly, but may wait
        up to timeout_ms for the background thread to provide a frame.

        Args:
            timeout_ms: Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 2000ms.

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

