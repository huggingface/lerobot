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
ZMQCamera - Captures frames from remote cameras via ZeroMQ using JSON protocol.

Protocol (unified across LeKiwi, MuJoCo sim, Unitree):
    {
        "timestamps": {"camera_name": float},
        "images": {"camera_name": "<base64-jpeg>"}
    }
"""

import base64
import json
import logging
import time
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
    Captures frames from remote cameras via ZeroMQ PUB/SUB sockets.

    All servers (LeKiwi, MuJoCo sim, Unitree) use the unified JSON protocol:
        {"timestamps": {...}, "images": {"camera_name": "<base64-jpeg>"}}

    Example:
        ```python
        from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig

        config = ZMQCameraConfig(
            server_address="192.168.123.164",
            port=5555,
            camera_name="head_camera"
        )
        camera = ZMQCamera(config)
        camera.connect()
        frame = camera.read()
        camera.disconnect()
        ```
    """

    def __init__(self, config: ZMQCameraConfig):
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
        return f"ZMQCamera({self.camera_name}@{self.server_address}:{self.port})"

    @property
    def is_connected(self) -> bool:
        return self._connected and self.context is not None and self.socket is not None

    def connect(self, warmup: bool = True) -> None:
        """Connect to ZMQ camera server."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        logger.info(f"Connecting to {self}...")

        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect(f"tcp://{self.server_address}:{self.port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self._connected = True

            # Validate connection with test frame
            test_frame = self.read()
            
            # Auto-detect resolution
            if self.width is None or self.height is None:
                h, w = test_frame.shape[:2]
                self.height = h
                self.width = w
                logger.info(f"{self} resolution: {w}x{h}")

            logger.info(f"{self} connected.")
            
            if warmup:
                time.sleep(0.1)

        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to connect to {self}: {e}")

    def _cleanup(self):
        """Clean up ZMQ resources."""
        self._connected = False
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None

    @staticmethod
    def find_cameras(
        subnet: str | None = None,
        ports: list[int] | None = None,
        timeout_ms: int = 200,
    ) -> list[dict[str, Any]]:
        """Scan network for ZMQ cameras (JSON protocol)."""
        import ipaddress
        import socket
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if ports is None:
            ports = [5554, 5555, 5556]

        # Auto-detect subnet
        if subnet is None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                subnet = ".".join(local_ip.split(".")[:-1]) + ".0/24"
            except Exception as e:
                logger.error(f"Failed to auto-detect subnet: {e}")
                return []

        try:
            network = ipaddress.ip_network(subnet, strict=False)
            hosts = [ipaddress.IPv4Address("127.0.0.1")] + list(network.hosts())
        except Exception as e:
            logger.error(f"Invalid subnet '{subnet}': {e}")
            return []

        def test_target(host_ip: str, port: int) -> dict | None:
            ctx = zmq.Context()
            sock = ctx.socket(zmq.SUB)
            sock.connect(f"tcp://{host_ip}:{port}")
            sock.setsockopt_string(zmq.SUBSCRIBE, "")
            sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
            time.sleep(0.1)

            msg = None
            for _ in range(3):
                try:
                    msg = sock.recv()
                    break
                except zmq.Again:
                    time.sleep(0.05)

            sock.close()
            ctx.term()

            if msg is None:
                return None

            # Try JSON decode
            try:
                data = json.loads(msg.decode("utf-8"))
                if isinstance(data, dict) and "images" in data:
                    cam_name = list(data["images"].keys())[0]
                    img_b64 = data["images"][cam_name]
                    img_bytes = base64.b64decode(img_b64)
                    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        h, w = frame.shape[:2]
                        return {
                            "name": f"ZMQ @ {host_ip}:{port}",
                            "type": "ZMQ",
                            "server_address": host_ip,
                            "port": port,
                            "camera_name": cam_name,
                            "resolution": f"{w}x{h}",
                        }
            except:
                pass

            return None

        found = []
        total = len(hosts) * len(ports)
        logger.info(f"Scanning {total} targets...")

        with ThreadPoolExecutor(max_workers=100) as ex:
            futures = [ex.submit(test_target, str(h), p) for h in hosts for p in ports]
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    found.append(res)
                    logger.info(f"  ✓ {res['server_address']}:{res['port']} ({res['camera_name']})")

        logger.info(f"Found {len(found)} camera(s).")
        return found

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        Read a single frame from the ZMQ camera.

        Returns:
            np.ndarray: Decoded frame (height, width, 3)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            message = self.socket.recv()
        except zmq.Again:
            raise TimeoutError(f"{self} timeout after {self.timeout_ms}ms")

        # Decode JSON message
        data = json.loads(message.decode("utf-8"))
        
        if "images" not in data:
            raise RuntimeError(f"{self} invalid message: missing 'images' key")

        images = data["images"]
        
        # Get image by camera name or first available
        if self.camera_name in images:
            img_b64 = images[self.camera_name]
        elif images:
            img_b64 = next(iter(images.values()))
        else:
            raise RuntimeError(f"{self} no images in message")

        # Decode base64 JPEG
        img_bytes = base64.b64decode(img_b64)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            raise RuntimeError(f"{self} failed to decode image")

        # Apply color conversion
        requested_mode = color_mode or self.color_mode
        if requested_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def _read_loop(self) -> None:
        """Background thread for async reading."""
        while self.stop_event and not self.stop_event.is_set():
            try:
                frame = self.read()
                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()
            except DeviceNotConnectedError:
                break
            except TimeoutError:
                pass
            except Exception as e:
                logger.warning(f"Read error: {e}")

    def _start_read_thread(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event:
            self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 10000) -> NDArray[Any]:
        """Read latest frame asynchronously (non-blocking)."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.thread or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"{self} async_read timeout after {timeout_ms}ms")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"{self} no frame available")

        return frame

    def disconnect(self) -> None:
        """Disconnect from ZMQ camera."""
        if not self.is_connected and not self.thread:
            raise DeviceNotConnectedError(f"{self} not connected.")

        self._stop_read_thread()
        self._cleanup()
        logger.info(f"{self} disconnected.")
