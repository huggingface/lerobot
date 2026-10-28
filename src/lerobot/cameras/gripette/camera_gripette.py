#!/usr/bin/env python

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

"""Gripette camera: JPEG frames from Pollen's Gripette gRPC ``StreamState``."""

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _gripette_available, require_package

from ..camera import Camera
from ..configs import ColorMode
from .configuration_gripette import GripetteCameraConfig

if TYPE_CHECKING or _gripette_available:
    from gripette.client import GripperClient
else:
    GripperClient: Any = None

logger = logging.getLogger(__name__)


class GripetteCamera(Camera):
    """JPEG frames from a Pollen Gripette gRPC server."""

    def __init__(self, config: GripetteCameraConfig):
        require_package("gripette", extra="gripette")
        super().__init__(config)
        self.config = config
        self.addr = config.addr
        self.color_mode = config.color_mode

        self._client: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._has_frame = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: NDArray[Any] | None = None
        self._latest_timestamp: float | None = None

    def __str__(self) -> str:
        return f"GripetteCamera({self.addr})"

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        raise NotImplementedError(
            "Gripette cameras are networked and must be configured with their gRPC address."
        )

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        logger.info(f"Connecting to {self}...")
        client = GripperClient(self.addr)
        client.connect()
        self._client = client
        self._stop.clear()
        self._has_frame.clear()
        self._thread = threading.Thread(target=self._consume, args=(client,), daemon=True)
        self._thread.start()
        if warmup and not self._has_frame.wait(timeout=5.0):
            self._shutdown()
            raise TimeoutError(f"No frame received from Gripette at {self.addr} within 5s.")
        logger.info(f"{self} connected.")

    def _consume(self, client: Any) -> None:
        try:
            for frame in client.stream():
                if self._stop.is_set():
                    break
                self._store(frame)
        except Exception:
            if not self._stop.is_set():
                logger.exception("Gripette camera stream closed unexpectedly.")

    def _store(self, frame: Any) -> None:
        image = None
        if frame.jpeg_data:
            bgr = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                width, height = self.width, self.height
                if width is not None and height is not None and bgr.shape[:2] != (height, width):
                    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if self.color_mode == ColorMode.RGB else bgr
        if image is None:
            return
        with self._lock:
            self._latest_frame = image
            self._latest_timestamp = time.perf_counter()
        self._has_frame.set()

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        if color_mode is not None:
            logger.warning(f"{self} read() color_mode is ignored; set it on GripetteCameraConfig.")
        self._has_frame.clear()
        return self.async_read(timeout_ms=5000)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        if not self._has_frame.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"{self} async_read timeout after {timeout_ms}ms")
        with self._lock:
            frame = self._latest_frame
            self._has_frame.clear()
        if frame is None:
            raise RuntimeError(f"{self} no frame available")
        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        with self._lock:
            frame = self._latest_frame
            timestamp = self._latest_timestamp
        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )
        return frame

    @check_if_not_connected
    def disconnect(self) -> None:
        self._shutdown()
        logger.info(f"{self} disconnected.")

    def _shutdown(self) -> None:
        self._stop.set()
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.exception("Failed to close the Gripette camera client.")
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
