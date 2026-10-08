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

"""Wrapper around Pollen's Gripette gRPC client.

Exposes the gripper's two motors as ``proximal.pos`` / ``distal.pos`` (radians, Gripette robot
frame: 0 = open, positive = closing) and decodes the server's JPEG ``StreamState`` into an RGB
camera frame. A background daemon thread consumes the stream and caches the latest state + image.
"""

import logging
import threading
from typing import Any

import cv2
import numpy as np
from gripette.client import GripperClient

logger = logging.getLogger(__name__)

MOTORS = ("proximal", "distal")


class GripetteGripper:
    def __init__(
        self,
        addr: str,
        motor_max: tuple[float, float],
        camera_name: str | None,
        camera_shape: tuple[int, int],
    ):
        self.addr = addr
        self.motor_max = motor_max
        self.camera_name = camera_name
        self.camera_shape = camera_shape

        self._client: GripperClient | None = None
        self._stop = threading.Event()
        self._has_frame = threading.Event()
        self._lock = threading.Lock()
        self._latest_pos: tuple[float, float] = (0.0, 0.0)
        self._latest_image: np.ndarray | None = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    def connect(self) -> None:
        self._client = GripperClient(self.addr)
        self._client.connect()
        self._client.torque_on()
        self._stop.clear()
        self._has_frame.clear()
        threading.Thread(target=self._consume_stream, args=(self._client,), daemon=True).start()
        if not self._has_frame.wait(timeout=5.0):
            self.disconnect(disable_torque=True)
            raise TimeoutError(f"No frame received from Gripette at {self.addr} within 5s.")

    def _consume_stream(self, client: GripperClient) -> None:
        try:
            for frame in client.stream():
                if self._stop.is_set():
                    break
                self._store(frame)
        except Exception:
            if not self._stop.is_set():
                logger.exception("Gripette stream closed unexpectedly.")

    def _store(self, frame: Any) -> None:
        image = None
        if self.camera_name is not None and frame.jpeg_data:
            bgr = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                height, width = self.camera_shape
                if bgr.shape[:2] != (height, width):
                    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._latest_pos = (frame.motor1, frame.motor2)
            if image is not None:
                self._latest_image = image
        self._has_frame.set()

    def send_goal(self, goals: dict[str, float]) -> dict[str, float]:
        """Move both motors, clipped to [0, motor_max] since the server rejects out-of-range goals.

        A motor missing from ``goals`` keeps its last observed position.
        """
        assert self._client is not None
        with self._lock:
            current = dict(zip(MOTORS, self._latest_pos, strict=True))
        clipped = [
            min(max(goals.get(motor, current[motor]), 0.0), max_pos)
            for motor, max_pos in zip(MOTORS, self.motor_max, strict=True)
        ]
        self._client.move(clipped[0], clipped[1])
        return dict(zip(MOTORS, clipped, strict=True))

    def read_observation(self) -> dict[str, Any]:
        with self._lock:
            obs: dict[str, Any] = {f"{m}.pos": p for m, p in zip(MOTORS, self._latest_pos, strict=True)}
            if self.camera_name is not None and self._latest_image is not None:
                obs[self.camera_name] = self._latest_image
        return obs

    def disconnect(self, disable_torque: bool) -> None:
        self._stop.set()
        if self._client is not None:
            try:
                if disable_torque:
                    self._client.torque_off()
            except Exception:
                logger.exception("Failed to disable Gripette torque on disconnect.")
            self._client.close()
        self._client = None
