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

"""Unitree Go2 quadruped (EDU) — high-level sport-mode integration.

Unlike :class:`~lerobot.robots.unitree_g1.UnitreeG1` (low-level joint
control at 250 Hz through an on-robot ZMQ bridge), the Go2 is driven with
sport-mode **body velocity commands** at tens of Hz, which work fine over
plain DDS from any Linux host on the dog's network — no bridge server, no
onboard companion computer.

Setup:
  1. Connect the host to the Go2 via Ethernet (dog is on 192.168.123.x)
     or put both on the same WiFi network.
  2. ``pip install unitree_sdk2py`` (Linux only — rides on cyclonedds).
  3. Find your interface name (``ip link``), then e.g.::

       lerobot-teleoperate \
           --robot.type=unitree_go2 \
           --robot.network_interface=enp2s0 \
           --teleop.type=gamepad

Actions are body-frame velocities ``x.vel`` (forward, m/s), ``y.vel``
(left, m/s), ``theta.vel`` (CCW yaw, rad/s) — the exact arguments of the
SDK's ``SportClient.Move``. Observations are planar sport-mode odometry
(``*.pos`` pose + ``*.vel`` body velocities) and the built-in front
camera, plus any extra configured cameras.
"""

from __future__ import annotations

import logging
import threading
from functools import cached_property
from typing import Any

import cv2
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.import_utils import require_package

from ..robot import Robot
from .config_unitree_go2 import UnitreeGo2Config

logger = logging.getLogger(__name__)

# DDS topic names follow Unitree SDK naming conventions
SPORT_MODE_STATE_TOPIC = "rt/sportmodestate"


class UnitreeGo2(Robot):
    """LeRobot interface to a Unitree Go2 over unitree_sdk2py sport mode."""

    config_class = UnitreeGo2Config
    name = "unitree_go2"

    def __init__(self, config: UnitreeGo2Config):
        super().__init__(config)
        self.config = config

        self._cameras = make_cameras_from_configs(config.cameras)

        # SDK handles — populated in connect(); the SDK import lives there
        # too so that configs, features and tests work on SDK-less hosts.
        self._sport = None
        self._video = None
        self._state_subscriber = None

        self._state_lock = threading.Lock()
        self._latest_state = None  # last SportModeState_ message
        self._connected = False

    # ------------------------------------------------------------------ #
    # Features
    # ------------------------------------------------------------------ #

    @cached_property
    def _odom_ft(self) -> dict[str, type]:
        return {
            "x.pos": float,
            "y.pos": float,
            "theta.pos": float,
            "x.vel": float,
            "y.vel": float,
            "theta.vel": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        ft: dict[str, tuple] = {}
        if self.config.use_front_camera:
            ft["front"] = (self.config.front_camera_height, self.config.front_camera_width, 3)
        for name, cam in self._cameras.items():
            ft[name] = (cam.height, cam.width, 3)
        return ft

    @property
    def observation_features(self) -> dict:
        return {**self._odom_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return {"x.vel": float, "y.vel": float, "theta.vel": float}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return
        require_package("unitree-sdk2py", extra="unitree_go2", import_name="unitree_sdk2py")

        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

        ChannelFactoryInitialize(self.config.domain_id, self.config.network_interface)

        sport = SportClient()
        sport.SetTimeout(5.0)
        sport.Init()
        self._sport = sport

        subscriber = ChannelSubscriber(SPORT_MODE_STATE_TOPIC, SportModeState_)
        subscriber.Init(self._on_sport_state, 10)
        self._state_subscriber = subscriber

        if self.config.use_front_camera:
            from unitree_sdk2py.go2.video.video_client import VideoClient

            video = VideoClient()
            video.SetTimeout(3.0)
            video.Init()
            self._video = video

        for cam in self._cameras.values():
            cam.connect()

        if self.config.stand_on_connect:
            self._sport.BalanceStand()

        self._connected = True
        self.configure()
        logger.info(
            "%s connected (iface=%s, domain=%d)",
            self,
            self.config.network_interface,
            self.config.domain_id,
        )

    def disconnect(self) -> None:
        if self._sport is not None:
            try:
                self._sport.StopMove()
            except Exception:
                logger.exception("StopMove on disconnect failed")
        for cam in self._cameras.values():
            try:
                cam.disconnect()
            except Exception:
                logger.exception("camera disconnect failed")
        self._sport = None
        self._video = None
        self._state_subscriber = None
        self._connected = False

    # Sport mode needs no calibration.
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #

    def get_observation(self) -> RobotObservation:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected.")

        obs: dict[str, Any] = dict.fromkeys(self._odom_ft, 0.0)
        with self._state_lock:
            state = self._latest_state
        if state is not None:
            obs["x.pos"] = float(state.position[0])
            obs["y.pos"] = float(state.position[1])
            obs["theta.pos"] = float(state.imu_state.rpy[2])
            obs["x.vel"] = float(state.velocity[0])
            obs["y.vel"] = float(state.velocity[1])
            obs["theta.vel"] = float(state.yaw_speed)

        if self.config.use_front_camera:
            obs["front"] = self._read_front_camera()

        for name, cam in self._cameras.items():
            obs[name] = cam.async_read()

        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected.")

        vx = float(np.clip(action.get("x.vel", 0.0), -self.config.max_x_vel, self.config.max_x_vel))
        vy = float(np.clip(action.get("y.vel", 0.0), -self.config.max_y_vel, self.config.max_y_vel))
        vyaw = float(
            np.clip(action.get("theta.vel", 0.0), -self.config.max_theta_vel, self.config.max_theta_vel)
        )

        self._sport.Move(vx, vy, vyaw)
        return {"x.vel": vx, "y.vel": vy, "theta.vel": vyaw}

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _on_sport_state(self, msg) -> None:
        with self._state_lock:
            self._latest_state = msg

    def _read_front_camera(self) -> np.ndarray:
        """Fetch one frame from the built-in front camera (RGB, HxWx3)."""
        h, w = self.config.front_camera_height, self.config.front_camera_width
        code, data = self._video.GetImageSample()
        if code != 0 or data is None:
            logger.warning("front camera GetImageSample failed (code=%s)", code)
            return np.zeros((h, w, 3), dtype=np.uint8)
        frame = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("front camera frame failed to decode")
            return np.zeros((h, w, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return frame
