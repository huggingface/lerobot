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

"""NexArm follower (slave) robot — LeRobot Robot subclass.

Mirrors the SO-100/SO-101 follower pattern: motors live in
``NexArmMotorsBus``, observations use ``sync_read("Present_Position")``,
and actions go through ``sync_write("Goal_Position", ...)`` with normalised
values (DEGREES / RANGE_0_100 by default). Calibration is performed via the
official ``set_half_turn_homings`` + ``record_ranges_of_motion`` flow.
"""

from __future__ import annotations

import logging
import time
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors.nexarm import NexArmMotorsBus
from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_nexarm_follower import NexArmFollowerConfig

logger = logging.getLogger(__name__)


class NexArmFollower(Robot):
    """NexArm 6-DOF desktop robot arm (follower / slave).

    The follower connects to the slave ESP32 via USB serial at 1 Mbps. The
    ESP32 forwards CMD 96/97/98 through an AT32F421 co-processor to the HX-30HM
    serial-bus servos. CMD 68 enables LeRobot bridge mode on connect.
    """

    config_class = NexArmFollowerConfig
    name = "nexarm_follower"

    def __init__(self, config: NexArmFollowerConfig):
        super().__init__(config)
        self.config = config
        self.bus = NexArmMotorsBus(
            port=config.port,
            calibration=self.calibration,
            baudrate=config.baudrate,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect(handshake=False)
        self.bus.enter_lerobot_mode()
        # Now that bridge mode is active, do a real handshake.
        self.bus.handshake()

        if calibrate and not self.is_calibrated:
            logger.info("No calibration found, running calibration")
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to keep the existing calibration for {self.id}, or type 'c' "
                "and press ENTER to run a new one: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Keeping existing calibration")
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.reset_calibration()
        self.bus.disable_torque()

        print(
            "Move all joints sequentially through their entire ranges of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )
        self.bus.record_ranges_of_motion()

        self.calibration = dict(self.bus.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.write_motion_params(acc=self.config.motion_acc, speed=self.config.motion_speed)
        self.bus.enable_torque()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        positions = self.bus.sync_read("Present_Position")
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        obs: RobotObservation = {f"{m}.pos": v for m, v in positions.items()}

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs[cam_key] = cam.read_latest(max_age_ms=1500)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {
            k.removesuffix(".pos"): float(v) for k, v in action.items() if k.endswith(".pos")
        }

        if self.config.max_relative_target is not None:
            present = self.bus.sync_read("Present_Position")
            goal_present = {
                name: (goal_pos[name], present[name]) for name in goal_pos if name in present
            }
            goal_pos = ensure_safe_goal_position(goal_present, self.config.max_relative_target)

        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{m}.pos": v for m, v in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:  # nosec B110
                pass

        if self.config.disable_torque_on_disconnect:
            try:
                # Hold current position briefly so servos don't drop on torque off.
                present = self.bus.sync_read("Present_Position")
                self.bus.sync_write("Goal_Position", present)
                time.sleep(0.4)
                self.bus.disable_torque()
            except Exception:  # nosec B110
                pass

        try:
            self.bus.exit_lerobot_mode()
        except Exception:  # nosec B110
            pass

        self.bus.disconnect(disable_torque=False)
        logger.info(f"{self} disconnected.")


__all__ = ["NexArmFollower", "JOINT_NAMES"]
