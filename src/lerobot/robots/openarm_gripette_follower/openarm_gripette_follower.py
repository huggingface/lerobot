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

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _gripette_available, require_package

from ..openarm_follower import OpenArmFollower
from .config_openarm_gripette_follower import OpenArmGripetteFollowerConfig

if TYPE_CHECKING or _gripette_available:
    from gripette.client import GripperClient
else:
    GripperClient: Any = None

logger = logging.getLogger(__name__)

_GRIPPER_MOTORS = ("proximal", "distal")
_GRIPPER_KEYS = {f"{motor}.pos" for motor in _GRIPPER_MOTORS}


class OpenArmGripetteFollower(OpenArmFollower):
    """7-joint OpenArm (Damiao/CAN) arm with a networked Gripette gripper (gRPC) instead of the CAN gripper."""

    config_class = OpenArmGripetteFollowerConfig
    name = "openarm_gripette_follower"

    def __init__(self, config: OpenArmGripetteFollowerConfig):
        require_package("gripette", extra="gripette")
        super().__init__(config)
        self.config: OpenArmGripetteFollowerConfig = config
        self._gripper_client: Any = None
        self._gripper_pos: tuple[float, float] = (0.0, 0.0)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._motors_ft, **dict.fromkeys(_GRIPPER_KEYS, float)}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **dict.fromkeys(_GRIPPER_KEYS, float), **self._cameras_ft}

    @property
    def is_connected(self) -> bool:
        return super().is_connected and self._gripper_client is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        self._gripper_client = GripperClient(self.config.gripette_addr)
        self._gripper_client.connect()
        self._gripper_client.torque_on()
        motor1, motor2 = self._gripper_client.read_motors()
        self._gripper_pos = (float(motor1), float(motor2))

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs = super().get_observation()
        motor1, motor2 = self._gripper_client.read_motors()
        self._gripper_pos = (float(motor1), float(motor2))
        obs["proximal.pos"], obs["distal.pos"] = self._gripper_pos
        return obs

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        arm_action = {k: v for k, v in action.items() if k not in _GRIPPER_KEYS}
        sent = super().send_action(arm_action, custom_kp=custom_kp, custom_kd=custom_kd)
        if not _GRIPPER_KEYS & action.keys():
            return sent

        # A missing motor keeps its last position; goals are clipped to [0, gripper_motor_max].
        motor_max = self.config.gripper_motor_max
        clipped = (
            min(max(action.get("proximal.pos", self._gripper_pos[0]), 0.0), motor_max[0]),
            min(max(action.get("distal.pos", self._gripper_pos[1]), 0.0), motor_max[1]),
        )
        self._gripper_client.move(clipped[0], clipped[1])
        self._gripper_pos = clipped
        return {**sent, "proximal.pos": clipped[0], "distal.pos": clipped[1]}

    @check_if_not_connected
    def disconnect(self) -> None:
        client = self._gripper_client
        try:
            if self.config.disable_torque_on_disconnect:
                client.torque_off()
        except Exception:
            logger.exception("Failed to disable Gripette torque on disconnect.")
        try:
            super().disconnect()
        finally:
            client.close()
            self._gripper_client = None
