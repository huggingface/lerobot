# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import sys
from enum import IntEnum
from typing import Any

import numpy as np

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_gamepad import GamepadTeleopConfig


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class GamepadTeleop(Teleoperator):
    """
    Teleop class to use gamepad inputs for control.
    """

    config_class = GamepadTeleopConfig
    name = "gamepad"

    def __init__(self, config: GamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.gamepad = None

    @property
    def action_features(self) -> dict:
        # Order: xyz deltas → optional yaw → optional gripper. Keeping gripper at the
        # last index preserves SAC's `DISCRETE_DIMENSION_INDEX = -1` convention and the
        # UR10GripperPenaltyProcessorStep's `action[-1]` indexing across yaw modes.
        names: dict[str, int] = {"delta_x": 0, "delta_y": 1, "delta_z": 2}
        idx = 3
        if self.config.use_yaw:
            names["delta_yaw"] = idx
            idx += 1
        if self.config.use_gripper:
            names["gripper"] = idx
            idx += 1
        return {"dtype": "float32", "shape": (idx,), "names": names}

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        # use HidApi for macos
        if sys.platform == "darwin":
            # NOTE: On macOS, pygame doesn’t reliably detect input from some controllers so we fall back to hidapi
            from .gamepad_utils import GamepadControllerHID as Gamepad
        else:
            from .gamepad_utils import GamepadController as Gamepad

        self.gamepad = Gamepad()
        self.gamepad.start()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        # Update the controller to get fresh inputs
        self.gamepad.update()

        # Get movement deltas from the controller. When use_yaw=True, the driver also reads
        # right stick X and returns a 4-tuple; otherwise the legacy 3-tuple is returned so
        # nothing changes for non-yaw runs.
        if self.config.use_yaw:
            delta_x, delta_y, delta_z, delta_yaw = self.gamepad.get_deltas(with_yaw=True)
        else:
            delta_x, delta_y, delta_z = self.gamepad.get_deltas()
            delta_yaw = 0.0

        # Apply symmetric deadzone with linear rescale outside it. Eliminates resting-stick
        # drift and spring-back overshoot that would otherwise accumulate into latched targets.
        # dz = max(0.0, float(self.config.deadzone))
        # if dz > 0.0 and dz < 1.0:
        #     scale = 1.0 / (1.0 - dz)

        #     def _dz(v: float) -> float:
        #         if v > dz:
        #             return (v - dz) * scale
        #         if v < -dz:
        #             return (v + dz) * scale
        #         return 0.0

        #     delta_x = _dz(float(delta_x))
        #     delta_y = _dz(float(delta_y))
        #     delta_z = _dz(float(delta_z))

        # Apply per-axis sign flips so the same physical stick direction maps to the
        # operator's intuitive forward/back/left/right/up/down regardless of the robot's
        # base-frame orientation at the workstation.
        sx = -1.0 if self.config.invert_delta_x else 1.0
        sy = -1.0 if self.config.invert_delta_y else 1.0
        sz = -1.0 if self.config.invert_delta_z else 1.0
        syaw = -1.0 if self.config.invert_delta_yaw else 1.0

        # Create action dict in canonical order (xyz → yaw → gripper) so downstream
        # processors that index by name don't depend on insertion order.
        action_dict: dict = {
            "delta_x": np.float32(sx * delta_x),
            "delta_y": np.float32(sy * delta_y),
            "delta_z": np.float32(sz * delta_z),
        }
        if self.config.use_yaw:
            action_dict["delta_yaw"] = np.float32(syaw * delta_yaw)

        # Default gripper action is to stay
        if self.config.use_gripper:
            gripper_command = self.gamepad.gripper_command()
            action_dict["gripper"] = gripper_action_map[gripper_command]

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the gamepad such as intervention status,
        episode termination, success indicators, etc.

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Update gamepad state to get fresh inputs
        self.gamepad.update()

        # Check if intervention is active
        is_intervention = self.gamepad.should_intervene()

        # Get episode end status
        episode_end_status = self.gamepad.get_episode_end_status()
        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def disconnect(self) -> None:
        """Disconnect from the gamepad."""
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    @property
    def is_connected(self) -> bool:
        """Check if gamepad is connected."""
        return self.gamepad is not None

    def calibrate(self) -> None:
        """Calibrate the gamepad."""
        # No calibration needed for gamepad
        pass

    def is_calibrated(self) -> bool:
        """Check if gamepad is calibrated."""
        # Gamepad doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the gamepad."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the gamepad."""
        # Gamepad doesn't support feedback
        pass
