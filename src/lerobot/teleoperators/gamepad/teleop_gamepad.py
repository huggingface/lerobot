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

import logging
import sys
from enum import IntEnum
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_gamepad import GamepadTeleopConfig

logger = logging.getLogger(__name__)


class GripperAction(IntEnum):
    """Gripper command levels produced by a gamepad's gripper buttons.

    **Attributes**:
        - **CLOSE** (`int`) -- Close the gripper.
        - **STAY** (`int`) -- Leave the gripper where it is.
        - **OPEN** (`int`) -- Open the gripper.
    """

    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class GamepadTeleop(Teleoperator):
    """Teleoperator that reads a gamepad's analog sticks and buttons via `pygame` (or `hidapi`).

    [`~teleoperators.Teleoperator.get_action`] reports the left stick as `delta_x`/`delta_y` and the
    right stick's vertical axis as `delta_z`, plus an optional gripper command. See `gamepad_utils.py`'s
    `GamepadController` (`pygame`) and `GamepadControllerHID` (`hidapi`) for the exact axis/button
    mapping.

    Args:
        config (`GamepadTeleopConfig`): Configuration for this gamepad teleoperator.
    """

    config_class = GamepadTeleopConfig
    name = "gamepad"

    def __init__(self, config: GamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.gamepad = None

        self.hidapi_fallback = config.hidapi_fallback
        if sys.platform == "darwin" and not self.hidapi_fallback:
            logger.warning(
                "On macOS, pygame may not reliably detect input from some controllers. "
                "If you experience issues, set `hidapi_fallback=true`."
            )

    @property
    def action_features(self) -> dict:
        """See [`~teleoperators.Teleoperator.action_features`].

        Returns:
            `dict`: A 3-element (or 4-element if `config.use_gripper` is `True`) `float32` vector named
            `delta_x`, `delta_y`, `delta_z`, and optionally `gripper`.
        """
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @property
    def feedback_features(self) -> dict:
        """See [`~teleoperators.Teleoperator.feedback_features`]. `GamepadTeleop` accepts no feedback."""
        return {}

    def connect(self) -> None:
        """See [`~teleoperators.Teleoperator.connect`].

        Starts a `GamepadControllerHID` if `config.hidapi_fallback` is `True`, otherwise a
        `GamepadController`.
        """
        if self.hidapi_fallback:
            from .gamepad_utils import GamepadControllerHID as Gamepad
        else:
            from .gamepad_utils import GamepadController as Gamepad

        self.gamepad = Gamepad()
        self.gamepad.start()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Read the gamepad's current stick positions and gripper button state.

        The left analog stick drives `delta_x`/`delta_y`; the right stick's vertical axis drives
        `delta_z`. When `config.use_gripper` is `True`, the gripper buttons additionally produce a
        `gripper` entry (one of `GripperAction.CLOSE`, `STAY`, or `OPEN`).

        Returns:
            `dict[str, Any]`: `delta_x`, `delta_y`, `delta_z`, and, if enabled, `gripper`.

        Raises:
            DeviceNotConnectedError: If [`~teleoperators.Teleoperator.connect`] has not been called.
        """
        # Update the controller to get fresh inputs
        self.gamepad.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.gamepad.get_deltas()

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        action_dict = {
            "delta_x": gamepad_action[0],
            "delta_y": gamepad_action[1],
            "delta_z": gamepad_action[2],
        }

        # Default gripper action is to stay
        gripper_action = GripperAction.STAY.value
        if self.config.use_gripper:
            gripper_command = self.gamepad.gripper_command()
            gripper_action = gripper_action_map[gripper_command]
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """Read auxiliary gamepad events used to drive episode control during recording.

        Holding the intervention button counts as an active intervention; the success/failure/rerecord
        buttons are read once as one-shot signals, then cleared.

        Returns:
            `dict[TeleopEvents, bool]`: Values for the [`~teleoperators.TeleopEvents`] keys
            `IS_INTERVENTION`, `TERMINATE_EPISODE`, `SUCCESS`, and `RERECORD_EPISODE`. All `False` if
            [`~teleoperators.Teleoperator.connect`] has not been called yet.
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
        """See [`~teleoperators.Teleoperator.disconnect`]. Stops and releases the underlying controller."""
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    @property
    def is_connected(self) -> bool:
        """See [`~teleoperators.Teleoperator.is_connected`]."""
        return self.gamepad is not None

    def calibrate(self) -> None:
        """See [`~teleoperators.Teleoperator.calibrate`]. No-op: the gamepad does not require calibration."""
        # No calibration needed for gamepad
        pass

    def is_calibrated(self) -> bool:
        """See [`~teleoperators.Teleoperator.is_calibrated`]. Always `True`: no calibration is required."""
        # Gamepad doesn't require calibration
        return True

    def configure(self) -> None:
        """See [`~teleoperators.Teleoperator.configure`]. No-op: the gamepad needs no configuration."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """See [`~teleoperators.Teleoperator.send_feedback`]. No-op: `GamepadTeleop` accepts no feedback."""
        # Gamepad doesn't support feedback
        pass
