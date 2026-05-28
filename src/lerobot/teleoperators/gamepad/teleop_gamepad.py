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
        self._pending_gripper_state_fn = None
        # Persistent gripper-width target in [-1, 1]; only used when
        # config.record_gripper_width is True. close = -1, open = +1
        # (uniform with the other action dims). Init to OPEN so the recorded
        # command stream matches the sim's default-open state before the user
        # ever presses the toggle.
        self._gripper_width_target: float = 1.0
        # Tracks intervention edge so we can re-seed the persistent gripper
        # target from live env state when the user starts intervening. Without
        # this sync the stale target (often +1 = open from reset) would force
        # the gripper open even if the policy was mid-grasp.
        self._prev_intervention: bool = False

    def set_gripper_state_fn(self, fn) -> None:
        """Register a callable that returns the env's current gripper state in
        [0,1] (0=open, 1=closed). Used by the R2 toggle to emit the opposite
        of the live state so it never desyncs from the policy/sim. Safe to
        call before or after ``connect()``."""
        self._pending_gripper_state_fn = fn
        if self.gamepad is not None:
            self.gamepad.gripper_state_fn = fn

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

        # Only pass stage_advance_button to the pygame controller — the HID
        # path doesn't support configurable buttons yet.
        if sys.platform == "darwin":
            self.gamepad = Gamepad()
        else:
            self.gamepad = Gamepad(stage_advance_button=self.config.stage_advance_button)
        if self._pending_gripper_state_fn is not None:
            self.gamepad.gripper_state_fn = self._pending_gripper_state_fn
        self.gamepad.start()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        # Update the controller to get fresh inputs
        self.gamepad.update()

        # On intervention edge (False->True), sync the persistent width target
        # from live env gripper state so the FIRST intervention step does not
        # snap the gripper to the stale latched value. gripper_state_fn returns
        # [0,1] (0=open, 1=closed); width target convention is +1=open, -1=close.
        cur_interv = bool(getattr(self.gamepad, "intervention_flag", False))
        if cur_interv and not self._prev_intervention:
            fn = getattr(self.gamepad, "gripper_state_fn", None)
            if fn is not None:
                try:
                    live = float(fn())  # [0,1]
                    live = max(0.0, min(1.0, live))
                    self._gripper_width_target = 1.0 - 2.0 * live
                except Exception:
                    pass
        self._prev_intervention = cur_interv

        # Get movement deltas from the controller. When use_yaw=True, the driver also reads
        # right stick X and returns a 4-tuple; otherwise the legacy 3-tuple is returned so
        # nothing changes for non-yaw runs.
        if self.config.use_yaw:
            delta_x, delta_y, delta_z, delta_yaw = self.gamepad.get_deltas(with_yaw=True)
        else:
            delta_x, delta_y, delta_z = self.gamepad.get_deltas()
            delta_yaw = 0.0

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

        if self.config.use_yaw and hasattr(self.gamepad, "get_yaw_delta"):
            action_dict["delta_yaw"] = float(self.gamepad.get_yaw_delta())

        # Default gripper action is to stay
        if self.config.use_gripper:
            gripper_command = self.gamepad.gripper_command()
            if getattr(self.config, "record_gripper_width", False):
                # Continuous-width mode: emit a [-1, +1] target uniform with
                # the other action dims. close → -1, open → +1, stay → hold
                # last. Persistent so policy commands are stable.
                if gripper_command == "close":
                    self._gripper_width_target = -1.0
                elif gripper_command == "open":
                    self._gripper_width_target = 1.0
                action_dict["gripper"] = float(self._gripper_width_target)
            else:
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
                TeleopEvents.STAGE_ADVANCE: False,
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

        stage_advance = False
        if hasattr(self.gamepad, "consume_stage_advance"):
            stage_advance = bool(self.gamepad.consume_stage_advance())

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
            TeleopEvents.STAGE_ADVANCE: stage_advance,
        }

    def reset_episode_state(self) -> None:
        """Clear latched per-episode state on the underlying controller so
        a new episode starts neutral. Safe to call before ``connect()``
        (no-op if the controller isn't initialized yet). Called by
        ``gym_manipulator.control_loop`` after each ``env.reset()``."""
        if self.gamepad is not None and hasattr(self.gamepad, "reset_episode_state"):
            self.gamepad.reset_episode_state()
        # Re-arm the continuous-width gripper target to OPEN so the previous
        # episode's last-latched state does not carry over into a fresh
        # episode.
        self._gripper_width_target = 1.0

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
