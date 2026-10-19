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

"""Drive a mobile robot's base from one teleoperator, with or without a second one for its arms.

Mobile manipulators split control across input devices: a keyboard drives the base while a leader arm
drives the arm. With holding enabled, uncommanded joints stay at their first measured positions,
so a mobile robot can be teleoperated and recorded from the keyboard alone.
"""

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.keyboard_base_processor import (
    KeyboardToBaseActionProcessorStep,
    make_keyboard_base_processor,
)
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig

from .keyboard import KeyboardTeleop
from .teleoperator import Teleoperator

if TYPE_CHECKING:
    from lerobot.robots import Robot

logger = logging.getLogger(__name__)


class MobileBaseTeleop:
    """Combine base and optional arm actions after mapping each device's input.

    The base normally emits robot action keys. A base_action_processor can translate raw inputs
    before composition; for a disconnected raw keyboard it must map an empty input to stopped
    base commands. Arm names can receive an explicit prefix. Holding uncommanded actions at their
    first observed values is opt-in. Overlapping base and arm action keys are rejected.
    """

    def __init__(
        self,
        robot: "Robot",
        base: Teleoperator,
        arm: Teleoperator | None = None,
        *,
        base_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
        | None = None,
        arm_action_prefix: str = "",
        hold_uncommanded: bool = False,
    ) -> None:
        self._raw_keys = isinstance(base, KeyboardTeleop) and base.name == "keyboard"
        if self._raw_keys and base_action_processor is None:
            raise ValueError(
                "A raw keyboard needs a base_action_processor; use keyboard_base for planar velocities."
            )
        self.robot = robot
        self.base = base
        self.arm = arm
        self.base_action_processor = base_action_processor
        self.arm_prefix = arm_action_prefix
        self.hold_uncommanded = hold_uncommanded
        self._hold: RobotAction | None = None
        self._observation: RobotObservation | None = None
        self._warned_base_disconnected = False

    @property
    def is_connected(self) -> bool:
        return self.base.is_connected and (self.arm is None or self.arm.is_connected)

    def connect(self) -> None:
        self._hold = None
        self._observation = None
        self._warned_base_disconnected = False
        if self.base_action_processor is not None:
            self.base_action_processor.reset()
        try:
            if self.arm is not None:
                self.arm.connect()
            self.base.connect()
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        # Always release the arm, even if disconnecting the base fails.
        try:
            if self.base.is_connected:
                self.base.disconnect()
        finally:
            if self.arm is not None and self.arm.is_connected:
                self.arm.disconnect()

    def on_observation(self, observation: RobotObservation) -> None:
        """Pass the raw observation to both input devices before requesting their actions."""
        self._observation = observation
        self.base.on_observation(observation)
        if self.arm is not None:
            self.arm.on_observation(observation)

    def _base_action(self, observation: RobotObservation) -> RobotAction:
        if not self.base.is_connected:
            if not self._warned_base_disconnected:
                logger.warning("The base teleoperator is disconnected, the base is stopped.")
                self._warned_base_disconnected = True
            action = {} if self._raw_keys else dict.fromkeys(self.base.action_features, 0.0)
        else:
            action = self.base.get_action()
        if self.base_action_processor is not None:
            return self.base_action_processor((action, observation))
        return action

    def _held_arm_action(self, observation: RobotObservation, base_keys: Iterable[str]) -> RobotAction:
        if self._hold is None:
            commanded_keys = set(base_keys)
            held_keys = [key for key in self.robot.action_features if key not in commanded_keys]
            missing = [key for key in held_keys if key not in observation]
            if missing:
                raise ValueError(
                    f"Cannot hold {self.robot.name}'s actions {missing} without an arm teleoperator: "
                    "they are not in its observation."
                )
            self._hold = {key: observation[key] for key in held_keys}
        return dict(self._hold)

    def get_action(self) -> RobotAction:
        """Combine device actions using the observation delivered by the control loop."""
        observation = self._observation
        if observation is None:
            raise RuntimeError("Call on_observation before requesting a mobile base action.")
        base_action = self._base_action(observation)
        if self.arm is not None:
            arm_action = {f"{self.arm_prefix}{key}": value for key, value in self.arm.get_action().items()}
        elif self.hold_uncommanded:
            arm_action = self._held_arm_action(observation, base_action.keys())
        else:
            arm_action = {}
        overlap = arm_action.keys() & base_action.keys()
        if overlap:
            raise ValueError(f"Base and arm teleoperators command overlapping actions: {sorted(overlap)}")
        return {**arm_action, **base_action}

    @classmethod
    def from_list(cls, robot: "Robot", teleops: list[Teleoperator]) -> "MobileBaseTeleop":
        """Builds from ``[arm, keyboard]`` (any order) or ``[keyboard]``: the list form of ``record_loop``."""
        keyboards = [t for t in teleops if isinstance(t, KeyboardTeleop)]
        others = [t for t in teleops if not isinstance(t, KeyboardTeleop)]
        if len(keyboards) != 1 or len(others) > 1:
            raise ValueError(
                "A teleoperator list must hold exactly one keyboard for the base and at most one other "
                f"teleoperator for the arms, got {[type(t).__name__ for t in teleops]}."
            )
        return make_mobile_base_teleop(robot, base=keyboards[0], arm=others[0] if others else None)


def make_mobile_base_teleop(
    robot: "Robot", base: Teleoperator, arm: Teleoperator | None = None, *, hold_uncommanded: bool = True
) -> MobileBaseTeleop:
    """Apply CLI/legacy LeKiwi defaults at construction, keeping composition robot-independent.

    Existing LeKiwi scripts may still supply a plain keyboard and LeKiwiClientConfig.teleop_keys.
    Translate that configuration into a processor here; never call a robot keyboard method.
    Other robots use keyboard_base or a teleoperator that already emits their base actions.
    """
    processor = None
    robot_config = getattr(robot, "config", None)
    lekiwi = isinstance(robot_config, LeKiwiClientConfig)
    if isinstance(base, KeyboardTeleop) and base.name == "keyboard":
        if not isinstance(robot_config, LeKiwiClientConfig):
            raise ValueError("Use keyboard_base or a base teleoperator that emits this robot's action keys.")
        processor = make_keyboard_base_processor(
            KeyboardToBaseActionProcessorStep(key_bindings=dict(robot_config.teleop_keys))
        )
    return MobileBaseTeleop(
        robot,
        base,
        arm,
        base_action_processor=processor,
        arm_action_prefix="arm_" if lekiwi else "",
        hold_uncommanded=hold_uncommanded,
    )
