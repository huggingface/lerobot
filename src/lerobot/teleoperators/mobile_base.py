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
drives the arm. Without a leader arm, the arms hold the position they had when driving started, so a
mobile robot can be teleoperated and recorded from the keyboard alone.
"""

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from lerobot.lerobot_types import RobotAction, RobotObservation

from .keyboard import KeyboardTeleop
from .teleoperator import Teleoperator

if TYPE_CHECKING:
    from lerobot.robots import Robot

logger = logging.getLogger(__name__)


@runtime_checkable
class KeyboardDrivenBase(Protocol):
    """A robot whose base is driven from the raw keys of a plain ``keyboard`` teleoperator.

    Robots implement this to map pressed keys to their base action (e.g. ``x.vel``, ``y.vel``,
    ``theta.vel``). The observation is passed so a mapping can be relative to measured state, e.g. a lift
    commanded from its current height.
    """

    def from_keyboard_to_base_action(
        self, pressed_keys: Iterable[str], observation: RobotObservation | None = None
    ) -> RobotAction: ...


class MobileBaseTeleop:
    """Combines a base teleoperator with an optional arm teleoperator into one robot action.

    The base teleoperator is either a plain ``keyboard``, whose pressed keys the robot maps through
    :meth:`KeyboardDrivenBase.from_keyboard_to_base_action`, or any teleoperator that already emits the
    robot's base action keys (``keyboard_rover``, a gamepad, ...).

    Arm actions are prefixed with the robot's ``arm_action_prefix`` attribute when it has one (LeKiwi's
    arm keys are ``arm_<joint>.pos``). Without an arm teleoperator, every robot action that the base does
    not command is held at the value observed on the first call.
    """

    def __init__(self, robot: "Robot", base: Teleoperator, arm: Teleoperator | None = None) -> None:
        self._raw_keys = type(base) is KeyboardTeleop
        if self._raw_keys and not isinstance(robot, KeyboardDrivenBase):
            raise TypeError(
                f"A plain `keyboard` cannot drive {robot.name}: the robot does not implement "
                "`from_keyboard_to_base_action`. Use a teleoperator that emits its base actions instead."
            )
        self.robot = robot
        self.base = base
        self.arm = arm
        self.arm_prefix: str = getattr(robot, "arm_action_prefix", "")
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
            # The keyboard disconnects itself on ESC; stop the base rather than fail the loop.
            if not self._warned_base_disconnected:
                logger.warning("The base teleoperator is disconnected, the base is stopped.")
                self._warned_base_disconnected = True
            if self._raw_keys:
                return self.robot.from_keyboard_to_base_action([], observation)  # type: ignore[attr-defined]
            return dict.fromkeys(self.base.action_features, 0.0)

        action = self.base.get_action()
        if self._raw_keys:
            return self.robot.from_keyboard_to_base_action(list(action), observation)  # type: ignore[attr-defined]
        return action

    def _held_arm_action(self, observation: RobotObservation, base_keys: Iterable[str]) -> RobotAction:
        if self._hold is None:
            held_keys = [key for key in self.robot.action_features if key not in set(base_keys)]
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
        else:
            arm_action = self._held_arm_action(observation, base_action.keys())
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
        return cls(robot, base=keyboards[0], arm=others[0] if others else None)
