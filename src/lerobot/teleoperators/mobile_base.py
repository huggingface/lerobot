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
from typing import TYPE_CHECKING, Any

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import RobotProcessorPipeline

from .config import TeleoperatorConfig
from .keyboard import KeyboardTeleop
from .teleoperator import Teleoperator

if TYPE_CHECKING:
    from lerobot.robots import Robot

logger = logging.getLogger(__name__)


class MobileBaseTeleop(Teleoperator):
    """Combine base and optional arm actions after mapping each device's input.

    The base normally emits robot action keys. A base_action_processor can translate raw inputs
    before composition; for a disconnected raw keyboard it must map an empty input to stopped
    base commands. Arm names can receive an explicit prefix. Holding uncommanded actions at their
    first observed values is opt-in. Overlapping base and arm action keys are rejected.
    """

    config_class = TeleoperatorConfig
    name = "mobile_base"

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
        self._owns_connections = False
        super().__init__(TeleoperatorConfig(id=getattr(base, "id", None)))
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
    def action_features(self) -> dict[str, type | tuple]:
        base_features = {} if self._raw_keys else dict(self.base.action_features)
        if self.base_action_processor is not None:
            features = {
                PipelineFeatureType.ACTION: base_features,
                PipelineFeatureType.OBSERVATION: dict(self.robot.observation_features),
            }
            mapped = self.base_action_processor.transform_features(features)[PipelineFeatureType.ACTION]
            base_features = {
                key: (float if value.shape == (1,) else value.shape)
                if isinstance(value, PolicyFeature)
                else value
                for key, value in mapped.items()
            }
        if self.arm is not None:
            arm_features = {
                f"{self.arm_prefix}{key}": value for key, value in self.arm.action_features.items()
            }
        elif self.hold_uncommanded:
            arm_features = {
                key: value for key, value in self.robot.action_features.items() if key not in base_features
            }
        else:
            arm_features = {}
        self._check_overlap(arm_features, base_features)
        return {**arm_features, **base_features}

    @property
    def feedback_features(self) -> dict[str, type]:
        base_features = self.base.feedback_features
        arm_features = (
            {f"{self.arm_prefix}{key}": value for key, value in self.arm.feedback_features.items()}
            if self.arm is not None
            else {}
        )
        self._check_overlap(arm_features, base_features)
        return {**arm_features, **base_features}

    @staticmethod
    def _check_overlap(arm: dict, base: dict) -> None:
        overlap = arm.keys() & base.keys()
        if overlap:
            raise ValueError(f"Base and arm teleoperators command overlapping actions: {sorted(overlap)}")

    @property
    def is_connected(self) -> bool:
        """Remain connected while any child needs cleanup, including after a keyboard disconnect."""
        return self.base.is_connected or (self.arm is not None and self.arm.is_connected)

    @property
    def is_calibrated(self) -> bool:
        return self.base.is_calibrated and (self.arm is None or self.arm.is_calibrated)

    def calibrate(self) -> None:
        self.base.calibrate()
        if self.arm is not None:
            self.arm.calibrate()

    def configure(self) -> None:
        self.base.configure()
        if self.arm is not None:
            self.arm.configure()

    def connect(self, calibrate: bool = True) -> None:
        self._owns_connections = True
        self._hold = None
        self._observation = None
        self._warned_base_disconnected = False
        if self.base_action_processor is not None:
            self.base_action_processor.reset()
        try:
            if self.arm is not None:
                self.arm.connect(calibrate=calibrate)
            self.base.connect(calibrate=calibrate)
        except Exception:
            self.disconnect()
            raise

    def __del__(self) -> None:
        # List-form recording borrows already connected devices for one phase.
        if getattr(self, "_owns_connections", False):
            super().__del__()

    def disconnect(self) -> None:
        # Always release the arm, even if disconnecting the base fails.
        try:
            if self.base.is_connected:
                self.base.disconnect()
        finally:
            if self.arm is not None and self.arm.is_connected:
                self.arm.disconnect()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        self._check_overlap(
            {f"{self.arm_prefix}{key}": value for key, value in self.arm.feedback_features.items()}
            if self.arm is not None
            else {},
            self.base.feedback_features,
        )
        for device, prefix in ((self.base, ""), (self.arm, self.arm_prefix)):
            if device is None:
                continue
            selected = {
                key: feedback[f"{prefix}{key}"]
                for key in device.feedback_features
                if f"{prefix}{key}" in feedback
            }
            if selected:
                device.send_feedback(selected)

    @property
    def supports_position_feedback(self) -> bool:
        """Advertise actuated handover only when every feedback recipient supports torque control."""
        devices = [
            device for device in (self.base, self.arm) if device is not None and device.feedback_features
        ]
        return bool(devices) and all(
            getattr(device, "supports_position_feedback", True)
            and callable(getattr(device, "enable_torque", None))
            and callable(getattr(device, "disable_torque", None))
            for device in devices
        )

    def enable_torque(self) -> None:
        """Enable torque on the devices that receive position feedback."""
        self._set_torque("enable_torque")

    def disable_torque(self) -> None:
        """Release the devices that receive position feedback for human control."""
        self._set_torque("disable_torque")

    def _set_torque(self, method: str) -> None:
        if not self.supports_position_feedback:
            raise NotImplementedError(
                "The composed devices do not support position feedback with torque control."
            )
        for device in (self.base, self.arm):
            if device is not None and device.feedback_features:
                getattr(device, method)()

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

    def get_action(self, observation: RobotObservation | None = None) -> RobotAction:
        """Combine device actions using the observation delivered by the control loop."""
        if observation is None:
            observation = self._observation
        else:
            self.on_observation(observation)
        if observation is None:
            raise RuntimeError("Call on_observation before requesting a mobile base action.")
        base_action = self._base_action(observation)
        if self.arm is not None:
            arm_action = {f"{self.arm_prefix}{key}": value for key, value in self.arm.get_action().items()}
        elif self.hold_uncommanded:
            arm_action = self._held_arm_action(observation, base_action.keys())
        else:
            arm_action = {}
        self._check_overlap(arm_action, base_action)
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
    """Apply robot-provided arm prefixes and optional legacy keyboard bindings."""
    processor = None
    if isinstance(base, KeyboardTeleop) and base.name == "keyboard":
        key_bindings = getattr(robot, "legacy_keyboard_bindings", None)
        if key_bindings is None:
            raise ValueError("Use keyboard_base or a base teleoperator that emits this robot's action keys.")
        processor = base.get_base_action_processor(key_bindings)
    return MobileBaseTeleop(
        robot,
        base,
        arm,
        base_action_processor=processor,
        arm_action_prefix=getattr(robot, "arm_action_prefix", ""),
        hold_uncommanded=hold_uncommanded,
    )
