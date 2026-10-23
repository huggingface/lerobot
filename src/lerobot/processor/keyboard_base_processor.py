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

"""Translate pressed keyboard keys into planar base velocities."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction, RobotObservation

from .converters import robot_action_observation_to_transition, transition_to_robot_action
from .pipeline import ProcessorStepRegistry, RobotActionProcessorStep, RobotProcessorPipeline


def default_base_key_bindings() -> dict[str, str]:
    """Return WASD translation, ZX rotation, and +/- speed controls."""
    return {
        "forward": "w",
        "backward": "s",
        "left": "a",
        "right": "d",
        "rotate_left": "z",
        "rotate_right": "x",
        "speed_up": "+",
        "speed_down": "-",
    }


@ProcessorStepRegistry.register("keyboard_to_base_action")
@dataclass
class KeyboardToBaseActionProcessorStep(RobotActionProcessorStep):
    """Map pressed-key dictionaries to x/y velocities in m/s and theta velocity in deg/s.

    Opposite directions cancel. Speed changes once per key press and starts at the lowest
    preset after reset. Key bindings and speed presets belong to this processor, not the robot.
    """

    key_bindings: dict[str, str] = field(default_factory=default_base_key_bindings)
    linear_speeds: tuple[float, ...] = (0.1, 0.2, 0.3)
    angular_speeds: tuple[float, ...] = (30.0, 60.0, 90.0)
    _speed_index: int = field(default=0, init=False, repr=False)
    _pressed: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        missing = default_base_key_bindings().keys() - self.key_bindings.keys()
        if missing:
            raise ValueError(f"Missing keyboard bindings: {sorted(missing)}")
        if not self.linear_speeds or len(self.linear_speeds) != len(self.angular_speeds):
            raise ValueError("Linear and angular speed presets must have the same nonzero length.")
        if any(not isfinite(v) or v < 0 for v in (*self.linear_speeds, *self.angular_speeds)):
            raise ValueError("Speed presets must be finite and nonnegative.")

    def action(self, action: RobotAction) -> RobotAction:
        # Raw keyboard values are None; other devices' values must pass through unchanged.
        pressed = {key for key, value in action.items() if value is None}
        passthrough = {key: value for key, value in action.items() if value is not None}
        overlap = passthrough.keys() & {"x.vel", "y.vel", "theta.vel"}
        if overlap:
            raise ValueError(f"Keyboard mapping would overwrite base commands: {sorted(overlap)}")
        newly_pressed = pressed - self._pressed
        self._pressed = pressed
        keys = self.key_bindings
        change = int(keys["speed_up"] in newly_pressed) - int(keys["speed_down"] in newly_pressed)
        self._speed_index = max(0, min(self._speed_index + change, len(self.linear_speeds) - 1))
        xy = self.linear_speeds[self._speed_index]
        theta = self.angular_speeds[self._speed_index]
        return {
            **passthrough,
            "x.vel": xy * (int(keys["forward"] in pressed) - int(keys["backward"] in pressed)),
            "y.vel": xy * (int(keys["left"] in pressed) - int(keys["right"] in pressed)),
            "theta.vel": theta * (int(keys["rotate_left"] in pressed) - int(keys["rotate_right"] in pressed)),
        }

    def reset(self) -> None:
        self._speed_index = 0
        self._pressed.clear()

    def get_config(self) -> dict[str, Any]:
        return {
            "key_bindings": dict(self.key_bindings),
            "linear_speeds": self.linear_speeds,
            "angular_speeds": self.angular_speeds,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return {
            **features,
            PipelineFeatureType.ACTION: {
                **features.get(PipelineFeatureType.ACTION, {}),
                **{
                    key: PolicyFeature(type=FeatureType.ACTION, shape=(1,))
                    for key in ("x.vel", "y.vel", "theta.vel")
                },
            },
        }


def make_keyboard_base_processor(
    step: KeyboardToBaseActionProcessorStep,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """Build a pipeline usable on its own or before combining base and arm actions."""
    return RobotProcessorPipeline(
        steps=[step],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
