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

"""Scale gamepad movement deltas into planar base velocities."""

import math
from dataclasses import dataclass
from typing import Any

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction

from .base_action_processor import BaseActionProcessorStep
from .pipeline import ProcessorStepRegistry


@BaseActionProcessorStep.register_subclass("gamepad")
@ProcessorStepRegistry.register("gamepad_to_base_action")
@dataclass
class GamepadToBaseActionProcessorStep(BaseActionProcessorStep):
    """Map delta_x/y/z to x/y velocity (m/s) and theta velocity (deg/s).

    The existing gamepad backend applies its stick deadzone. Translation is
    bounded by linear_speed even when both axes are fully deflected.
    """

    linear_speed: float = 0.1
    angular_speed: float = 30.0

    def __post_init__(self) -> None:
        if any(not math.isfinite(speed) or speed < 0 for speed in (self.linear_speed, self.angular_speed)):
            raise ValueError("Gamepad speeds must be finite and nonnegative.")

    def action(self, action: RobotAction) -> RobotAction:
        axes = ("delta_x", "delta_y", "delta_z")
        x, y, theta = (float(action[key]) for key in axes)
        if not all(math.isfinite(value) for value in (x, y, theta)):
            x = y = theta = 0.0
        scale = max(1.0, math.hypot(x, y))
        return {
            **{key: value for key, value in action.items() if key not in axes},
            "x.vel": self.linear_speed * x / scale,
            "y.vel": self.linear_speed * y / scale,
            "theta.vel": self.angular_speed * max(-1.0, min(theta, 1.0)),
        }

    def get_config(self) -> dict[str, Any]:
        return {"linear_speed": self.linear_speed, "angular_speed": self.angular_speed}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return {
            **features,
            PipelineFeatureType.ACTION: {
                **{
                    key: value
                    for key, value in features.get(PipelineFeatureType.ACTION, {}).items()
                    if key not in ("delta_x", "delta_y", "delta_z")
                },
                **{
                    key: PolicyFeature(type=FeatureType.ACTION, shape=(1,))
                    for key in ("x.vel", "y.vel", "theta.vel")
                },
            },
        }
