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

"""Explicit hardware contracts: never guess joint units or camera geometry."""

import math
from dataclasses import dataclass, field


@dataclass
class PositionAxis:
    minimum: float
    maximum: float
    max_speed: float
    unit: str
    description: str
    tracking_tolerance: float
    measurement_tolerance: float = 0.001

    def __post_init__(self):
        values = (
            self.minimum,
            self.maximum,
            self.max_speed,
            self.tracking_tolerance,
            self.measurement_tolerance,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Position limits must be finite")
        if self.minimum > self.maximum or self.max_speed <= 0 or self.tracking_tolerance <= 0:
            raise ValueError("Invalid position bounds, speed or tracking tolerance")
        if self.measurement_tolerance < 0:
            raise ValueError("Measurement tolerance must be nonnegative")
        if not self.unit.strip() or not self.description.strip():
            raise ValueError("Every axis needs its native unit and physical description")


@dataclass
class CartesianArm:
    """Fixed-base MuJoCo model mapping, independent of the robot's motor transport.

    Native commands convert to model coordinates as q = native * scale + offset.
    Angles are [yaw, pitch, roll], relative to the measured orientation at /start.
    """

    model_path: str
    site: str
    joints: dict[str, str]  # LeRobot action key -> model joint name
    frame_description: str
    position_low: list[float]
    position_high: list[float]
    joint_scale: dict[str, float] = field(default_factory=dict)
    joint_offset: dict[str, float] = field(default_factory=dict)
    gripper: str | None = None
    angle_low: list[float] = field(default_factory=lambda: [-math.pi, 0.0, 0.0])
    angle_high: list[float] = field(default_factory=lambda: [math.pi, 0.0, 0.0])
    linear_speed: float = 0.03
    angular_speed: float = 0.15
    position_tolerance: float = 0.0002
    rotation_tolerance: float = 0.002

    def __post_init__(self):
        if not self.joints or len(set(self.joints.values())) != len(self.joints):
            raise ValueError("Cartesian joint mappings must be nonempty and unique")
        if not self.model_path or not self.site or not self.frame_description.strip():
            raise ValueError("Cartesian control needs a model, tool site and frame description")
        for low, high in ((self.position_low, self.position_high), (self.angle_low, self.angle_high)):
            if (
                len(low) != 3
                or len(high) != 3
                or any(
                    not math.isfinite(a) or not math.isfinite(b) or a > b
                    for a, b in zip(low, high, strict=True)
                )
            ):
                raise ValueError("Cartesian bounds must be three finite ordered pairs")
        if any(not a <= 0 <= b for a, b in zip(self.angle_low, self.angle_high, strict=True)):
            raise ValueError("Relative orientation bounds must contain the start orientation (zero)")
        for values in (self.joint_scale, self.joint_offset):
            if set(values) - set(self.joints) or not all(math.isfinite(v) for v in values.values()):
                raise ValueError("Invalid native-to-model joint mapping")
        if any(v == 0 for v in self.joint_scale.values()):
            raise ValueError("Joint scales cannot be zero")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (self.linear_speed, self.angular_speed, self.position_tolerance, self.rotation_tolerance)
        ):
            raise ValueError("Cartesian speeds and tolerances must be finite and positive")


@dataclass
class AgentSettings:
    model: str = "openai/gpt-6.1-sol"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    wire: str = "responses"
    effort: str = "low"
    service_tier: str | None = None
    max_llm_calls: int = 100
    max_retries: int = 3
    max_speed_frac: float = 0.1
    images: str = "always"
    transcript_echo: bool = True
    prior_learnings: str | None = None
    log_dir: str = "outputs/agent-rollout"
    robot_notes: str = ""
    axes: dict[str, PositionAxis] = field(default_factory=dict)
    cartesian: dict[str, CartesianArm] = field(default_factory=dict)
    settle_s: float = 0.5
    observation_timeout_s: float = 1.0
    return_home_on_done: bool = False

    def __post_init__(self):
        if not math.isfinite(self.settle_s) or self.settle_s < 0:
            raise ValueError("settle_s must be finite and nonnegative")
        if not math.isfinite(self.observation_timeout_s) or self.observation_timeout_s <= 0:
            raise ValueError("observation_timeout_s must be finite and positive")
