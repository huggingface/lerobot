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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("ds4_arm")
@dataclass
class DS4ArmConfig(TeleoperatorConfig):
    """Configuration for the DualShock 4 → SO-ARM101 teleoperator.

    Stick / button layout (mirrors ds4_follower1.py):
      LS X  → shoulder_pan    LS Y  → shoulder_lift
      RS Y  → elbow_flex      RS X  → wrist_roll
      L1/R1 → wrist_flex      D-pad ↑↓ → gripper
    """

    # Ordered list of motor names matching the SO-101 follower bus
    motor_names: list[str] = field(
        default_factory=lambda: [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
    )

    # Control feel
    deadzone: float = 0.15
    smooth_alpha: float = 0.18  # EMA weight for analog axes

    # Speed presets (deg/s in normalised action units)
    speed_slow: float = 40.0
    speed_normal: float = 60.0
    speed_fast: float = 90.0
    speed_dpad: float = 30.0  # gripper (digital)
    speed_l1r1: float = 30.0  # wrist_flex (digital)

    # Target update rate
    fps: float = 30.0

    # Pygame joystick index (0 = first detected controller)
    joystick_index: int = 0
