#!/usr/bin/env python

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
from ..unitree_g1.config_unitree_g1 import ExoskeletonArmPortConfig


@TeleoperatorConfig.register_subclass("asimov_exo")
@dataclass
class AsimovExoTeleoperatorConfig(TeleoperatorConfig):
    """The Unitree G1 exoskeleton arms, retargeted to Asimov 1 by end-effector pose."""

    left_arm_config: ExoskeletonArmPortConfig = field(default_factory=ExoskeletonArmPortConfig)
    right_arm_config: ExoskeletonArmPortConfig = field(default_factory=ExoskeletonArmPortConfig)

    # Comma-separated Asimov joint names (e.g. "L_Wrist_Yaw,R_Wrist_Yaw") that IK won't move.
    frozen_joints: str = ""
    # Scales exo end-effector offsets from the shoulder. None: Asimov reach / exo reach.
    arm_scale: float | None = None
    # IK weight of end-effector orientation relative to position (5-DOF arms can't match both).
    rotation_weight: float = 0.02
    # Hand roll added about the forearm axis, in degrees; mirrored (left +, right -).
    wrist_roll_offset_deg: float = 60.0
    # EMA window (in reads) on exo joint angles, alpha = 2 / (n + 1); 1 disables smoothing.
    ema_window: int = 3

    # Match --robot.control_mode. "arms": exo arms drive Asimov's arms via IK.
    # "velocity": exo joysticks drive locomotion (left stick: walk/strafe, right stick x: turn).
    # "whole_body": both at once.
    control_mode: str = "arms"
    # Normalized stick deflection below which an axis reads zero.
    joystick_deadzone: float = 0.1
    # Velocity at full stick deflection; the robot clamps to its own limits.
    max_linear_velocity: float = 0.3  # m/s
    max_lateral_velocity: float = 0.2  # m/s
    max_angular_velocity: float = 30.0  # deg/s

    def __post_init__(self):
        if self.ema_window < 1:
            raise ValueError("ema_window must be >= 1.")
        if self.control_mode not in ("arms", "velocity", "whole_body"):
            raise ValueError("control_mode must be 'arms', 'velocity' or 'whole_body'.")
        if not 0 <= self.joystick_deadzone < 1:
            raise ValueError("joystick_deadzone must be in [0, 1).")
