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
