#!/usr/bin/env python

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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@dataclass
class ExoskeletonArmPortConfig:
    """Serial port configuration for an exoskeleton arm."""
    port: str = ""
    baud_rate: int = 115200


@TeleoperatorConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1TeleoperatorConfig(TeleoperatorConfig):
    """Configuration for bimanual exoskeleton arms to control Unitree G1 arms via IK."""

    left_arm_config: ExoskeletonArmPortConfig = field(default_factory=lambda: ExoskeletonArmPortConfig(port="/dev/ttyACM1"))
    right_arm_config: ExoskeletonArmPortConfig = field(default_factory=lambda: ExoskeletonArmPortConfig(port="/dev/ttyACM0"))
    
    # Frozen joints (comma-separated joint names that won't be moved by IK)
    frozen_joints: str = ""
    
    # Enable Meshcat 3D visualization
    visualize: bool = False
    show_axes: bool = True
