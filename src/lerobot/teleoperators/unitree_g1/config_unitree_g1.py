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


@dataclass
class ExoskeletonArmPortConfig:
    """Serial port configuration for individual exoskeleton arm."""

    port: str = ""
    baud_rate: int = 115200


@TeleoperatorConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1TeleoperatorConfig(TeleoperatorConfig):
    left_arm_config: ExoskeletonArmPortConfig = field(default_factory=ExoskeletonArmPortConfig)
    right_arm_config: ExoskeletonArmPortConfig = field(default_factory=ExoskeletonArmPortConfig)

    # Frozen joints (comma-separated joint names that won't be moved by IK)
    frozen_joints: str = ""
