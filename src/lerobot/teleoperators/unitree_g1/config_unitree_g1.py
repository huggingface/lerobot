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
    """Serial port configuration for one exoskeleton arm.

    Args:
        port (`str`, *optional*, defaults to `""`):
            Serial port the exoskeleton arm's sensor board is connected to, e.g. `/dev/ttyUSB0`. An empty
            string disables exoskeleton control for that arm.
        baud_rate (`int`, *optional*, defaults to 115200):
            Baud rate for the serial connection.
    """

    port: str = ""
    baud_rate: int = 115200


@TeleoperatorConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1TeleoperatorConfig(TeleoperatorConfig):
    """Configuration for the Unitree G1 bimanual exoskeleton teleoperator.

    Args:
        left_arm_config (`ExoskeletonArmPortConfig`, *optional*):
            Serial port settings for the left exoskeleton arm. Leave `port` empty to run without exoskeleton
            control on this side.
        right_arm_config (`ExoskeletonArmPortConfig`, *optional*):
            Serial port settings for the right exoskeleton arm. Leave `port` empty to run without
            exoskeleton control on this side.
        frozen_joints (`str`, *optional*, defaults to `""`):
            Comma-separated G1 arm joint names to exclude from the exoskeleton-driven inverse kinematics.
            These joints are held at their neutral pose instead of being tracked.
        id (`str`, *optional*):
            Identifier for this particular unit, used to tell apart several teleoperators of the same
            type. It also names the calibration file, so keep it stable for a given piece of hardware.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to a per-teleoperator directory under
            the LeRobot calibration home.
    """

    left_arm_config: ExoskeletonArmPortConfig = field(default_factory=ExoskeletonArmPortConfig)
    right_arm_config: ExoskeletonArmPortConfig = field(default_factory=ExoskeletonArmPortConfig)

    # Frozen joints (comma-separated joint names that won't be moved by IK)
    frozen_joints: str = ""
