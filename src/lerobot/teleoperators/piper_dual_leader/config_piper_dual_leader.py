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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_dual_leader")
@dataclass
class PIPERDualLeaderConfig(TeleoperatorConfig):
    """Configuration for Piper Dual Leader Teleoperator.

    This teleoperator is designed for hardware-level teleoperation where:
    - PC connects to Follower arms via USB
    - Leader arms send control commands via CAN bus to Followers
    - This teleoperator reads Master control frames from Follower's CAN interface

    Attributes:
        left_port: CAN interface name for left follower arm (e.g., "can_left")
        right_port: CAN interface name for right follower arm (e.g., "can_right")
    """

    left_port: str = "can_left"
    right_port: str = "can_right"
