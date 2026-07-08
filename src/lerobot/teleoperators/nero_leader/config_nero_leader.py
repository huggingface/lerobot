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
class NeroLeaderConfigBase:
    port: str = "can0"
    can_interface: str = "socketcan"
    firmware_version: str = "V111"
    enable_drag_teach: bool = True
    reset_can_on_connect: bool = True
    enable_retries: int = 50
    enable_retry_interval: float = 0.05

    # Joint names matching pyAgxArm 7-DOF Nero output order
    joint_names: list[str] = field(
        default_factory=lambda: [f"joint{i}" for i in range(1, 8)]
    )

    # Face-to-face mirror: -1 = flip sign, 1 = keep
    # J1 flip (base rotates opposite), J3/J5/J6 flip (pitch direction opposite)
    # J2/J4/J7 keep (J4 asymmetric limits, cannot flip)
    mirror_sign: list[int] = field(
        default_factory=lambda: [1, 1, 1, 1, 1, 1, 1]
    )


@TeleoperatorConfig.register_subclass("nero_leader")
@dataclass
class NeroLeaderConfig(TeleoperatorConfig, NeroLeaderConfigBase):
    pass
