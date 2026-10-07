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


@dataclass
class OpenArmMiniConfigBase:
    """Base configuration for the OpenArm Mini teleoperator (Feetech STS3215, 7DOF + gripper)."""

    # Serial port for the Feetech bus (e.g., "/dev/ttyUSB0").
    port: str

    # Side of the arm: "left" or "right". Controls per-joint direction flips applied
    # during readout. If `None`, no flipping is applied.
    side: str | None = None

    use_degrees: bool = True


@TeleoperatorConfig.register_subclass("openarm_mini")
@dataclass
class OpenArmMiniConfig(TeleoperatorConfig, OpenArmMiniConfigBase):
    pass
