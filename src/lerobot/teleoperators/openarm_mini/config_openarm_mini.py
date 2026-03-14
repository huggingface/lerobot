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


@TeleoperatorConfig.register_subclass("openarm_mini")
@dataclass
class OpenArmMiniConfig(TeleoperatorConfig):
    """Configuration for OpenArm Mini teleoperator with Feetech motors (dual arms)."""

    port_right: str = "/dev/ttyUSB0"
    port_left: str = "/dev/ttyUSB1"

    use_degrees: bool = True

    def __post_init__(self):
        if self.id is None:
            right_port = self.port_right.replace("/", "_").replace("\\", "_").strip("_") or "unknown"
            left_port = self.port_left.replace("/", "_").replace("\\", "_").strip("_") or "unknown"
            self.id = f"openarm_mini_{right_port}_{left_port}"
