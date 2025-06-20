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


@TeleoperatorConfig.register_subclass("bimanual_teleop")
@dataclass
class BimanualTeleopConfig(TeleoperatorConfig):
    # Ports for the two leader arms
    left_port: str  # Left leader arm port
    right_port: str  # Right leader arm port

    # Optional separate ID for each arm to load individual calibrations
    left_id: str | None = None
    right_id: str | None = None

    use_degrees: bool = False
