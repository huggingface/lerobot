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

from ..teleoperator import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("openarms_mini")
@dataclass
class OpenArmsMiniConfig(TeleoperatorConfig):
    """Configuration for OpenArms Mini teleoperator with Feetech motors (dual arms)."""
    
    # Serial ports for left and right arms
    port_right: str = "/dev/ttyUSB0"  # Serial port for right arm
    port_left: str = "/dev/ttyUSB1"   # Serial port for left arm
    
    # Whether to use degrees mode (True) or normalized mode (False)
    use_degrees: bool = True

