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
from ..homunculus.config_homunculus import HomunculusArmConfig


@TeleoperatorConfig.register_subclass("unitree_g1_bimanual")
@dataclass
class UnitreeG1BimanualConfig(TeleoperatorConfig):
    """Configuration for bimanual Homunculus arms to control Unitree G1 arms."""

    left_arm_config: HomunculusArmConfig
    right_arm_config: HomunculusArmConfig

