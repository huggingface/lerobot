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


@TeleoperatorConfig.register_subclass("homunculus_glove")
@dataclass
class HomunculusGloveConfig(TeleoperatorConfig):
    port: str  # Port to connect to the glove
    side: str  # "left" / "right"
    baud_rate: int = 115_200

    def __post_init__(self):
        if self.side not in ["right", "left"]:
            raise ValueError(self.side)


@TeleoperatorConfig.register_subclass("homunculus_arm")
@dataclass
class HomunculusArmConfig(TeleoperatorConfig):
    port: str  # Port to connect to the arm
    baud_rate: int = 115_200
