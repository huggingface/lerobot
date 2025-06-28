#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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


@TeleoperatorConfig.register_subclass("dual_so100_leader")
@dataclass
class DualSO100LeaderConfig(TeleoperatorConfig):
    # Port to connect to the left arm
    # left_port: str = "/dev/ttyACM0"
    left_port: str

    # Port to connect to the right arm
    # right_port: str = "/dev/ttyACM1"
    right_port: str

    # ID for the left arm (used to load calibration)
    # left_id: str = "left_leader"
    left_id: str

    # ID for the right arm (used to load calibration)
    # right_id: str = "right_leader"
    right_id: str