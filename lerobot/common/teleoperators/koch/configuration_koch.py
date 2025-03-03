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


@TeleoperatorConfig.register_subclass("koch")
@dataclass
class KochTeleopConfig(TeleoperatorConfig):
    # Port to connect to the teloperator
    port: str

    # Sets the arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False

    # motors
    shoulder_pan: tuple = (1, "xl330-m077")
    shoulder_lift: tuple = (2, "xl330-m077")
    elbow_flex: tuple = (3, "xl330-m077")
    wrist_flex: tuple = (4, "xl330-m077")
    wrist_roll: tuple = (5, "xl330-m077")
    gripper: tuple = (6, "xl330-m077")
