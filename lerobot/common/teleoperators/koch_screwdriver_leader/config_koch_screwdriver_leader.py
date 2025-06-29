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


@TeleoperatorConfig.register_subclass("koch_screwdriver_leader")
@dataclass
class KochScrewdriverLeaderConfig(TeleoperatorConfig):
    # Port to connect to the arm
    port: str

    # Sets the arm in torque mode with the gripper motor set to this value. This makes it possible to squeeze
    # the gripper and have it spring back to an open position on its own.
    gripper_open_pos: float = 50.0

    # Maximum offset applied to the gripper position when sending haptic feedback.
    # A value in encoder units (0-100) that will be added or subtracted from
    # `gripper_open_pos` in proportion to the received feedback strength.
    # For example with `haptic_range=4`, a feedback value of 1.0 will command the
    # gripper to move 4 encoder counts from its neutral position, creating a small
    # force the user can feel.  Defaults to 4 which is perceivable yet safe.
    haptic_range: float = 4.0
