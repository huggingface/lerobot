#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
class OpenArmV1DynamixelConfigBase:
    """Base configuration for the OpenARM v1 Dynamixel leader (XL330-M288, 2x7DOF + 2 grippers).

    Unlike :class:`OpenArmMiniConfig`, which drives one arm per serial port, this leader
    carries **both arms on a single bus**: an OpenRB-150 board hosts all 16 servos
    (right arm ids 1-8, left arm ids 9-16). One instance therefore emits actions for
    both arms, and there is no bimanual wrapper.
    """

    # Serial port for the OpenRB-150 board hosting all 16 servos
    # (e.g. "/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_...-if00").
    port: str

    # Emit joint positions in degrees. Grippers are always normalised to 0-100
    # and converted to follower degrees on readout.
    use_degrees: bool = True

    # Emit the two gripper joints alongside the 14 arm joints. Turn off when the
    # follower is configured without grippers (7 axes per arm).
    include_gripper: bool = True


@TeleoperatorConfig.register_subclass("openarm_v1_dynamixel")
@dataclass
class OpenArmV1DynamixelConfig(TeleoperatorConfig, OpenArmV1DynamixelConfigBase):
    pass
