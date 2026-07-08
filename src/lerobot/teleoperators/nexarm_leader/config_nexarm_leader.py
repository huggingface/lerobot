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


@TeleoperatorConfig.register_subclass("nexarm_leader")
@dataclass
class NexArmLeaderConfig(TeleoperatorConfig):
    """Configuration for the NexArm leader (master) teleoperator.

    The leader connects to the master ESP32 via USB serial. The master
    directly controls HX-30HM servos. In teleop mode, torque is disabled
    so the operator can freely drag the arm.

    Attributes:
        port: Serial port for the master ESP32 (e.g. "COM18", "/dev/ttyUSB0").
        baudrate: Serial baud rate (default 1 Mbps).
    """

    port: str
    baudrate: int = 1_000_000
