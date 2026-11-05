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

from .calibration_gui import RangeFinderGUI
from .damiao import DamiaoMotorsBus
from .damiao.damiao import MotorState
from .motors_bus import (
    DriveMode,
    Motor,
    MotorCalibration,
    MotorNormMode,
    MotorsBusBase,
    NameOrID,
    SerialMotorsBus,
)

# TODO: rename the homonymic MotorState classes so the exported name is unambiguous:
# motors.damiao.damiao.MotorState (exported), motors.robstride.robstride.MotorState,
# robots.unitree_g1.unitree_g1.MotorState, robots.unitree_g1.unitree_sdk2_socket.LowStateMsg.MotorState.

__all__ = [
    "DamiaoMotorsBus",
    "DriveMode",
    "Motor",
    "MotorCalibration",
    "MotorNormMode",
    "MotorState",
    "MotorsBusBase",
    "NameOrID",
    "RangeFinderGUI",
    "SerialMotorsBus",
]
