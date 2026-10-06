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


"""Deprecated: `SerialMotorsBus` drives Feetech servos, from each motor's model.

`FeetechMotorsBus` and the enums below stay for one release, so code written for them keeps
working; they will then be removed.
"""

import warnings
from enum import Enum

from ..motors_bus import Motor, MotorCalibration, SerialMotorsBus


class OperatingMode(Enum):
    """Values of the Feetech operating mode register; use `SerialMotorsBus.set_operating_mode` instead."""

    POSITION = 0
    VELOCITY = 1
    PWM = 2
    STEP = 3


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


class FeetechMotorsBus(SerialMotorsBus):
    """Deprecated alias of `SerialMotorsBus`, which finds each motor's family from its model."""

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        protocol_version: int | None = None,
    ):
        warnings.warn(
            "`FeetechMotorsBus` is deprecated and will be removed in a future release: use `SerialMotorsBus` "
            "from `lerobot.motors`, which finds each motor's family from its model"
            + ("; `protocol_version` is ignored." if protocol_version is not None else "."),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(port, motors, calibration)

    @classmethod
    def scan_port(cls, port: str, model: str = "sts3215") -> dict[int, list[int]]:
        return super().scan_port(port, model)
