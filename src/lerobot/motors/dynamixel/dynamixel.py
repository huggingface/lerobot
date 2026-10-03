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

from enum import Enum
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _rustypot_available, require_package

from ..motors_bus import MotorCalibration, NameOrID, SerialMotorsBus, Value

if TYPE_CHECKING or _rustypot_available:
    import rustypot
else:
    rustypot = None


class OperatingMode(Enum):
    # DYNAMIXEL only controls current(torque) regardless of speed and position. This mode is ideal for a
    # gripper or a system that only uses current(torque) control or a system that has additional
    # velocity/position controllers.
    CURRENT = 0

    # This mode controls velocity. This mode is identical to the Wheel Mode(endless) from existing DYNAMIXEL.
    # This mode is ideal for wheel-type robots.
    VELOCITY = 1

    # This mode controls position. This mode is identical to the Joint Mode from existing DYNAMIXEL. Operating
    # position range is limited by the Max Position Limit(48) and the Min Position Limit(52). This mode is
    # ideal for articulated robots that each joint rotates less than 360 degrees.
    POSITION = 3

    # This mode controls position. This mode is identical to the Multi-turn Position Control from existing
    # DYNAMIXEL. 512 turns are supported(-256[rev] ~ 256[rev]). This mode is ideal for multi-turn wrists or
    # conveyor systems or a system that requires an additional reduction gear. Note that Max Position
    # Limit(48), Min Position Limit(52) are not used on Extended Position Control Mode.
    EXTENDED_POSITION = 4

    # This mode controls both position and current(torque). Up to 512 turns are supported (-256[rev] ~
    # 256[rev]). This mode is ideal for a system that requires both position and current control such as
    # articulated robots or grippers.
    CURRENT_POSITION = 5

    # This mode directly controls PWM output. (Voltage Control Mode)
    PWM = 16


class DriveMode(Enum):
    NON_INVERTED = 0
    INVERTED = 1


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


class DynamixelMotorsBus(SerialMotorsBus):
    """`SerialMotorsBus` for Dynamixel servos, which speak protocol v2.

    Control table reference:
    https://emanual.robotis.com/docs/en/dxl/protocol2/
    """

    apply_drive_mode = False

    @staticmethod
    def _servos() -> tuple[Any, ...]:
        require_package("rustypot", extra="rustypot-dep")
        return (rustypot.Xl330PyController, rustypot.Xl430PyController)

    def configure_motors(self, return_delay_time=0) -> None:
        # By default, Dynamixel motors have a 500µs delay response time (corresponding to a value of 250 on
        # the 'Return_Delay_Time' address). We ensure this is reduced to the minimum of 2µs (value of 0).
        for motor in self.motors:
            self.write("Return_Delay_Time", motor, return_delay_time)

    @property
    def is_calibrated(self) -> bool:
        return self.calibration == self.read_calibration()

    def read_calibration(self) -> dict[str, MotorCalibration]:
        offsets = self.sync_read("Homing_Offset", normalize=False)
        mins = self.sync_read("Min_Position_Limit", normalize=False)
        maxes = self.sync_read("Max_Position_Limit", normalize=False)
        drive_modes = self.sync_read("Drive_Mode", normalize=False)

        calibration = {}
        for motor, m in self.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=int(drive_modes[motor]),
                homing_offset=int(offsets[motor]),
                range_min=int(mins[motor]),
                range_max=int(maxes[motor]),
            )

        return calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        for motor, calibration in calibration_dict.items():
            self.write("Homing_Offset", motor, calibration.homing_offset)
            self.write("Min_Position_Limit", motor, calibration.range_min)
            self.write("Max_Position_Limit", motor, calibration.range_max)

        if cache:
            self.calibration = calibration_dict

    def disable_torque(self, motors: int | str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.DISABLED.value, num_retry=num_retry)

    def _disable_torque(self, motor: int, num_retry: int = 0) -> None:
        self._write("Torque_Enable", motor, TorqueMode.DISABLED.value, num_retry=num_retry)

    def enable_torque(self, motors: int | str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.ENABLED.value, num_retry=num_retry)

    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        """
        On Dynamixel Motors:
        Present_Position = Actual_Position + Homing_Offset
        """
        half_turn_homings: dict[NameOrID, Value] = {}
        for motor, pos in positions.items():
            max_res = self.resolution(self._get_motor_model(motor)) - 1
            half_turn_homings[motor] = int(max_res / 2) - pos

        return half_turn_homings
