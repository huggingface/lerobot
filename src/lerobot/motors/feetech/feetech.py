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
from pprint import pformat
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _rustypot_available, require_package

from ..motors_bus import MotorCalibration, NameOrID, SerialMotorsBus, Value

if TYPE_CHECKING or _rustypot_available:
    import rustypot
else:
    rustypot = None


class OperatingMode(Enum):
    # position servo mode
    POSITION = 0
    # The motor is in constant speed mode, which is controlled by parameter 0x2e, and the highest bit 15 is
    # the direction bit
    VELOCITY = 1
    # PWM open-loop speed regulation mode, with parameter 0x2c running time parameter control, bit11 as
    # direction bit
    PWM = 2
    # In step servo mode, the number of step progress is represented by parameter 0x2a, and the highest bit 15
    # is the direction bit
    STEP = 3


class DriveMode(Enum):
    NON_INVERTED = 0
    INVERTED = 1


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


class FeetechMotorsBus(SerialMotorsBus):
    """`SerialMotorsBus` for Feetech servos, which speak protocol v1.

    STS/SMS and SCS servos differ in byte order, in the registers they have and in
    whether they answer Sync Read. All of that comes from the rustypot definition of
    each motor's model, so the bus only has to ask.
    """

    apply_drive_mode = True

    @staticmethod
    def _servos() -> tuple[Any, ...]:
        require_package("rustypot", extra="rustypot-dep")
        return (rustypot.Sts3215PyController, rustypot.Scs0009PyController)

    def _assert_same_firmware(self) -> None:
        firmware_versions = {}
        for id_ in self.ids:
            major = self._read("Firmware_Major_Version", id_)
            minor = self._read("Firmware_Minor_Version", id_)
            firmware_versions[id_] = f"{major}.{minor}"

        if len(set(firmware_versions.values())) != 1:
            raise RuntimeError(
                "Some Motors use different firmware versions:"
                f"\n{pformat(firmware_versions)}\n"
                "Update their firmware first using Feetech's software. "
                "Visit https://www.feetechrc.com/software."
            )

    def _handshake(self) -> None:
        self._assert_motors_exist()
        self._assert_same_firmware()

    def configure_motors(self, return_delay_time=0, maximum_acceleration=254, acceleration=254) -> None:
        for motor in self.motors:
            # By default, Feetech motors have a 500µs delay response time (corresponding to a value of 250 on
            # the 'Return_Delay_Time' address). We ensure this is reduced to the minimum of 2µs (value of 0).
            self.write("Return_Delay_Time", motor, return_delay_time)
            # Set 'Maximum_Acceleration' to 254 to speedup acceleration and deceleration of the motors.
            # SCS servos have no such register.
            if self._has_register(motor, "Maximum_Acceleration"):
                self.write("Maximum_Acceleration", motor, maximum_acceleration)
            self.write("Acceleration", motor, acceleration)

            # Clear bit 4 (0x10) of the Phase register (0x12) to set angle feedback mode to 0.
            # This forces position readings to be in the range [0, resolution - 1] and prevents overflow or negative values.
            # Only known to be necessary for the STS3215.
            if self.motors[motor].model == "sts3215":
                phase = self.read("Phase", motor, normalize=False)
                if phase & 0x10:
                    self.write("Phase", motor, phase & ~0x10)

    @property
    def is_calibrated(self) -> bool:
        motors_calibration = self.read_calibration()
        if set(motors_calibration) != set(self.calibration):
            return False

        # SCS servos have no homing offset register; `read_calibration` reports 0 for them.
        return all(
            self.calibration[motor].range_min == cal.range_min
            and self.calibration[motor].range_max == cal.range_max
            and (
                self.calibration[motor].homing_offset == cal.homing_offset
                or not self._has_register(motor, "Homing_Offset")
            )
            for motor, cal in motors_calibration.items()
        )

    def read_calibration(self) -> dict[str, MotorCalibration]:
        offsets, mins, maxes = {}, {}, {}
        for motor in self.motors:
            mins[motor] = self.read("Min_Position_Limit", motor, normalize=False)
            maxes[motor] = self.read("Max_Position_Limit", motor, normalize=False)
            offsets[motor] = (
                self.read("Homing_Offset", motor, normalize=False)
                if self._has_register(motor, "Homing_Offset")
                else 0
            )

        calibration = {}
        for motor, m in self.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=int(offsets[motor]),
                range_min=int(mins[motor]),
                range_max=int(maxes[motor]),
            )

        return calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        for motor, calibration in calibration_dict.items():
            if self._has_register(motor, "Homing_Offset"):
                self.write("Homing_Offset", motor, calibration.homing_offset)
            self.write("Min_Position_Limit", motor, calibration.range_min)
            self.write("Max_Position_Limit", motor, calibration.range_max)

        if cache:
            self.calibration = calibration_dict

    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        """
        On Feetech Motors:
        Present_Position = Actual_Position - Homing_Offset
        """
        half_turn_homings: dict[NameOrID, Value] = {}
        for motor, pos in positions.items():
            max_res = self.resolution(self._get_motor_model(motor)) - 1
            half_turn_homings[motor] = pos - int(max_res / 2)

        return half_turn_homings

    def disable_torque(self, motors: int | str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.DISABLED.value, num_retry=num_retry)
            self.write("Lock", motor, 0, num_retry=num_retry)

    def _disable_torque(self, motor: int, num_retry: int = 0) -> None:
        self._write("Torque_Enable", motor, TorqueMode.DISABLED.value, num_retry=num_retry)
        self._write("Lock", motor, 0, num_retry=num_retry)

    def enable_torque(self, motors: int | str | list[str] | None = None, num_retry: int = 0) -> None:
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, TorqueMode.ENABLED.value, num_retry=num_retry)
            self.write("Lock", motor, 1, num_retry=num_retry)
