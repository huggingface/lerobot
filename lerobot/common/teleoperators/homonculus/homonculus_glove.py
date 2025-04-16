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

import logging
import os
import pickle  # nosec
import threading
import time
from collections import deque
from enum import Enum

import numpy as np
import serial

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_homonculus import HomonculusGloveConfig

logger = logging.getLogger(__name__)

LOWER_BOUND_LINEAR = -100
UPPER_BOUND_LINEAR = 200


class CalibrationMode(Enum):
    # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
    DEGREE = 0
    # Joints with linear motions (like gripper of Aloha) are expressed in nominal range of [0, 100]
    LINEAR = 1


class HomonculusGlove(Teleoperator):
    config_class = HomonculusGloveConfig
    name = "homonculus_glove"

    def __init__(self, config: HomonculusGloveConfig):
        super().__init__(config)
        self.config = config
        self.serial = serial.Serial(config.port, config.baud_rate, timeout=1)
        self.buffer_size = config.buffer_size

        self.joints = [
            "thumb_0",
            "thumb_1",
            "thumb_2",
            "thumb_3",
            "index_0",
            "index_1",
            "index_2",
            "middle_0",
            "middle_1",
            "middle_2",
            "ring_0",
            "ring_1",
            "ring_2",
            "pinky_0",
            "pinky_1",
            "pinky_2",
            "battery_voltage",  # TODO(aliberts): Should this be in joints?
        ]
        # Initialize a buffer (deque) for each joint
        self.joints_buffer = {joint: deque(maxlen=self.buffer_size) for joint in self.joints}
        # Last read dictionary
        self.last_d = dict.fromkeys(self.joints, 100)

        self.calibration = None
        self.thread = threading.Thread(target=self.async_read, daemon=True)

    @property
    def action_feature(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.joints),),
            "names": {"motors": self.joints},
        }

    @property
    def feedback_feature(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.thread.is_alive() and self.serial.is_open

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if not self.serial.is_open:
            self.serial.open()
        self.thread.start()
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        raise NotImplementedError  # TODO

    def calibrate(self) -> None:
        raise NotImplementedError  # TODO

    def configure(self) -> None:
        raise NotImplementedError  # TODO

    def get_action(self) -> dict[str, float]:
        raise NotImplementedError  # TODO

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.thread.join()
        self.serial.close()
        logger.info(f"{self} disconnected.")

    ### WIP below ###

    @property
    def joint_names(self):
        return list(self.last_d.keys())

    def read(self, motor_names: list[str] | None = None):
        """
        Return the most recent (single) values from self.last_d,
        optionally applying calibration.
        """
        if motor_names is None:
            motor_names = self.joint_names

        # Get raw (last) values
        values = np.array([self.last_d[k] for k in motor_names])

        # Apply calibration if available
        if self.calibration is not None:
            values = self.apply_calibration(values, motor_names)
            print(values)
        return values

    def read_running_average(self, motor_names: list[str] | None = None, linearize=False):
        """
        Return the AVERAGE of the most recent self.buffer_size (or fewer, if not enough data) readings
        for each joint, optionally applying calibration.
        """
        if motor_names is None:
            motor_names = self.joint_names

        # Gather averaged readings from buffers
        smoothed_vals = []
        for name in motor_names:
            buf = self.joints_buffer[name]
            if len(buf) == 0:
                # If no data has been read yet, fall back to last_d
                smoothed_vals.append(self.last_d[name])
            else:
                # Otherwise, average over the existing buffer
                smoothed_vals.append(np.mean(buf))

        smoothed_vals = np.array(smoothed_vals, dtype=np.float32)

        # Apply calibration if available
        if self.calibration is not None:
            smoothed_vals = self.apply_calibration(smoothed_vals, motor_names)

        return smoothed_vals

    def async_read(self):
        """
        Continuously read from the serial buffer in its own thread,
        store into `self.last_d` and also append to the rolling buffer (joint_buffer).
        """
        while True:
            if self.serial.in_waiting > 0:
                self.serial.flush()
                vals = self.serial.readline().decode("utf-8").strip()
                vals = vals.split(" ")
                if len(vals) != 17:
                    continue
                vals = [int(val) for val in vals]

                d = {
                    "thumb_0": vals[0],
                    "thumb_1": vals[1],
                    "thumb_2": vals[2],
                    "thumb_3": vals[3],
                    "index_0": vals[4],
                    "index_1": vals[5],
                    "index_2": vals[6],
                    "middle_0": vals[7],
                    "middle_1": vals[8],
                    "middle_2": vals[9],
                    "ring_0": vals[10],
                    "ring_1": vals[11],
                    "ring_2": vals[12],
                    "pinky_0": vals[13],
                    "pinky_1": vals[14],
                    "pinky_2": vals[15],
                    "battery_voltage": vals[16],
                }

                # Update the last_d dictionary
                self.last_d = d

                # Also push these new values into the rolling buffers
                for joint_name, joint_val in d.items():
                    self.joints_buffer[joint_name].append(joint_val)

    def run_calibration(self):
        print("\nMove hand to open position")
        input("Press Enter to continue...")
        open_pos_list = []
        for _ in range(100):
            open_pos = self.read()
            open_pos_list.append(open_pos)
            time.sleep(0.01)
        open_pos = np.array(open_pos_list)
        max_open_pos = open_pos.max(axis=0)
        min_open_pos = open_pos.min(axis=0)

        print(f"{max_open_pos=}")
        print(f"{min_open_pos=}")

        print("\nMove hand to closed position")
        input("Press Enter to continue...")
        closed_pos_list = []
        for _ in range(100):
            closed_pos = self.read()
            closed_pos_list.append(closed_pos)
            time.sleep(0.01)
        closed_pos = np.array(closed_pos_list)
        max_closed_pos = closed_pos.max(axis=0)
        closed_pos[closed_pos < 1000] = 60000
        min_closed_pos = closed_pos.min(axis=0)

        print(f"{max_closed_pos=}")
        print(f"{min_closed_pos=}")

        open_pos = np.array([max_open_pos, max_closed_pos]).max(axis=0)
        closed_pos = np.array([min_open_pos, min_closed_pos]).min(axis=0)

        # INVERSION
        for i, jname in enumerate(self.joint_names):
            if jname in [
                "thumb_0",
                "thumb_3",
                "index_2",
                "middle_2",
                "ring_2",
                "pinky_2",
                "index_0",
            ]:
                tmp_pos = open_pos[i]
                open_pos[i] = closed_pos[i]
                closed_pos[i] = tmp_pos

        print()
        print(f"{open_pos=}")
        print(f"{closed_pos=}")

        homing_offset = [0] * len(self.joint_names)
        drive_mode = [0] * len(self.joint_names)
        calib_modes = [CalibrationMode.LINEAR.name] * len(self.joint_names)

        calib_dict = {
            "homing_offset": homing_offset,
            "drive_mode": drive_mode,
            "start_pos": open_pos,
            "end_pos": closed_pos,
            "calib_mode": calib_modes,
            "motor_names": self.joint_names,
        }

        file_path = "examples/hopejr/settings/hand_calib.pkl"

        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                pickle.dump(calib_dict, f)  # TODO(aliberts): use json
            print(f"Dictionary saved to {file_path}")

        # return calib_dict
        self.set_calibration(calib_dict)

    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Convert from unsigned int32 joint position range [0, 2**32[ to the universal float32 nominal degree range ]-180.0, 180.0[ with
        a "zero position" at 0 degree.

        Note: We say "nominal degree range" since the motors can take values outside this range. For instance, 190 degrees, if the motor
        rotate more than a half a turn from the zero position. However, most motors can't rotate more than 180 degrees and will stay in this range.

        Joints values are original in [0, 2**32[ (unsigned int32). Each motor are expected to complete a full rotation
        when given a goal position that is + or - their resolution. For instance, feetech xl330-m077 have a resolution of 4096, and
        at any position in their original range, let's say the position 56734, they complete a full rotation clockwise by moving to 60830,
        or anticlockwise by moving to 52638. The position in the original range is arbitrary and might change a lot between each motor.
        To harmonize between motors of the same model, different robots, or even models of different brands, we propose to work
        in the centered nominal degree range ]-180, 180[.
        """
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from unsigned int32 original range [0, 2**32] to signed float32 range
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Rescale the present position to a nominal range [0, 100] %,
                # useful for joints with linear motions like Aloha gripper
                values[i] = (values[i] - start_pos) / (end_pos - start_pos) * 100

                if (values[i] < LOWER_BOUND_LINEAR) or (values[i] > UPPER_BOUND_LINEAR):
                    if name == "pinky_1" and (values[i] < LOWER_BOUND_LINEAR):
                        values[i] = end_pos
                    else:
                        msg = (
                            f"Wrong motor position range detected for {name}. "
                            f"Expected to be in nominal range of [0, 100] % (a full linear translation), "
                            f"with a maximum range of [{LOWER_BOUND_LINEAR}, {UPPER_BOUND_LINEAR}] % to account for some imprecision during calibration, "
                            f"but present value is {values[i]} %. "
                            "This might be due to a cable connection issue creating an artificial jump in motor values. "
                            "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                        )
                        print(msg)
                        # raise JointOutOfRangeError(msg)

        return values
