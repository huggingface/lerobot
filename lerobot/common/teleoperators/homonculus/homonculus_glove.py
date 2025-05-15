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
import threading
import time
from enum import Enum

import serial

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import MotorCalibration

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
    """
    HomonculusGlove designed by Hugging Face.
    """

    config_class = HomonculusGloveConfig
    name = "homonculus_glove"

    def __init__(self, config: HomonculusGloveConfig):
        super().__init__(config)
        self.config = config
        self.serial = serial.Serial(config.port, config.baud_rate, timeout=1)

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
        ]
        self._state = dict.fromkeys(self.joints, 100)
        self.thread = threading.Thread(target=self._async_read, daemon=True, name=f"{self} _async_read")
        self._lock = threading.Lock()

    @property
    def action_features(self) -> dict:
        return {f"{joint}.pos": float for joint in self.joints}

    @property
    def feedback_features(self) -> dict:
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
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.calibration_fpath.is_file()

    def calibrate(self) -> None:
        input("\nMove hand to open position. Press Enter to continue...")
        open_pos_start = self._read(normalize=False)
        open_pos_mins = open_pos_start.copy()
        open_pos_maxes = open_pos_start.copy()
        for _ in range(100):
            positions = self._read(normalize=False)
            open_pos_mins = {joint: min(positions[joint], min_) for joint, min_ in open_pos_mins.items()}
            open_pos_maxes = {joint: max(positions[joint], max_) for joint, max_ in open_pos_maxes.items()}
            time.sleep(0.01)

        print(f"{open_pos_mins=}")
        print(f"{open_pos_maxes=}")

        print("\nMove hand to closed position")
        input("Press Enter to continue...")
        closed_pos_start = self._read(normalize=False)
        closed_pos_mins = closed_pos_start.copy()
        closed_pos_maxes = closed_pos_start.copy()
        for _ in range(100):
            positions = self._read(normalize=False)
            closed_pos_mins = {joint: min(positions[joint], min_) for joint, min_ in closed_pos_mins.items()}
            closed_pos_maxes = {
                joint: max(positions[joint], max_) for joint, max_ in closed_pos_maxes.items()
            }
            time.sleep(0.01)

        open_pos = {joint: min(open_pos_maxes[joint], closed_pos_maxes[joint]) for joint in self.joints}
        closed_pos = {joint: min(closed_pos_maxes[joint], closed_pos_maxes[joint]) for joint in self.joints}

        # Due to magnet placement on the following joints, we need to inverse their min/max
        inverted_joints = [
            "thumb_0",
            "thumb_3",
            "index_2",
            "middle_2",
            "ring_2",
            "pinky_2",
            "index_0",
        ]
        for joint in inverted_joints:
            tmp_pos = open_pos[joint]
            open_pos[joint] = closed_pos[joint]
            closed_pos[joint] = tmp_pos

        self.calibration = {}
        for id_, joint in enumerate(self.joints):
            self.calibration[joint] = MotorCalibration(
                id=id_,
                drive_mode=1 if joint in inverted_joints else 0,
                homing_offset=0,
                range_min=open_pos[joint],
                range_max=closed_pos[joint],
            )

        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        pass

    def _normalize(self, values: dict[str, int], joints: list[str] | None):
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
        # if joints is None:
        #     joints = self.motor_names

        # # Convert from unsigned int32 original range [0, 2**32] to signed float32 range
        # values = values.astype(np.float32)

        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for joint, val in values.items():
            min_ = self.calibration[joint].range_min
            max_ = self.calibration[joint].range_max
            bounded_val = min(max_, max(min_, val))

            normalized_values[joint] = ((bounded_val - min_) / (max_ - min_)) * 100

        return normalized_values

    def _read(self, joints: list[str] | None = None, normalize: bool = True) -> dict[str, int | float]:
        """
        Return the most recent (single) values from self.last_d,
        optionally applying calibration.
        """
        with self._lock:
            state = self._state

        if joints is not None:
            state = {k: v for k, v in state.items() if k in joints}

        if normalize:
            state = self._normalize(state, joints)

        return state

    def _async_read(self):
        """
        Continuously read from the serial buffer in its own thread and sends values to the main thread through
        a queue.
        """
        while True:
            if self.serial.in_waiting > 0:
                self.serial.flush()
                positions = self.serial.readline().decode("utf-8").strip().split(" ")
                if len(positions) != len(self.joints):
                    continue

                try:
                    joint_positions = {
                        joint: int(pos) for joint, pos in zip(self.joints, positions, strict=True)
                    }
                except ValueError:
                    continue

                with self._lock:
                    self._state = joint_positions

    def get_action(self) -> dict[str, float]:
        joint_positions = self._read()
        return {f"{joint}.pos": pos for joint, pos in joint_positions.items()}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.thread.join()
        self.serial.close()
        logger.info(f"{self} disconnected.")
