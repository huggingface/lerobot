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
from collections import deque
from pprint import pformat
from typing import Deque, Dict, Optional

import serial

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.motors_bus import MotorCalibration, MotorNormMode
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..teleoperator import Teleoperator
from .config_homunculus import HomunculusArmConfig

logger = logging.getLogger(__name__)


class HomunculusArm(Teleoperator):
    """
    Homunculus Arm designed by Hugging Face.
    """

    config_class = HomunculusArmConfig
    name = "homunculus_arm"

    def __init__(self, config: HomunculusArmConfig):
        super().__init__(config)
        self.config = config
        self.serial = serial.Serial(config.port, config.baud_rate, timeout=1)
        self.serial_lock = threading.Lock()

        self.joints = {
            "shoulder_pitch": MotorNormMode.RANGE_M100_100,
            "shoulder_yaw": MotorNormMode.RANGE_M100_100,
            "shoulder_roll": MotorNormMode.RANGE_M100_100,
            "elbow_flex": MotorNormMode.RANGE_M100_100,
            "wrist_roll": MotorNormMode.RANGE_M100_100,
            "wrist_yaw": MotorNormMode.RANGE_M100_100,
            "wrist_pitch": MotorNormMode.RANGE_M100_100,
        }
        n = 50
        # EMA parameters ---------------------------------------------------
        self.n: int = n
        self.alpha: float = 2 / (n + 1)
        # one deque *per joint* so we can inspect raw history if needed
        self._buffers: Dict[str, Deque[int]] = {
            joint: deque(maxlen=n)
            for joint in (
                "shoulder_pitch",
                "shoulder_yaw",
                "shoulder_roll",
                "elbow_flex",
                "wrist_roll",
                "wrist_yaw",
                "wrist_pitch",
            )
        }
        # running EMA value per joint – lazily initialised on first read
        self._ema: Dict[str, Optional[float]] = dict.fromkeys(self._buffers)

        self._state: dict[str, float] | None = None
        self.new_state_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._read_loop, daemon=True, name=f"{self} _read_loop")
        self.state_lock = threading.Lock()

    @property
    def action_features(self) -> dict:
        return {f"{joint}.pos": float for joint in self.joints}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        with self.serial_lock:
            return self.serial.is_open and self.thread.is_alive()

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if not self.serial.is_open:
            self.serial.open()
        self.thread.start()

        # wait for the thread to ramp up & 1st state to be ready
        if not self.new_state_event.wait(timeout=2):
            raise TimeoutError(f"{self}: Timed out waiting for state after 2s.")

        if not self.is_calibrated and calibrate:
            self.calibrate()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.calibration_fpath.is_file()

    def calibrate(self) -> None:
        print(
            "\nMove all joints through their entire range of motion."
            "\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self._record_ranges_of_motion()

        self.calibration = {}
        for id_, joint in enumerate(self.joints):
            self.calibration[joint] = MotorCalibration(
                id=id_,
                drive_mode=0,
                homing_offset=0,
                range_min=range_mins[joint],
                range_max=range_maxes[joint],
            )

        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    # TODO(Steven): This function is copy/paste from the `HomunculusGlove` class. Consider moving it to an utility to reduce duplicated code.
    def _record_ranges_of_motion(
        self, joints: list[str] | None = None, display_values: bool = True
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Interactively record the min/max encoder values of each joint.

        Move the joints while the method streams live positions. Press :kbd:`Enter` to finish.

        Args:
            joints (list[str] | None, optional):  Joints to record. Defaults to every joint (`None`).
            display_values (bool, optional): When `True` (default) a live table is printed to the console.

        Raises:
            TypeError: `joints` is not `None` or a list.
            ValueError: any joint's recorded min and max are the same.

        Returns:
            tuple[dict[str, int], dict[str, int]]: Two dictionaries *mins* and *maxes* with the extreme values
            observed for each joint.
        """
        if joints is None:
            joints = list(self.joints)
        elif not isinstance(joints, list):
            raise TypeError(joints)

        display_len = max(len(key) for key in joints)

        start_positions = self._read(joints, normalize=False)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        user_pressed_enter = False
        while not user_pressed_enter:
            positions = self._read(joints, normalize=False)
            mins = {joint: int(min(positions[joint], min_)) for joint, min_ in mins.items()}
            maxes = {joint: int(max(positions[joint], max_)) for joint, max_ in maxes.items()}

            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<{display_len}} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for joint in joints:
                    print(
                        f"{joint:<{display_len}} | {mins[joint]:>6} | {positions[joint]:>6} | {maxes[joint]:>6}"
                    )

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:
                # Move cursor up to overwrite the previous output
                move_cursor_up(len(joints) + 3)

        same_min_max = [joint for joint in joints if mins[joint] == maxes[joint]]
        if same_min_max:
            raise ValueError(f"Some joints have the same min and max values:\n{pformat(same_min_max)}")

        return mins, maxes

    def configure(self) -> None:
        pass

    # TODO(Steven): This function is copy/paste from the `HomunculusGlove` class. Consider moving it to an utility to reduce duplicated code.
    def _normalize(self, values: dict[str, int]) -> dict[str, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for joint, val in values.items():
            min_ = self.calibration[joint].range_min
            max_ = self.calibration[joint].range_max
            drive_mode = self.calibration[joint].drive_mode
            bounded_val = min(max_, max(min_, val))

            if self.joints[joint] is MotorNormMode.RANGE_M100_100:
                norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                normalized_values[joint] = -norm if drive_mode else norm
            elif self.joints[joint] is MotorNormMode.RANGE_0_100:
                norm = ((bounded_val - min_) / (max_ - min_)) * 100
                normalized_values[joint] = 100 - norm if drive_mode else norm

        return normalized_values

    def _apply_ema(self, raw: Dict[str, int]) -> Dict[str, float]:
        """Update buffers & running EMA values; return smoothed dict."""
        smoothed: Dict[str, float] = {}
        for joint, value in raw.items():
            # maintain raw history
            self._buffers[joint].append(value)

            # initialise on first run
            if self._ema[joint] is None:
                self._ema[joint] = float(value)
            else:
                self._ema[joint] = self.alpha * value + (1 - self.alpha) * self._ema[joint]

            smoothed[joint] = self._ema[joint]
        return smoothed

    def _read(
        self, joints: list[str] | None = None, normalize: bool = True, timeout: float = 1
    ) -> dict[str, int | float]:
        """
        Return the most recent (single) values from self.last_d,
        optionally applying calibration.
        """
        if not self.new_state_event.wait(timeout=timeout):
            raise TimeoutError(f"{self}: Timed out waiting for state after {timeout}s.")

        with self.state_lock:
            state = self._state

        self.new_state_event.clear()

        if state is None:
            raise RuntimeError(f"{self} Internal error: Event set but no state available.")

        if joints is not None:
            state = {k: v for k, v in state.items() if k in joints}

        if normalize:
            state = self._normalize(state)

        state = self._apply_ema(state)

        return state

    def _read_loop(self):
        """
        Continuously read from the serial buffer in its own thread and sends values to the main thread through
        a queue.
        """
        while not self.stop_event.is_set():
            try:
                raw_values = None
                with self.serial_lock:
                    if self.serial.in_waiting > 0:
                        self.serial.flush()
                        raw_values = self.serial.readline().decode("utf-8").strip().split(" ")
                if raw_values is None or len(raw_values) != 21:  # 16 raw + 5 angle values
                    continue

                joint_angles = {
                    "shoulder_pitch": int(raw_values[19]),
                    "shoulder_yaw": int(raw_values[18]),
                    "shoulder_roll": int(raw_values[20]),
                    "elbow_flex": int(raw_values[17]),
                    "wrist_roll": int(raw_values[16]),
                    "wrist_yaw": int(raw_values[1]),
                    "wrist_pitch": int(raw_values[0]),
                }

                with self.state_lock:
                    self._state = joint_angles
                self.new_state_event.set()

            except Exception as e:
                logger.debug(f"Error reading frame in background thread for {self}: {e}")

    def get_action(self) -> dict[str, float]:
        joint_positions = self._read()
        return {f"{joint}.pos": pos for joint, pos in joint_positions.items()}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.stop_event.set()
        self.thread.join(timeout=1)
        self.serial.close()
        logger.info(f"{self} disconnected.")
