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
from .config_homonculus import HomonculusArmConfig

logger = logging.getLogger(__name__)

LOWER_BOUND_LINEAR = -100
UPPER_BOUND_LINEAR = 200


class CalibrationMode(Enum):
    # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
    DEGREE = 0
    # Joints with linear motions (like gripper of Aloha) are expressed in nominal range of [0, 100]
    LINEAR = 1


class HomonculusArm(Teleoperator):
    config_class = HomonculusArmConfig
    name = "homonculus_arm"

    def __init__(self, config: HomonculusArmConfig):
        self.config = config
        self.serial = serial.Serial(config.port, config.baud_rate, timeout=1)
        self.buffer_size = config.buffer_size

        self.joints = [
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
            "elbow_flex",
            "shoulder_roll",
            "shoulder_yaw",
            "shoulder_pitch",
        ]
        # Initialize a buffer (deque) for each joint
        self.joints_buffer = {joint: deque(maxlen=self.buffer_size) for joint in self.joints}

        # Last read dictionary
        self.last_d = dict.fromkeys(self.joints, 100)

        # For adaptive EMA, we store a "previous smoothed" state per joint
        self.adaptive_ema_state = dict.fromkeys(self.joints)
        self.kalman_state = {joint: {"x": None, "P": None} for joint in self.joints}

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

        # print(motor_names)
        print(values)

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
            buf = self.joint_buffer[name]
            if len(buf) == 0:
                # If no data has been read yet, fall back to last_d
                smoothed_vals.append(self.last_d[name])
            else:
                # Otherwise, average over the existing buffer
                smoothed_vals.append(np.mean(buf))

        smoothed_vals = np.array(smoothed_vals, dtype=np.float32)

        # Apply calibration if available
        if self.calibration is not None:
            if False:
                for i, joint_name in enumerate(motor_names):
                    # Re-use the same raw_min / raw_max from the calibration
                    calib_idx = self.calibration["motor_names"].index(joint_name)
                    min_reading = self.calibration["start_pos"][calib_idx]
                    max_reading = self.calibration["end_pos"][calib_idx]

                    b_value = smoothed_vals[i]
                    print(joint_name)
                    if joint_name == "elbow_flex":
                        print("elbow")
                        try:
                            smoothed_vals[i] = int(
                                min_reading
                                + (max_reading - min_reading)
                                * np.arcsin((b_value - min_reading) / (max_reading - min_reading))
                                / (np.pi / 2)
                            )
                        except Exception:
                            print("not working")
                            print(smoothed_vals)
                            print("not working")
            smoothed_vals = self.apply_calibration(smoothed_vals, motor_names)
        return smoothed_vals

    def read_kalman_filter(
        self, process_noise: float = 1.0, measurement_noise: float = 100.0, motors: list[str] | None = None
    ) -> np.ndarray:
        """
        Return a Kalman-filtered reading for each requested joint.

        We store a separate Kalman filter (x, P) per joint. For each new measurement Z:
          1) Predict:
             x_pred = x    (assuming no motion model)
             P_pred = P + Q
          2) Update:
             K = P_pred / (P_pred + R)
             x = x_pred + K * (Z - x_pred)
             P = (1 - K) * P_pred

        Args:
            process_noise (float, optional): Process noise (Q). Larger Q means the estimate can change more
                freely. Defaults to 1.0.
            measurement_noise (float, optional): Measurement noise (R). Larger R means we trust our sensor
                less. Defaults to 100.0.
            motors (list[str] | None, optional): If None, all joints are filtered. Defaults to None.

        Returns:
            np.ndarray: Kalman-filtered positions.
        """
        if motors is None:
            motors = self.joint_names

        current_vals = np.array([self.last_d[name] for name in motors], dtype=np.float32)
        filtered_vals = np.zeros_like(current_vals)

        for i, name in enumerate(motors):
            # Retrieve the filter state for this joint
            x = self.kalman_state[name]["x"]
            p = self.kalman_state[name]["P"]
            z = current_vals[i]

            # If this is the first reading, initialize
            if x is None or p is None:
                x = z
                p = 1.0  # or some large initial uncertainty

            # 1) Predict step
            x_pred = x  # no velocity model, so x_pred = x
            p_pred = p + process_noise

            # 2) Update step
            kalman_gain = p_pred / (p_pred + measurement_noise)  # Kalman gain
            x_new = x_pred + kalman_gain * (z - x_pred)  # new state estimate
            p_new = (1 - kalman_gain) * p_pred  # new covariance

            # Save back
            self.kalman_state[name]["x"] = x_new
            self.kalman_state[name]["P"] = p_new

            filtered_vals[i] = x_new

        if self.calibration is not None:
            filtered_vals = self.apply_calibration(filtered_vals, motors)

        return filtered_vals

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

                if len(vals) != 7:
                    continue
                try:
                    vals = [int(val) for val in vals]  # remove last digit
                except ValueError:
                    self.serial.flush()
                    vals = self.serial.readline().decode("utf-8").strip()
                    vals = vals.split(" ")
                    vals = [int(val) for val in vals]
                d = {
                    "wrist_roll": vals[0],
                    "wrist_yaw": vals[1],
                    "wrist_pitch": vals[2],
                    "elbow_flex": vals[3],
                    "shoulder_roll": vals[4],
                    "shoulder_yaw": vals[5],
                    "shoulder_pitch": vals[6],
                }

                # Update the last_d dictionary
                self.last_d = d

                # Also push these new values into the rolling buffers
                for joint_name, joint_val in d.items():
                    self.joint_buffer[joint_name].append(joint_val)

                # Optional: short sleep to avoid busy-loop
                # time.sleep(0.001)

    def run_calibration(self, robot):
        robot.arm_bus.write("Acceleration", 50)
        n_joints = len(self.joint_names)

        max_open_all = np.zeros(n_joints, dtype=np.float32)
        min_open_all = np.zeros(n_joints, dtype=np.float32)
        max_closed_all = np.zeros(n_joints, dtype=np.float32)
        min_closed_all = np.zeros(n_joints, dtype=np.float32)

        for i, jname in enumerate(self.joint_names):
            print(f"\n--- Calibrating joint '{jname}' ---")

            joint_idx = robot.arm_calib_dict["motor_names"].index(jname)
            open_val = robot.arm_calib_dict["start_pos"][joint_idx]
            print(f"Commanding {jname} to OPEN position {open_val}...")
            robot.arm_bus.write("Goal_Position", [open_val], [jname])

            input("Physically verify or adjust the joint. Press Enter when ready to capture...")

            open_pos_list = []
            for _ in range(100):
                all_joints_vals = self.read()  # read entire arm
                open_pos_list.append(all_joints_vals[i])  # store only this joint
                time.sleep(0.01)

            # Convert to numpy and track min/max
            open_array = np.array(open_pos_list, dtype=np.float32)
            max_open_all[i] = open_array.max()
            min_open_all[i] = open_array.min()
            closed_val = robot.arm_calib_dict["end_pos"][joint_idx]
            if jname == "elbow_flex":
                closed_val = closed_val - 700
            closed_val = robot.arm_calib_dict["end_pos"][joint_idx]
            print(f"Commanding {jname} to CLOSED position {closed_val}...")
            robot.arm_bus.write("Goal_Position", [closed_val], [jname])

            input("Physically verify or adjust the joint. Press Enter when ready to capture...")

            closed_pos_list = []
            for _ in range(100):
                all_joints_vals = self.read()
                closed_pos_list.append(all_joints_vals[i])
                time.sleep(0.01)

            closed_array = np.array(closed_pos_list, dtype=np.float32)
            # Some thresholding for closed positions
            # closed_array[closed_array < 1000] = 60000

            max_closed_all[i] = closed_array.max()
            min_closed_all[i] = closed_array.min()

            robot.arm_bus.write("Goal_Position", [int((closed_val + open_val) / 2)], [jname])

        open_pos = np.maximum(max_open_all, max_closed_all)
        closed_pos = np.minimum(min_open_all, min_closed_all)

        for i, jname in enumerate(self.joint_names):
            if jname not in ["wrist_pitch", "shoulder_pitch"]:
                # Swap open/closed for these joints
                tmp_pos = open_pos[i]
                open_pos[i] = closed_pos[i]
                closed_pos[i] = tmp_pos

        # Debug prints
        print("\nFinal open/closed arrays after any swaps/inversions:")
        print(f"open_pos={open_pos}")
        print(f"closed_pos={closed_pos}")

        homing_offset = [0] * n_joints
        drive_mode = [0] * n_joints
        calib_modes = [CalibrationMode.LINEAR.name] * n_joints

        calib_dict = {
            "homing_offset": homing_offset,
            "drive_mode": drive_mode,
            "start_pos": open_pos,
            "end_pos": closed_pos,
            "calib_mode": calib_modes,
            "motor_names": self.joint_names,
        }
        file_path = "examples/hopejr/settings/arm_calib.pkl"

        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                pickle.dump(calib_dict, f)  # TODO(aliberts): use json
            print(f"Dictionary saved to {file_path}")

        self.set_calibration(calib_dict)

    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """
        Example calibration that linearly maps [start_pos, end_pos] to [0,100].
        Extend or modify for your needs.
        """
        if motor_names is None:
            motor_names = self.joint_names

        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Rescale the present position to [0, 100]
                values[i] = (values[i] - start_pos) / (end_pos - start_pos) * 100

                # Check boundaries
                if (values[i] < LOWER_BOUND_LINEAR) or (values[i] > UPPER_BOUND_LINEAR):
                    # If you want to handle out-of-range differently:
                    # raise JointOutOfRangeError(msg)
                    msg = (
                        f"Wrong motor position range detected for {name}. "
                        f"Value = {values[i]} %, expected within [{LOWER_BOUND_LINEAR}, {UPPER_BOUND_LINEAR}]"
                    )
                    print(msg)

        return values
