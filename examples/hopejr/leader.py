from lerobot.common.robot_devices.motors.feetech import (
    CalibrationMode,
    FeetechMotorsBus,
)
import serial
import threading
import time
from typing import Callable
import pickle
import cv2
import numpy as np
from collections import deque
import json 
import os
LOWER_BOUND_LINEAR = -100
UPPER_BOUND_LINEAR = 200

class HomonculusArm:
    def __init__(self, serial_port: str = "/dev/ttyACM1", baud_rate: int = 115200):
        self.serial_port = serial_port
        self.baud_rate = 115200
        self.serial = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
        
        # Number of past values to keep in memory
        self.buffer_size = 10
        
        # Initialize a buffer (deque) for each joint
        self.joint_buffer = {
            "wrist_roll": deque(maxlen=self.buffer_size),
            "wrist_pitch": deque(maxlen=self.buffer_size),
            "wrist_yaw": deque(maxlen=self.buffer_size),
            "elbow_flex": deque(maxlen=self.buffer_size),
            "shoulder_roll": deque(maxlen=self.buffer_size),
            "shoulder_yaw": deque(maxlen=self.buffer_size),
            "shoulder_pitch": deque(maxlen=self.buffer_size),
        }

        # Start the reading thread
        self.thread = threading.Thread(target=self.async_read, daemon=True)
        self.thread.start()

        # Last read dictionary
        self.last_d = {
            "wrist_roll": 100,
            "wrist_pitch": 100,
            "wrist_yaw": 100,
            "elbow_flex": 100,
            "shoulder_roll": 100,
            "shoulder_yaw": 100,
            "shoulder_pitch": 100,
        }
        self.calibration = None

        # For adaptive EMA, we store a "previous smoothed" state per joint
        self.adaptive_ema_state = {
            "wrist_roll": None,
            "wrist_pitch": None,
            "wrist_yaw": None,
            "elbow_flex": None,
            "shoulder_roll": None,
            "shoulder_yaw": None,
            "shoulder_pitch": None,
        }

        self.kalman_state = {
            joint: {"x": None, "P": None} for joint in self.joint_buffer.keys()
        }

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

        #print(motor_names)
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

                    B_value = smoothed_vals[i]
                    print(joint_name)
                    if joint_name == "elbow_flex":
                        print('elbow')
                        try:
                            smoothed_vals[i] = int(min_reading+(max_reading - min_reading)*np.arcsin((B_value-min_reading)/(max_reading-min_reading))/(np.pi / 2))
                        except:
                            print('not working')
                            print(smoothed_vals)
                            print('not working')
            smoothed_vals = self.apply_calibration(smoothed_vals, motor_names)
        return smoothed_vals

    def read_kalman_filter(
        self,
        Q: float = 1.0,
        R: float = 100.0,
        motor_names: list[str] | None = None
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

        :param Q: Process noise. Larger Q means the estimate can change more freely.
        :param R: Measurement noise. Larger R means we trust our sensor less.
        :param motor_names: If not specified, all joints are filtered.
        :return: Kalman-filtered positions as a numpy array.
        """
        if motor_names is None:
            motor_names = self.joint_names

        current_vals = np.array([self.last_d[name] for name in motor_names], dtype=np.float32)
        filtered_vals = np.zeros_like(current_vals)

        for i, name in enumerate(motor_names):
            # Retrieve the filter state for this joint
            x = self.kalman_state[name]["x"]
            P = self.kalman_state[name]["P"]
            Z = current_vals[i]

            # If this is the first reading, initialize
            if x is None or P is None:
                x = Z
                P = 1.0  # or some large initial uncertainty

            # 1) Predict step
            x_pred = x  # no velocity model, so x_pred = x
            P_pred = P + Q

            # 2) Update step
            K = P_pred / (P_pred + R)  # Kalman gain
            x_new = x_pred + K * (Z - x_pred)    # new state estimate
            P_new = (1 - K) * P_pred            # new covariance

            # Save back
            self.kalman_state[name]["x"] = x_new
            self.kalman_state[name]["P"] = P_new

            filtered_vals[i] = x_new

        if self.calibration is not None:
            filtered_vals = self.apply_calibration(filtered_vals, motor_names)

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
                    vals = [int(val) for val in vals]#remove last digit
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
            #closed_array[closed_array < 1000] = 60000

            max_closed_all[i] = closed_array.max()
            min_closed_all[i] = closed_array.min()

            robot.arm_bus.write("Goal_Position", [int((closed_val+open_val)/2)], [jname])

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
                pickle.dump(calib_dict, f)
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


class HomonculusGlove:
    def __init__(self, serial_port: str = "/dev/ttyACM1", baud_rate: int = 115200):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.serial = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
        
        # Number of past values to keep in memory
        self.buffer_size = 10
        
        # Initialize a buffer (deque) for each joint
        self.joint_buffer = {
            "thumb_0": deque(maxlen=self.buffer_size),
            "thumb_1": deque(maxlen=self.buffer_size),
            "thumb_2": deque(maxlen=self.buffer_size),
            "thumb_3": deque(maxlen=self.buffer_size),
            "index_0": deque(maxlen=self.buffer_size),
            "index_1": deque(maxlen=self.buffer_size),
            "index_2": deque(maxlen=self.buffer_size),
            "middle_0": deque(maxlen=self.buffer_size),
            "middle_1": deque(maxlen=self.buffer_size),
            "middle_2": deque(maxlen=self.buffer_size),
            "ring_0": deque(maxlen=self.buffer_size),
            "ring_1": deque(maxlen=self.buffer_size),
            "ring_2": deque(maxlen=self.buffer_size),
            "pinky_0": deque(maxlen=self.buffer_size),
            "pinky_1": deque(maxlen=self.buffer_size),
            "pinky_2": deque(maxlen=self.buffer_size),
            "battery_voltage": deque(maxlen=self.buffer_size),
        }

        # Start the reading thread
        self.thread = threading.Thread(target=self.async_read, daemon=True)
        self.thread.start()

        # Last read dictionary
        self.last_d = {
            "thumb_0": 100,
            "thumb_1": 100,
            "thumb_2": 100,
            "thumb_3": 100,
            "index_0": 100,
            "index_1": 100,
            "index_2": 100,
            "middle_0": 100,
            "middle_1": 100,
            "middle_2": 100,
            "ring_0": 100,
            "ring_1": 100,
            "ring_2": 100,
            "pinky_0": 100,
            "pinky_1": 100,
            "pinky_2": 100,
            "battery_voltage": 100,
        }
        self.calibration = None

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
                    self.joint_buffer[joint_name].append(joint_val)

    def run_calibration(self):
        print("\nMove arm to open position")
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

        print("\nMove arm to closed position")
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

        # INVERTION
        # INVERTION
        # INVERTION
        # INVERTION
        # INVERTION
        # INVERTION
        # INVERTION
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
                pickle.dump(calib_dict, f)
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

    # def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
    #     """Inverse of `apply_calibration`."""
    #     if motor_names is None:
    #         motor_names = self.motor_names

    #     for i, name in enumerate(motor_names):
    #         calib_idx = self.calibration["motor_names"].index(name)
    #         calib_mode = self.calibration["calib_mode"][calib_idx]

    #         if CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
    #             start_pos = self.calibration["start_pos"][calib_idx]
    #             end_pos = self.calibration["end_pos"][calib_idx]

    #             # Convert from nominal lnear range of [0, 100] % to
    #             # actual motor range of values which can be arbitrary.
    #             values[i] = values[i] / 100 * (end_pos - start_pos) + start_pos

    #     values = np.round(values).astype(np.int32)
    #     return values

class EncoderReader:
    def __init__(self, serial_port="/dev/ttyUSB1", baud_rate=115200):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.serial = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
        
        # Start a background thread to continuously read from the serial port
        self.thread = threading.Thread(target=self.async_read, daemon=True)
        self.thread.start()
        
        # Store the latest encoder reading in this dictionary
        self.last_d = {"encoder": 500}

    def async_read(self):
        while True:
            # Read one line from serial
            line = self.serial.readline().decode("utf-8").strip()
            if line:
                try:
                    val = int(line)  # Parse the incoming line as integer
                    self.last_d["encoder"] = val
                except ValueError:
                    # If we couldn't parse it as an integer, just skip
                    pass

    def read(self):
        """
        Returns the last encoder value that was read.
        """
        return self.last_d["encoder"]

class Tac_Man:
    def __init__(self, serial_port="/dev/ttyUSB1", baud_rate=115200):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.serial = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
        
        # Start a background thread to continuously read from the serial port
        self.thread = threading.Thread(target=self.async_read, daemon=True)
        self.thread.start()
        
        # Store the latest encoder readings in this list
        self.last_d = [0, 0, 0]  # Default values for three readings

    def async_read(self):
        while True:
            # Read one line from serial
            line = self.serial.readline().decode("utf-8").strip()
            if line:
                try:
                    # Parse the incoming line as three comma-separated integers
                    values = [int(val) for val in line.split(",")]
                    if len(values) == 3:  # Ensure we have exactly three values
                        self.last_d = values
                except ValueError:
                    # If parsing fails, skip this line
                    pass

    def read(self):
        """
        Returns the last encoder values that were read as a list of three integers.
        """
        return self.last_d
