
from lerobot.common.robot_devices.motors.feetech import (
    CalibrationMode,
    FeetechMotorsBus,
)
import yaml
from lerobot.common.robot_devices.motors.feetech import (
    CalibrationMode,
    FeetechMotorsBus,
)
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera,OpenCVCameraConfig
import torch
import serial
import threading
import time
import pickle
import numpy as np
from collections import deque
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

class HopeJrRobot:
    robot_type = "hopejr"
    def __init__(self, cameras=None):
        self.arm_port = "/dev/tty.usbmodem585A0082361"
        self.hand_port = "/dev/tty.usbmodem58760436961"
        self.cameras = {}
        config = OpenCVCameraConfig(fps=60, width=1920, height=1080)
        self.cameras["camera1"] = OpenCVCamera(1,config=config)
        self.has_camera = True
        self.num_cameras = 1
        self.observation_dict = {}
        self.arm_bus = FeetechMotorsBus(
            port = self.arm_port,
            motors={
                # "motor1": (1, "sts3250"),
                # "motor2": (2, "sts3250"),
                # "motor3": (3, "sts3250"),
                
                #"shoulder_pitch": [1, "sts3215"],
                "shoulder_pitch": [1, "sm8512bl"],
                "shoulder_yaw": [2, "sts3250"],  # TODO: sts3250
                "shoulder_roll": [3, "sts3250"],  # TODO: sts3250
                "elbow_flex": [4, "sts3250"],
                "wrist_roll": [5, "sts3215"],
                "wrist_yaw": [6, "sts3215"],
                "wrist_pitch": [7, "sts3215"],
            },
            protocol_version=0,
        )
        self.hand_bus = FeetechMotorsBus(
            port=self.hand_port,

        motors = {
            # Thumb
            "thumb_basel_rotation": [1, "scs0009"],
            "thumb_mcp": [3, "scs0009"],
            "thumb_pip": [4, "scs0009"],
            "thumb_dip": [13, "scs0009"],

            # Index
            "index_thumb_side": [5, "scs0009"],
            "index_pinky_side": [6, "scs0009"],
            "index_flexor": [16, "scs0009"],

            # Middle
            "middle_thumb_side": [8, "scs0009"],
            "middle_pinky_side": [9, "scs0009"],
            "middle_flexor": [2, "scs0009"],

            # Ring
            "ring_thumb_side": [11, "scs0009"],
            "ring_pinky_side": [12, "scs0009"],
            "ring_flexor": [7, "scs0009"],

            # Pinky
            "pinky_thumb_side": [14, "scs0009"],
            "pinky_pinky_side": [15, "scs0009"],
            "pinky_flexor": [10, "scs0009"],
        },
            protocol_version=1,#1
            group_sync_read=False,
        )

        self.arm_calib_dict = self.get_arm_calibration()
        self.hand_calib_dict = self.get_hand_calibration()
        self.is_connected = False
        #init self.exoskeleton as an empty 
        self.exoskeleton = None
        self.glove = None

    def apply_arm_config(self, config_file):
        #with open(config_file, "r") as file:
        #    config = yaml.safe_load(file)
        #for param, value in config.get("robot", {}).get("arm_bus", {}).items():
        #    self.arm_bus.write(param, value)
        return
    def apply_hand_config(config_file, robot):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        for param, value in config.get("robot", {}).get("hand_bus", {}).items():
            robot.arm_bus.write(param, value)

    def get_hand_calibration(self):
        homing_offset = [0] * len(self.hand_bus.motor_names)
        drive_mode = [0] * len(self.hand_bus.motor_names)
        
        start_pos = [
            750,  # thumb_basel_rotation
            100,  # thumb_mcp
            700,  # thumb_pip
            100,  # thumb_dip

            800,  # index_thumb_side
            950,  # index_pinky_side
            0,  # index_flexor

            250,  # middle_thumb_side
            850,  # middle_pinky_side
            0,  # middle_flexor

            850,  # ring_thumb_side
            900,  # ring_pinky_side
            0,  # ring_flexor

            00,  # pinky_thumb_side
            950,  # pinky_pinky_side
            0,  # pinky_flexor
        ]

        end_pos = [
            start_pos[0] - 550,  # thumb_basel_rotation
            start_pos[1] + 400,  # thumb_mcp
            start_pos[2] + 300,  # thumb_pip
            start_pos[3] + 200,  # thumb_dip

            start_pos[4] - 700,  # index_thumb_side
            start_pos[5] - 300,  # index_pinky_side
            start_pos[6] + 600,  # index_flexor

            start_pos[7] + 700,  # middle_thumb_side
            start_pos[8] - 400,  # middle_pinky_side
            start_pos[9] + 600,  # middle_flexor

            start_pos[10] - 600,  # ring_thumb_side
            start_pos[11] - 400,  # ring_pinky_side
            start_pos[12] + 600,  # ring_flexor

            start_pos[13] + 400,  # pinky_thumb_side
            start_pos[14] - 450,  # pinky_pinky_side
            start_pos[15] + 600,  # pinky_flexor
        ]


        

        calib_modes = [CalibrationMode.LINEAR.name] * len(self.hand_bus.motor_names)

        calib_dict = {
            "homing_offset": homing_offset,
            "drive_mode": drive_mode,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "calib_mode": calib_modes,
            "motor_names": self.hand_bus.motor_names,
        }
        return calib_dict
    
    def get_arm_calibration(self):

        homing_offset = [0] * len(self.arm_bus.motor_names)
        drive_mode = [0] * len(self.arm_bus.motor_names)

        start_pos = [
            1800,   # shoulder_up
            2800,  # shoulder_forward
            1800,  # shoulder_roll
            1200,  # bend_elbow
            700,  # wrist_roll
            1850,  # wrist_yaw
            1700,  # wrist_pitch
        ]

        end_pos = [
            2800,  # shoulder_up
            3150,  # shoulder_forward
            400,  #shoulder_roll
            2300,  # bend_elbow
            2300,  # wrist_roll
            2150,  # wrist_yaw
            2300,  # wrist_pitch
        ]

        calib_modes = [CalibrationMode.LINEAR.name] * len(self.arm_bus.motor_names)

        calib_dict = {
            "homing_offset": homing_offset,
            "drive_mode": drive_mode,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "calib_mode": calib_modes,
            "motor_names": self.arm_bus.motor_names,
        }
        return calib_dict
    
    def connect(self):
        self.arm_bus.connect()
        self.hand_bus.connect()
        #read pos
        print(self.hand_bus.read("Present_Position"))
        #move arm elbow to half way
        self.arm_bus.write("Goal_Position", 2000, "elbow_flex")
        self.arm_bus.write("Goal_Position", 700, "wrist_roll")        
        self.arm_bus.write("Goal_Position", 1150, "shoulder_roll")
        #print(self.arm_bus.read("Present_Position", "shoulder_pitch"))
        #print(self.arm_bus.read("Present_Position",["shoulder_yaw","shoulder_roll","elbow_flex","wrist_roll","wrist_yaw","wrist_pitch"]))
        #robot.arm_bus.write("Goal_Position", robot.arm_calib_dict["start_pos"][0]*1 +robot.arm_calib_dict["end_pos"][0]*0, ["wrist_roll"])
        for i in range(10):
            time.sleep(0.1)
            #self.apply_arm_config('examples/hopejr/settings/config.yaml')

        # #calibrate arm
        arm_calibration = self.get_arm_calibration()
        self.exoskeleton = HomonculusArm(serial_port="/dev/tty.usbmodem1201")

        calibrate_exoskeleton = False
        if calibrate_exoskeleton:   
            self.exoskeleton.run_calibration(self)

        file_path = "examples/hopejr/settings/arm_calib.pkl"
        with open(file_path, "rb") as f:
            calib_dict = pickle.load(f)
        print("Loaded dictionary:", calib_dict)
        self.exoskeleton.set_calibration(calib_dict)

        #calibrate hand
        hand_calibration = self.get_hand_calibration()
        self.glove = HomonculusGlove(serial_port = "/dev/tty.usbmodem1101")
        calibrate_glove = False
        if calibrate_glove:
            self.glove.run_calibration()
            
        file_path = "examples/hopejr/settings/hand_calib.pkl"
        with open(file_path, "rb") as f:
            calib_dict = pickle.load(f)
        print("Loaded dictionary:", calib_dict)
        self.glove.set_calibration(calib_dict)

        self.hand_bus.set_calibration(hand_calibration)
        #self.arm_bus.set_calibration(arm_calibration)

        for camera in self.cameras.values():
            camera.connect()

        self.is_connected = True
    
    def run_calibration(self): ...

    def teleop_step(self, record_data=False):
        self.apply_arm_config('examples/hopejr/settings/config.yaml')
        #robot.arm_bus.write("Acceleration", 50, "shoulder_yaw")
        joint_names = ["shoulder_pitch", "shoulder_yaw", "shoulder_roll", "elbow_flex", "wrist_roll", "wrist_yaw", "wrist_pitch"]
        #only wrist roll
        #joint_names = ["shoulder_pitch"]
        joint_values = self.exoskeleton.read(motor_names=joint_names)

        #joint_values = joint_values.round().astype(int)
        joint_dict = {k: v for k, v in zip(joint_names, joint_values, strict=False)}

        motor_values = []
        motor_names = []
        motor_names += ["shoulder_pitch", "shoulder_yaw", "shoulder_roll", "elbow_flex", "wrist_roll", "wrist_yaw", "wrist_pitch"]
        #motor_names += ["shoulder_pitch"]
        motor_values += [joint_dict[name] for name in motor_names]
        #remove 50 from shoulder_roll
        #motor_values += [joint_dict[name] for name in motor_names]

        motor_values = np.array(motor_values)
        motor_values = np.clip(motor_values, 0, 100)

        print(motor_names, motor_values)
        freeze_arm = True
        if not freeze_arm:
            self.robot.arm_bus.write("Goal_Position", motor_values, motor_names)
        freeze_fingers = False
        if not freeze_fingers:#include hand
            hand_joint_names = []
            hand_joint_names += ["thumb_3", "thumb_2", "thumb_1", "thumb_0"]#, "thumb_3"]
            hand_joint_names += ["index_0", "index_1", "index_2"]
            hand_joint_names += ["middle_0", "middle_1", "middle_2"]
            hand_joint_names += ["ring_0", "ring_1", "ring_2"]
            hand_joint_names += ["pinky_0", "pinky_1", "pinky_2"]
            hand_joint_values = self.glove.read(hand_joint_names)
            hand_joint_values = hand_joint_values.round( ).astype(int)
            hand_joint_dict = {k: v for k, v in zip(hand_joint_names, hand_joint_values, strict=False)}

            hand_motor_values = []
            hand_motor_names = []

            # Thumb
            hand_motor_names += ["thumb_basel_rotation", "thumb_mcp", "thumb_pip", "thumb_dip"]#, "thumb_MCP"]
            hand_motor_values += [
                hand_joint_dict["thumb_3"],
                hand_joint_dict["thumb_2"],
                hand_joint_dict["thumb_1"],
                hand_joint_dict["thumb_0"]
            ]

            # # Index finger
            index_splay = 0.1
            hand_motor_names += ["index_flexor", "index_pinky_side", "index_thumb_side"]
            hand_motor_values += [
                hand_joint_dict["index_2"],
                (100 - hand_joint_dict["index_0"]) * index_splay + hand_joint_dict["index_1"] * (1 - index_splay),
                hand_joint_dict["index_0"] * index_splay + hand_joint_dict["index_1"] * (1 - index_splay),
            ]

            # Middle finger
            middle_splay = 0.1
            hand_motor_names += ["middle_flexor", "middle_pinky_side", "middle_thumb_side"]
            hand_motor_values += [
                hand_joint_dict["middle_2"],
                hand_joint_dict["middle_0"] * middle_splay + hand_joint_dict["middle_1"] * (1 - middle_splay),
                (100 - hand_joint_dict["middle_0"]) * middle_splay + hand_joint_dict["middle_1"] * (1 - middle_splay),
            ]

            # # Ring finger
            ring_splay = 0.1
            hand_motor_names += ["ring_flexor", "ring_pinky_side", "ring_thumb_side"]
            hand_motor_values += [
                hand_joint_dict["ring_2"],
                (100 - hand_joint_dict["ring_0"]) * ring_splay + hand_joint_dict["ring_1"] * (1 - ring_splay),
                hand_joint_dict["ring_0"] * ring_splay + hand_joint_dict["ring_1"] * (1 - ring_splay),
            ]

            # # Pinky finger
            pinky_splay = -.1
            hand_motor_names += ["pinky_flexor", "pinky_pinky_side", "pinky_thumb_side"]
            hand_motor_values += [
                hand_joint_dict["pinky_2"],
                hand_joint_dict["pinky_0"] * pinky_splay + hand_joint_dict["pinky_1"] * (1 - pinky_splay),
                (100 - hand_joint_dict["pinky_0"]) * pinky_splay + hand_joint_dict["pinky_1"] * (1 - pinky_splay),
                ]

            hand_motor_values = np.array(hand_motor_values)
            hand_motor_values = np.clip(hand_motor_values, 0, 100)
            self.hand_bus.write("Acceleration", 255, hand_motor_names)
            self.hand_bus.write("Goal_Position", hand_motor_values, hand_motor_names)
        
        all_pos = np.concatenate((motor_values, hand_motor_values))
        all_pos = torch.tensor(all_pos, dtype=torch.float32)
        obs = {"observation.state": all_pos}
        action = {"action":all_pos}

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            # self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            # self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        for name in self.cameras:
            obs[f"observation.images.{name}"] = images[name]

        return obs, action

    def send_action(self, action):
        #self.arm_bus.write("Goal_Position", action[:7], self.arm_bus.motor_names)
        self.hand_bus.write("Goal_Position", action[7:], self.hand_bus.motor_names)
        
    def disconnect(self):
            self.arm_bus.disconnect()
            self.hand_bus.disconnect()
            self.is_connected = False
    
    def capture_observation(self): 
        self.apply_arm_config('examples/hopejr/settings/config.yaml')
        #robot.arm_bus.write("Acceleration", 50, "shoulder_yaw")
        joint_names = ["shoulder_pitch", "shoulder_yaw", "shoulder_roll", "elbow_flex", "wrist_roll", "wrist_yaw", "wrist_pitch"]
        #only wrist roll
        #joint_names = ["shoulder_pitch"]
        joint_values = self.exoskeleton.read(motor_names=joint_names)

        #joint_values = joint_values.round().astype(int)
        joint_dict = {k: v for k, v in zip(joint_names, joint_values, strict=False)}

        motor_values = []
        motor_names = []
        motor_names += ["shoulder_pitch", "shoulder_yaw", "shoulder_roll", "elbow_flex", "wrist_roll", "wrist_yaw", "wrist_pitch"]
        #motor_names += ["shoulder_pitch"]
        motor_values += [joint_dict[name] for name in motor_names]
        #remove 50 from shoulder_roll
        #motor_values += [joint_dict[name] for name in motor_names]

        motor_values = np.array(motor_values)
        motor_values = np.clip(motor_values, 0, 100)

        print(motor_names, motor_values)
        freeze_fingers = False
        if not freeze_fingers:#include hand
            hand_joint_names = []
            hand_joint_names += ["thumb_3", "thumb_2", "thumb_1", "thumb_0"]#, "thumb_3"]
            hand_joint_names += ["index_0", "index_1", "index_2"]
            hand_joint_names += ["middle_0", "middle_1", "middle_2"]
            hand_joint_names += ["ring_0", "ring_1", "ring_2"]
            hand_joint_names += ["pinky_0", "pinky_1", "pinky_2"]
            hand_joint_values = self.glove.read(hand_joint_names)
            hand_joint_values = hand_joint_values.round( ).astype(int)
            hand_joint_dict = {k: v for k, v in zip(hand_joint_names, hand_joint_values, strict=False)}

            hand_motor_values = []
            hand_motor_names = []

            # Thumb
            hand_motor_names += ["thumb_basel_rotation", "thumb_mcp", "thumb_pip", "thumb_dip"]#, "thumb_MCP"]
            hand_motor_values += [
                hand_joint_dict["thumb_3"],
                hand_joint_dict["thumb_2"],
                hand_joint_dict["thumb_1"],
                hand_joint_dict["thumb_0"]
            ]

            # # Index finger
            index_splay = 0.1
            hand_motor_names += ["index_flexor", "index_pinky_side", "index_thumb_side"]
            hand_motor_values += [
                hand_joint_dict["index_2"],
                (100 - hand_joint_dict["index_0"]) * index_splay + hand_joint_dict["index_1"] * (1 - index_splay),
                hand_joint_dict["index_0"] * index_splay + hand_joint_dict["index_1"] * (1 - index_splay),
            ]

            # Middle finger
            middle_splay = 0.1
            hand_motor_names += ["middle_flexor", "middle_pinky_side", "middle_thumb_side"]
            hand_motor_values += [
                hand_joint_dict["middle_2"],
                hand_joint_dict["middle_0"] * middle_splay + hand_joint_dict["middle_1"] * (1 - middle_splay),
                (100 - hand_joint_dict["middle_0"]) * middle_splay + hand_joint_dict["middle_1"] * (1 - middle_splay),
            ]

            # # Ring finger
            ring_splay = 0.1
            hand_motor_names += ["ring_flexor", "ring_pinky_side", "ring_thumb_side"]
            hand_motor_values += [
                hand_joint_dict["ring_2"],
                (100 - hand_joint_dict["ring_0"]) * ring_splay + hand_joint_dict["ring_1"] * (1 - ring_splay),
                hand_joint_dict["ring_0"] * ring_splay + hand_joint_dict["ring_1"] * (1 - ring_splay),
            ]

            # # Pinky finger
            pinky_splay = -.1
            hand_motor_names += ["pinky_flexor", "pinky_pinky_side", "pinky_thumb_side"]
            hand_motor_values += [
                hand_joint_dict["pinky_2"],
                hand_joint_dict["pinky_0"] * pinky_splay + hand_joint_dict["pinky_1"] * (1 - pinky_splay),
                (100 - hand_joint_dict["pinky_0"]) * pinky_splay + hand_joint_dict["pinky_1"] * (1 - pinky_splay),
                ]

            hand_motor_values = np.array(hand_motor_values)
            hand_motor_values = np.clip(hand_motor_values, 0, 100)

        all_pos = np.concatenate((motor_values, hand_motor_values))
        all_pos = torch.tensor(all_pos, dtype=torch.float32)
        obs = {"observation.state": all_pos}

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            # self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            # self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        for name in self.cameras:
            obs[f"observation.images.{name}"] = images[name]

        return obs