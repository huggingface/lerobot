import time
import traceback
from copy import deepcopy
from typing import Protocol

import numpy as np
import trossen_arm as trossen

from ..motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value


TROSSEN_ARM_MODELS = {
    "V0_LEADER": [trossen.Model.wxai_v0, trossen.StandardEndEffector.wxai_v0_leader],
    "V0_FOLLOWER": [trossen.Model.wxai_v0, trossen.StandardEndEffector.wxai_v0_follower],
}

# Default motor configuration for Trossen arms
DEFAULT_TROSSEN_MOTORS = {
    "joint_0": Motor(id=1, model="4340", norm_mode="degrees"),
    "joint_1": Motor(id=2, model="4340", norm_mode="degrees"),
    "joint_2": Motor(id=3, model="4340", norm_mode="degrees"),
    "joint_3": Motor(id=4, model="4310", norm_mode="degrees"),
    "joint_4": Motor(id=5, model="4310", norm_mode="degrees"),
    "joint_5": Motor(id=6, model="4310", norm_mode="degrees"),
}

# Control table for Trossen motors (simplified - using position control)
TROSSEN_CONTROL_TABLE = {
    "4340": {
        "Present_Position": (132, 4),  # Current position
        "Goal_Position": (116, 4),     # Target position
        "Torque_Enable": (64, 1),      # Torque on/off
        "External_Efforts": (140, 4),  # External forces
    },
    "4310": {
        "Present_Position": (132, 4),
        "Goal_Position": (116, 4),
        "Torque_Enable": (64, 1),
        "External_Efforts": (140, 4),
    }
}

# Model resolution table (assuming 12-bit encoders)
TROSSEN_RESOLUTION_TABLE = {
    "4340": 4096,
    "4310": 4096,
}

# Normalized data that should be scaled
TROSSEN_NORMALIZED_DATA = ["Goal_Position", "Present_Position"]


class TrossenPortHandler(Protocol):
    """Protocol for Trossen port handler interface."""
    def is_open(self) -> bool: ...
    def openPort(self) -> bool: ...
    def closePort(self) -> None: ...
    def clearPort(self) -> None: ...
    def setBaudRate(self, baudrate: int) -> bool: ...
    def getBaudRate(self) -> int: ...
    def setPacketTimeout(self, packet_length: int) -> None: ...
    def setPacketTimeoutMillis(self, msec: float) -> None: ...
    def isPacketTimeout(self) -> bool: ...
    def getCurrentTime(self) -> float: ...
    def getTimeSinceStart(self) -> float: ...
    is_using: bool


class TrossenArmDriver(MotorsBus):
    """
    The `TrossenArmDriver` class provides an interface for controlling
    Trossen Robotics' robotic arms. It leverages the trossen_arm for communication with arms.

    This class allows for configuration, torque management, and motion control of robotic arms. It includes features for handling connection states, moving the
    arm to specified poses, and logging timestamps for debugging and performance analysis.

    ### Key Features:
    - **Multi-motor Control:** Supports multiple motors connected to a bus.
    - **Mode Switching:** Enables switching between position and gravity compensation modes.
    - **Home and Sleep Pose Management:** Automatically transitions the arm to home and sleep poses for safe operation.
    - **Error Handling:** Raises specific exceptions for connection and operational errors.
    - **Logging:** Captures timestamps for operations to aid in debugging.

    ### Example Usage:
    ```python
    motors = {
        "joint_0": Motor(id=1, model="4340", norm_mode="degrees"),
        "joint_1": Motor(id=2, model="4340", norm_mode="degrees"),
        "joint_2": Motor(id=3, model="4340", norm_mode="degrees"),
        "joint_3": Motor(id=4, model="4310", norm_mode="degrees"),
        "joint_4": Motor(id=5, model="4310", norm_mode="degrees"),
        "joint_5": Motor(id=6, model="4310", norm_mode="degrees"),
    }
    arm_driver = TrossenArmDriver(
        port="192.168.1.2",
        motors=motors,
        model="V0_LEADER",
    )
    arm_driver.connect()

    # Read motor positions
    positions = arm_driver.read("Present_Position", "joint_0")

    # Move to a new position (Home Pose)
    # Last joint is the gripper, which is in range [0, 450]
    arm_driver.write("Goal_Position", "joint_0", 0)

    # Disconnect when done
    arm_driver.disconnect()
    ```
    """

    # Required class attributes for MotorsBus
    apply_drive_mode = False
    available_baudrates = [1000000]  # Trossen uses network communication, not serial
    default_baudrate = 1000000
    default_timeout = 1000
    model_baudrate_table = {"4340": {1000000: 1}, "4310": {1000000: 1}}
    model_ctrl_table = deepcopy(TROSSEN_CONTROL_TABLE)
    model_encoding_table = {"4340": {}, "4310": {}}  # No special encoding needed
    model_number_table = {"4340": 4340, "4310": 4310}
    model_resolution_table = deepcopy(TROSSEN_RESOLUTION_TABLE)
    normalized_data = deepcopy(TROSSEN_NORMALIZED_DATA)

    def __init__(
        self,
        port: str,  # This will be the IP address for Trossen
        motors: dict[str, Motor] | None = None,
        calibration: dict[str, MotorCalibration] | None = None,
        model: str = "V0_LEADER",
        mock: bool = False,
    ):
        # Use default motors if none provided
        if motors is None:
            motors = deepcopy(DEFAULT_TROSSEN_MOTORS)
        
        # Initialize the base class
        super().__init__(port, motors, calibration)
        
        # Trossen-specific attributes
        self.model = model
        self.mock = mock
        self.driver = None
        self._is_connected = False
        self.logs = {}
        self.fps = 30
        self.home_pose = [0, np.pi / 3, np.pi / 6, np.pi / 5, 0, 0]
        self.sleep_pose = [0, 0, 0, 0, 0, 0]

        # Minimum time to move for the arm
        self.MIN_TIME_TO_MOVE = 3.0 / self.fps

    @property
    def is_connected(self) -> bool:
        """Check if the Trossen arm is connected."""
        return self._is_connected and self.driver is not None

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        """Trossen arms don't use traditional protocols, so this is always compatible."""
        pass

    def _handshake(self) -> None:
        """Perform handshake with Trossen arm."""
        self._assert_motors_exist()
        # For Trossen, we just verify the driver is configured
        if self.driver is None:
            raise ConnectionError("Trossen driver not initialized")

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        """Find a single motor on the Trossen arm."""
        # For Trossen arms, motors are pre-configured
        motor_id = self.motors[motor].id
        model_number = self.model_number_table[self.motors[motor].model]
        return self.default_baudrate, motor_id

    def configure_motors(self) -> None:
        """Configure Trossen motors with recommended settings."""
        # Trossen motors are pre-configured, but we can set some parameters
        for motor_name in self.motors:
            # Set torque enable to 1 (enabled)
            self.write("Torque_Enable", motor_name, 1, normalize=False)

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Disable torque on selected motors."""
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, 0, normalize=False)

    def _disable_torque(self, motor_id: int, model: str, num_retry: int = 0) -> None:
        """Disable torque on a specific motor by ID."""
        motor_name = self._id_to_name(motor_id)
        self.disable_torque([motor_name], num_retry)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Enable torque on selected motors."""
        for motor in self._get_motors_list(motors):
            self.write("Torque_Enable", motor, 1, normalize=False)

    @property
    def is_calibrated(self) -> bool:
        """Check if motors are calibrated."""
        # For Trossen arms, we assume they come pre-calibrated
        return True

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Read calibration from Trossen motors."""
        calibration = {}
        for motor_name, motor in self.motors.items():
            # For Trossen, we use default calibration ranges
            model = motor.model
            max_res = self.model_resolution_table[model] - 1
            calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=0,
                range_min=0,
                range_max=max_res,
            )
        return calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration]) -> None:
        """Write calibration to Trossen motors."""
        # Trossen arms don't support calibration writing
        # We just store it in memory
        self.calibration = calibration_dict

    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        """Calculate half-turn homing offsets for Trossen motors."""
        half_turn_homings = {}
        for motor, pos in positions.items():
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1
            half_turn_homings[motor] = pos - int(max_res / 2)
        return half_turn_homings

    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        """Encode sign for Trossen motors."""
        # Trossen motors don't need special sign encoding
        return ids_values

    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        """Decode sign for Trossen motors."""
        # Trossen motors don't need special sign decoding
        return ids_values

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        """Split value into byte chunks for Trossen communication."""
        if length == 1:
            return [value & 0xFF]
        elif length == 2:
            return [value & 0xFF, (value >> 8) & 0xFF]
        elif length == 4:
            return [
                value & 0xFF,
                (value >> 8) & 0xFF,
                (value >> 16) & 0xFF,
                (value >> 24) & 0xFF,
            ]
        else:
            raise ValueError(f"Unsupported length: {length}")

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        """Ping all motors on the Trossen arm."""
        try:
            # For Trossen, we return the expected motor IDs and models
            result = {}
            for motor_name, motor in self.motors.items():
                result[motor.id] = self.model_number_table[motor.model]
            return result
        except Exception as e:
            if raise_on_error:
                raise ConnectionError(f"Failed to ping Trossen arm: {e}")
            return None

    def connect(self, handshake: bool = True) -> None:
        """Connect to the Trossen arm."""
        if self.is_connected:
            print(f"TrossenArmDriver({self.port}) is already connected.")
            return

        print(f"Connecting to {self.model} arm at {self.port}...")

        # Initialize the driver
        self.driver = trossen.TrossenArmDriver()

        # Get the model configuration
        try:
            model_name, model_end_effector = TROSSEN_ARM_MODELS[self.model]
        except KeyError as e:
            raise ValueError(f"Unsupported model: {self.model}") from e

        print("Configuring the drivers...")

        # Configure the driver
        try:
            self.driver.configure(model_name, model_end_effector, self.port, True)
        except Exception:
            traceback.print_exc()
            print(f"Failed to configure the driver for the {self.model} arm at {self.port}.")
            raise

        # Move the arms to the home pose
        self.driver.set_all_modes(trossen.Mode.position)
        self.driver.set_all_positions(self.home_pose, 2.0, False)

        # Mark as connected
        self._is_connected = True

    def reconnect(self):
        """Reconnect to the Trossen arm."""
        try:
            model_name, model_end_effector = TROSSEN_ARM_MODELS[self.model]
        except KeyError as e:
            raise ValueError(f"Unsupported model: {self.model}") from e
        try:
            self.driver.configure(model_name, model_end_effector, self.port, True)
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to configure the driver for the {self.model} arm at {self.port}.")
            raise e

        self._is_connected = True

    def read(self, data_name: str, motor: str, *, normalize: bool = True, num_retry: int = 0) -> Value:
        """Read data from a Trossen motor."""
        if not self.is_connected:
            print(f"TrossenArmDriver({self.port}) is not connected. You need to run `connect()`.")
            return None

        start_time = time.perf_counter()

        # Read the present position of the motors
        if data_name == "Present_Position":
            # Get the positions of the motors
            values = self.driver.get_all_positions()
            # Return the specific motor's position
            motor_id = self.motors[motor].id
            motor_index = motor_id - 1  # Assuming IDs start at 1
            if motor_index < len(values):
                value = values[motor_index]
            else:
                value = 0.0
        elif data_name == "External_Efforts":
            values = self.driver.get_all_external_efforts()
            motor_id = self.motors[motor].id
            motor_index = motor_id - 1
            if motor_index < len(values):
                value = values[motor_index]
            else:
                value = 0.0
        else:
            value = 0.0
            print(f"Data name: {data_name} is not supported for reading.")

        self.logs["delta_timestamp_s_read"] = time.perf_counter() - start_time

        # Convert to the expected format
        if normalize and data_name in self.normalized_data:
            # Convert radians to degrees for normalization
            value = np.degrees(value)
        
        return value

    def write(self, data_name: str, motor: str, value: Value, *, normalize: bool = True, num_retry: int = 0) -> None:
        """Write data to a Trossen motor."""
        if not self.is_connected:
            print(f"TrossenArmDriver({self.port}) is not connected. You need to run `connect()`.")
            return

        start_time = time.perf_counter()

        # Write the goal position of the motors
        if data_name == "Goal_Position":
            if normalize:
                # Convert degrees to radians
                value = np.radians(value)
            
            # For Trossen, we need to set all positions at once
            # Get current positions
            current_positions = self.driver.get_all_positions()
            motor_id = self.motors[motor].id
            motor_index = motor_id - 1
            
            # Update the specific motor's position
            if motor_index < len(current_positions):
                current_positions[motor_index] = value
                self.driver.set_all_positions(current_positions, self.MIN_TIME_TO_MOVE, False)

        # Enable or disable the torque of the motors
        elif data_name == "Torque_Enable":
            # Set the arms to POSITION mode
            if value == 1:
                self.driver.set_all_modes(trossen.Mode.position)
            else:
                self.driver.set_all_modes(trossen.Mode.external_effort)
                self.driver.set_all_external_efforts([0.0] * self.driver.get_num_joints(), 0.0, True)
        elif data_name == "Reset":
            self.driver.set_all_modes(trossen.Mode.velocity)
            self.driver.set_all_velocities([0.0] * self.driver.get_num_joints(), 0.0, False)
            self.driver.set_all_modes(trossen.Mode.position)
            self.driver.set_all_positions(self.home_pose, 2.0, False)
        elif data_name == "External_Efforts":
            # For Trossen, we need to set all efforts at once
            current_efforts = self.driver.get_all_external_efforts()
            motor_id = self.motors[motor].id
            motor_index = motor_id - 1
            if motor_index < len(current_efforts):
                current_efforts[motor_index] = value
                self.driver.set_all_external_efforts(current_efforts, 0.0, False)
        else:
            print(f"Data name: {data_name} value: {value} is not supported for writing.")

        self.logs["delta_timestamp_s_write"] = time.perf_counter() - start_time

    def sync_read(self, data_name: str, motors: str | list[str] | None = None, *, normalize: bool = True, num_retry: int = 0) -> dict[str, Value]:
        """Synchronized read from multiple Trossen motors."""
        if not self.is_connected:
            print(f"TrossenArmDriver({self.port}) is not connected. You need to run `connect()`.")
            return {}

        motors_list = self._get_motors_list(motors)
        result = {}

        if data_name == "Present_Position":
            # Get all positions at once
            all_positions = self.driver.get_all_positions()
            
            for motor in motors_list:
                motor_id = self.motors[motor].id
                motor_index = motor_id - 1
                if motor_index < len(all_positions):
                    value = all_positions[motor_index]
                    if normalize:
                        value = np.degrees(value)
                    result[motor] = value
                else:
                    result[motor] = 0.0
        else:
            # Fall back to individual reads for other data types
            for motor in motors_list:
                result[motor] = self.read(data_name, motor, normalize=normalize, num_retry=num_retry)

        return result

    def sync_write(self, data_name: str, values: Value | dict[str, Value], *, normalize: bool = True, num_retry: int = 0) -> None:
        """Synchronized write to multiple Trossen motors."""
        if not self.is_connected:
            print(f"TrossenArmDriver({self.port}) is not connected. You need to run `connect()`.")
            return

        if data_name == "Goal_Position":
            # Get current positions
            current_positions = self.driver.get_all_positions()
            
            # Update positions for specified motors
            if isinstance(values, dict):
                for motor, value in values.items():
                    motor_id = self.motors[motor].id
                    motor_index = motor_id - 1
                    if motor_index < len(current_positions):
                        if normalize:
                            value = np.radians(value)
                        current_positions[motor_index] = value
            else:
                # If single value, apply to all motors
                if normalize:
                    values = np.radians(values)
                current_positions = [values] * len(current_positions)
            
            # Set all positions at once
            self.driver.set_all_positions(current_positions, self.MIN_TIME_TO_MOVE, False)
        else:
            # Fall back to individual writes for other data types
            if isinstance(values, dict):
                for motor, value in values.items():
                    self.write(data_name, motor, value, normalize=normalize, num_retry=num_retry)
            else:
                for motor in self.motors:
                    self.write(data_name, motor, values, normalize=normalize, num_retry=num_retry)

    def disconnect(self, disable_torque: bool = True) -> None:
        """Disconnect from the Trossen arm."""
        if not self.is_connected:
            print(f"TrossenArmDriver ({self.port}) is not connected.")
            return

        if disable_torque:
            self.driver.set_all_modes(trossen.Mode.velocity)
            self.driver.set_all_velocities([0.0] * self.driver.get_num_joints(), 0.0, False)
            self.driver.set_all_modes(trossen.Mode.position)
            self.driver.set_all_positions(self.home_pose, 2.0, True)
            self.driver.set_all_positions(self.sleep_pose, 2.0, False)

        self._is_connected = False

    def __del__(self):
        """Cleanup when the object is destroyed."""
        if getattr(self, "_is_connected", False):
            self.disconnect()