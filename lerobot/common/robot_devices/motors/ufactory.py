import enum
import threading
import time

import numpy as np
from xarm.wrapper import XArmAPI

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0


class XArmWrapper:
    """Wrapper for the xArm Python SDK"""

    def __init__(
        self,
        port: str,
        motors: dict[str, tuple[int, str]],
        mock=False,
    ):
        print("Initializing XArmWrapper")  # Debug print
        self.port = port
        self.motors = motors
        self.mock = mock

        self.calibration = None
        self.is_connected = False
        self.logs = {}

        self.api = None

        self.MAX_SPEED_LIMIT = None
        self.MAX_ACC_LIMIT = None

        # Stop event
        self.stop_event = threading.Event()

        # Create and start the digital input monitoring thread
        print("Creating monitor thread")  # Debug print
        self.monitor_input_thread = threading.Thread(
            target=self.monitor_digital_input, args=(self.stop_event,)
        )

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def connect(self):
        print("Connecting to xArm")  # Debug print
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"DynamixelMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        if self.mock:
            print("Mock mode, not connecting to real device")  # Debug print
            return
        else:
            self.api = XArmAPI(self.port)

        try:
            if not self.api.connected:
                raise OSError(f"Failed to connect to xArm API @ '{self.port}'.")
            print("Successfully connected to xArm")  # Debug print
        except Exception as e:
            print(f"Exception while connecting in XArmWrapper: {e}")
            raise

        # Allow to read and write
        self.is_connected = True

        # Start the monitoring thread after successful connection
        self.monitor_input_thread.start()
        print("Monitor thread started")  # Debug print

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        pass  # TODO (@vmayoral): implement if of interest

    def read(self, data_name, motor_names: str | list[str] | None = None):
        pass  # TODO (@vmayoral): implement if of interest

    def enable(self, follower: bool = False):
        self.api.motion_enable(enable=True)
        self.api.clean_error()
        if follower:
            self.api.set_mode(1)
        else:
            self.api.set_mode(0)
        self.api.set_state(state=0)
        #
        self.api.set_gripper_mode(0)
        self.api.set_gripper_enable(True)
        self.api.set_gripper_speed(5000)  # default speed, as there's no way to fetch gripper speed from API

        # # Initialize the global speed and acceleration limits
        # self.initialize_limits()  # not acting as expected

        # assume leader by default
        if not follower:
            self.api.set_mode(2)
            self.api.set_state(0)
            # Light up the digital output 2 (button), to signal manual mode
            self.api.set_tgpio_digital(ionum=2, value=1)

    def disconnect(self):
        print("Disconnecting from xArm")  # Debug print
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FeetechMotorsBus({self.port}) is not connected. Try running `motors_bus.connect()` first."
            )

        # Turn off manual mode after recording
        self.api.set_mode(0)
        self.api.set_state(0)
        # Light down the digital output 2 (button), to signal manual mode
        self.api.set_tgpio_digital(ionum=2, value=0)
        # Disconnect both arms
        self.api.disconnect()

        # Stop events and threads
        self.stop_event.set()
        print("Waiting for monitor thread to join")  # Debug print
        self.monitor_input_thread.join()
        print("Monitor thread joined")  # Debug print

        # Signal as disconnected
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

    def initialize_limits(self):
        # heuristic: 1/3 of the max speed and acceleration limits
        #  for testing purposes
        self.MAX_SPEED_LIMIT = max(self.api.joint_speed_limit) / 3
        self.MAX_ACC_LIMIT = max(self.api.joint_acc_limit) / 3

    def get_position(self):
        code, angles = self.api.get_servo_angle()
        code_gripper, pos_gripper = self.api.get_gripper_position()
        # pos = angles[:-1] + [pos_gripper]  # discard 7th dof, which is not present in U850
        pos = angles + [pos_gripper]
        return pos

    def set_position(self, position: np.ndarray):
        angles = position[:-1].tolist()
        gripper_pos = int(position[-1])

        # joints
        self.api.set_servo_angle_j(angles=angles, is_radian=False, wait=False)

        # gripper
        self.api.set_gripper_position(pos=gripper_pos, wait=False)

    def monitor_digital_input(self, stop_event):
        print("Starting monitor_digital_input")  # Debug print
        single_click_time = 0.2
        double_click_time = 0.5
        long_click_time = 1.0

        last_press_time = 0
        last_click_time = 0
        long_click_detected = False
        click_count = 0
        long_click_state = True  # starts in manual mode

        while not stop_event.is_set():
            try:
                if self.api is not None and self.is_connected:
                    code, value = self.api.get_tgpio_digital(ionum=2)
                    # print(f"Digital input read: code={code}, value={value}")  # Debug print
                    if code == 0:
                        current_time = time.time()

                        if value == 1:  # Button pressed
                            if last_press_time == 0:
                                last_press_time = current_time
                            elif (
                                not long_click_detected and current_time - last_press_time >= long_click_time
                            ):
                                print("Long click detected -> Switching manual mode")
                                long_click_detected = True
                                long_click_state = not long_click_state
                                if long_click_state:
                                    self.api.set_tgpio_digital(ionum=2, value=1)
                                    # manual mode
                                    self.api.clean_error()
                                    self.api.set_mode(2)
                                    self.api.set_state(0)
                                else:
                                    self.api.set_tgpio_digital(ionum=2, value=0)
                                    # disable manual mode
                                    self.api.clean_error()
                                    self.api.set_mode(0)
                                    self.api.set_state(0)
                        else:  # Button released
                            if last_press_time != 0:
                                press_duration = current_time - last_press_time

                                if not long_click_detected:
                                    if press_duration < single_click_time:
                                        click_count += 1
                                        if click_count == 1:
                                            last_click_time = current_time
                                        elif click_count == 2:
                                            if current_time - last_click_time < double_click_time:
                                                print("Double click detected -> Open gripper")
                                                self.api.set_gripper_position(
                                                    pos=600, wait=False
                                                )  # Open gripper
                                                click_count = 0
                                            else:
                                                print("Single click detected -> Close gripper")
                                                self.api.set_gripper_position(
                                                    pos=50, wait=False
                                                )  # Close gripper
                                                click_count = 1
                                                last_click_time = current_time
                                    else:
                                        print("Single click detected -> Close gripper")
                                        self.api.set_gripper_position(pos=50, wait=False)  # Close gripper
                                        click_count = 0

                                last_press_time = 0
                                long_click_detected = False

                        # Reset click count if too much time has passed since last click
                        if click_count == 1 and current_time - last_click_time >= double_click_time:
                            print("Single click detected -> Close gripper")
                            self.api.set_gripper_position(pos=50, wait=False)  # Close gripper
                            click_count = 0

                else:
                    print("API not connected, waiting...")
                    time.sleep(1)  # Wait a bit longer before checking again
            except Exception as e:
                print(f"Error in monitor_digital_input: {e}")  # Debug print
            time.sleep(0.01)  # Check every 10ms for more precise detection
        print("Exiting monitor_digital_input")  # Debug print

    def robot_reset(self):
        """Reset the robot to a safe state"""
        self.api.set_gripper_position(pos=600, wait=True)  # Open gripper
