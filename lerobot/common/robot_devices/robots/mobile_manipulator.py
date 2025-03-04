import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import zmq

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.feetech import TorqueMode
from lerobot.common.robot_devices.motors.utils import MotorsBus, make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.configs import LeKiwiRobotConfig
from lerobot.common.robot_devices.robots.feetech_calibration import run_arm_manual_calibration
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceNotConnectedError

PYNPUT_AVAILABLE = True
try:
    # Only import if there's a valid X server or if we're not on a Pi
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        print("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    print(f"Could not import pynput: {e}")


class MobileManipulator:
    """
    MobileManipulator is a class for connecting to and controlling a remote mobile manipulator robot.
    The robot includes a three omniwheel mobile base and a remote follower arm.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    def __init__(self, config: LeKiwiRobotConfig):
        """
        Expected keys in config:
          - ip, port, video_port for the remote connection.
          - calibration_dir, leader_arms, follower_arms, max_relative_target, etc.
        """
        self.robot_type = config.type
        self.config = config
        self.remote_ip = config.ip
        self.remote_port = config.port
        self.remote_port_video = config.video_port
        self.calibration_dir = Path(self.config.calibration_dir)
        self.logs = {}

        self.teleop_keys = self.config.teleop_keys

        # For teleoperation, the leader arm (local) is used to record the desired arm pose.
        self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)

        self.follower_arms = make_motors_buses_from_configs(self.config.follower_arms)

        self.cameras = make_cameras_from_configs(self.config.cameras)

        self.is_connected = False

        self.last_frames = {}
        self.last_present_speed = {}
        self.last_remote_arm_state = torch.zeros(6, dtype=torch.float32)

        # Define three speed levels and a current index
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]
        self.speed_index = 0  # Start at slow

        # ZeroMQ context and sockets.
        self.context = None
        self.cmd_socket = None
        self.video_socket = None

        # Keyboard state for base teleoperation.
        self.running = True
        self.pressed_keys = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "rotate_left": False,
            "rotate_right": False,
        }

        if PYNPUT_AVAILABLE:
            print("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release,
            )
            self.listener.start()
        else:
            print("pynput not available - skipping local keyboard listener.")
            self.listener = None

    def get_motor_names(self, arms: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arms.items() for motor in bus.motors]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        follower_arm_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        observations = ["x_mm", "y_mm", "theta"]
        combined_names = follower_arm_names + observations
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(combined_names),),
                "names": combined_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(combined_names),),
                "names": combined_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available = []
        for name in self.leader_arms:
            available.append(get_arm_id(name, "leader"))
        for name in self.follower_arms:
            available.append(get_arm_id(name, "follower"))
        return available

    def on_press(self, key):
        try:
            # Movement
            if key.char == self.teleop_keys["forward"]:
                self.pressed_keys["forward"] = True
            elif key.char == self.teleop_keys["backward"]:
                self.pressed_keys["backward"] = True
            elif key.char == self.teleop_keys["left"]:
                self.pressed_keys["left"] = True
            elif key.char == self.teleop_keys["right"]:
                self.pressed_keys["right"] = True
            elif key.char == self.teleop_keys["rotate_left"]:
                self.pressed_keys["rotate_left"] = True
            elif key.char == self.teleop_keys["rotate_right"]:
                self.pressed_keys["rotate_right"] = True

            # Quit teleoperation
            elif key.char == self.teleop_keys["quit"]:
                self.running = False
                return False

            # Speed control
            elif key.char == self.teleop_keys["speed_up"]:
                self.speed_index = min(self.speed_index + 1, 2)
                print(f"Speed index increased to {self.speed_index}")
            elif key.char == self.teleop_keys["speed_down"]:
                self.speed_index = max(self.speed_index - 1, 0)
                print(f"Speed index decreased to {self.speed_index}")

        except AttributeError:
            # e.g., if key is special like Key.esc
            if key == keyboard.Key.esc:
                self.running = False
                return False

    def on_release(self, key):
        try:
            if hasattr(key, "char"):
                if key.char == self.teleop_keys["forward"]:
                    self.pressed_keys["forward"] = False
                elif key.char == self.teleop_keys["backward"]:
                    self.pressed_keys["backward"] = False
                elif key.char == self.teleop_keys["left"]:
                    self.pressed_keys["left"] = False
                elif key.char == self.teleop_keys["right"]:
                    self.pressed_keys["right"] = False
                elif key.char == self.teleop_keys["rotate_left"]:
                    self.pressed_keys["rotate_left"] = False
                elif key.char == self.teleop_keys["rotate_right"]:
                    self.pressed_keys["rotate_right"] = False
        except AttributeError:
            pass

    def connect(self):
        if not self.leader_arms:
            raise ValueError("MobileManipulator has no leader arm to connect.")
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.calibrate_leader()

        # Set up ZeroMQ sockets to communicate with the remote mobile robot.
        self.context = zmq.Context()
        self.cmd_socket = self.context.socket(zmq.PUSH)
        connection_string = f"tcp://{self.remote_ip}:{self.remote_port}"
        self.cmd_socket.connect(connection_string)
        self.cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.video_socket = self.context.socket(zmq.PULL)
        video_connection = f"tcp://{self.remote_ip}:{self.remote_port_video}"
        self.video_socket.connect(video_connection)
        self.video_socket.setsockopt(zmq.CONFLATE, 1)
        print(
            f"[INFO] Connected to remote robot at {connection_string} and video stream at {video_connection}."
        )
        self.is_connected = True

    def load_or_run_calibration_(self, name, arm, arm_type):
        arm_id = get_arm_id(name, arm_type)
        arm_calib_path = self.calibration_dir / f"{arm_id}.json"

        if arm_calib_path.exists():
            with open(arm_calib_path) as f:
                calibration = json.load(f)
        else:
            print(f"Missing calibration file '{arm_calib_path}'")
            calibration = run_arm_manual_calibration(arm, self.robot_type, name, arm_type)
            print(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
            arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
            with open(arm_calib_path, "w") as f:
                json.dump(calibration, f)

        return calibration

    def calibrate_leader(self):
        for name, arm in self.leader_arms.items():
            # Connect the bus
            arm.connect()

            # Disable torque on all motors
            for motor_id in arm.motors:
                arm.write("Torque_Enable", TorqueMode.DISABLED.value, motor_id)

            # Now run calibration
            calibration = self.load_or_run_calibration_(name, arm, "leader")
            arm.set_calibration(calibration)

    def calibrate_follower(self):
        for name, bus in self.follower_arms.items():
            bus.connect()

            # Disable torque on all motors
            for motor_id in bus.motors:
                bus.write("Torque_Enable", 0, motor_id)

            # Then filter out wheels
            arm_only_dict = {k: v for k, v in bus.motors.items() if not k.startswith("wheel_")}
            if not arm_only_dict:
                continue

            original_motors = bus.motors
            bus.motors = arm_only_dict

            calibration = self.load_or_run_calibration_(name, bus, "follower")
            bus.set_calibration(calibration)

            bus.motors = original_motors

    def _get_data(self):
        """
        Polls the video socket for up to 15 ms. If data arrives, decode only
        the *latest* message, returning frames, speed, and arm state. If
        nothing arrives for any field, use the last known values.
        """
        frames = {}
        present_speed = {}
        remote_arm_state_tensor = torch.zeros(6, dtype=torch.float32)

        # Poll up to 15 ms
        poller = zmq.Poller()
        poller.register(self.video_socket, zmq.POLLIN)
        socks = dict(poller.poll(15))
        if self.video_socket not in socks or socks[self.video_socket] != zmq.POLLIN:
            # No new data arrived → reuse ALL old data
            return (self.last_frames, self.last_present_speed, self.last_remote_arm_state)

        # Drain all messages, keep only the last
        last_msg = None
        while True:
            try:
                obs_string = self.video_socket.recv_string(zmq.NOBLOCK)
                last_msg = obs_string
            except zmq.Again:
                break

        if not last_msg:
            # No new message → also reuse old
            return (self.last_frames, self.last_present_speed, self.last_remote_arm_state)

        # Decode only the final message
        try:
            observation = json.loads(last_msg)

            images_dict = observation.get("images", {})
            new_speed = observation.get("present_speed", {})
            new_arm_state = observation.get("follower_arm_state", None)

            # Convert images
            for cam_name, image_b64 in images_dict.items():
                if image_b64:
                    jpg_data = base64.b64decode(image_b64)
                    np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
                    frame_candidate = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame_candidate is not None:
                        frames[cam_name] = frame_candidate

            # If remote_arm_state is None and frames is None there is no message then use the previous message
            if new_arm_state is not None and frames is not None:
                self.last_frames = frames

                remote_arm_state_tensor = torch.tensor(new_arm_state, dtype=torch.float32)
                self.last_remote_arm_state = remote_arm_state_tensor

                present_speed = new_speed
                self.last_present_speed = new_speed
            else:
                frames = self.last_frames

                remote_arm_state_tensor = self.last_remote_arm_state

                present_speed = self.last_present_speed

        except Exception as e:
            print(f"[DEBUG] Error decoding video message: {e}")
            # If decode fails, fall back to old data
            return (self.last_frames, self.last_present_speed, self.last_remote_arm_state)

        return frames, present_speed, remote_arm_state_tensor

    def _process_present_speed(self, present_speed: dict) -> torch.Tensor:
        state_tensor = torch.zeros(3, dtype=torch.int32)
        if present_speed:
            decoded = {key: MobileManipulator.raw_to_degps(value) for key, value in present_speed.items()}
            if "1" in decoded:
                state_tensor[0] = decoded["1"]
            if "2" in decoded:
                state_tensor[1] = decoded["2"]
            if "3" in decoded:
                state_tensor[2] = decoded["3"]
        return state_tensor

    def teleop_step(
        self, record_data: bool = False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("MobileManipulator is not connected. Run `connect()` first.")

        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # e.g. 0.1, 0.25, or 0.4
        theta_speed = speed_setting["theta"]  # e.g. 30, 60, or 90

        # Prepare to assign the position of the leader to the follower
        arm_positions = []
        for name in self.leader_arms:
            pos = self.leader_arms[name].read("Present_Position")
            pos_tensor = torch.from_numpy(pos).float()
            # Instead of pos_tensor.item(), use tolist() to convert the entire tensor to a list
            arm_positions.extend(pos_tensor.tolist())

        # (The rest of your code for generating wheel commands remains unchanged)
        x_cmd = 0.0  # m/s forward/backward
        y_cmd = 0.0  # m/s lateral
        theta_cmd = 0.0  # deg/s rotation
        if self.pressed_keys["forward"]:
            x_cmd += xy_speed
        if self.pressed_keys["backward"]:
            x_cmd -= xy_speed
        if self.pressed_keys["left"]:
            y_cmd += xy_speed
        if self.pressed_keys["right"]:
            y_cmd -= xy_speed
        if self.pressed_keys["rotate_left"]:
            theta_cmd += theta_speed
        if self.pressed_keys["rotate_right"]:
            theta_cmd -= theta_speed

        wheel_commands = self.body_to_wheel_raw(x_cmd, y_cmd, theta_cmd)

        message = {"raw_velocity": wheel_commands, "arm_positions": arm_positions}
        self.cmd_socket.send_string(json.dumps(message))

        if not record_data:
            return

        obs_dict = self.capture_observation()

        arm_state_tensor = torch.tensor(arm_positions, dtype=torch.float32)

        wheel_velocity_tuple = self.wheel_raw_to_body(wheel_commands)
        wheel_velocity_mm = (
            wheel_velocity_tuple[0] * 1000.0,
            wheel_velocity_tuple[1] * 1000.0,
            wheel_velocity_tuple[2],
        )
        wheel_tensor = torch.tensor(wheel_velocity_mm, dtype=torch.float32)
        action_tensor = torch.cat([arm_state_tensor, wheel_tensor])
        action_dict = {"action": action_tensor}

        return obs_dict, action_dict

    def capture_observation(self) -> dict:
        """
        Capture observations from the remote robot: current follower arm positions,
        present wheel speeds (converted to body-frame velocities: x, y, theta),
        and a camera frame.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected. Run `connect()` first.")

        frames, present_speed, remote_arm_state_tensor = self._get_data()

        body_state = self.wheel_raw_to_body(present_speed)

        body_state_mm = (body_state[0] * 1000.0, body_state[1] * 1000.0, body_state[2])  # Convert x,y to mm/s
        wheel_state_tensor = torch.tensor(body_state_mm, dtype=torch.float32)
        combined_state_tensor = torch.cat((remote_arm_state_tensor, wheel_state_tensor), dim=0)

        obs_dict = {"observation.state": combined_state_tensor}

        # Loop over each configured camera
        for cam_name, cam in self.cameras.items():
            frame = frames.get(cam_name, None)
            if frame is None:
                # Create a black image using the camera's configured width, height, and channels
                frame = np.zeros((cam.height, cam.width, cam.channels), dtype=np.uint8)
            obs_dict[f"observation.images.{cam_name}"] = torch.from_numpy(frame)

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected. Run `connect()` first.")

        # Ensure the action tensor has at least 9 elements:
        #   - First 6: arm positions.
        #   - Last 3: base commands.
        if action.numel() < 9:
            # Pad with zeros if there are not enough elements.
            padded = torch.zeros(9, dtype=action.dtype)
            padded[: action.numel()] = action
            action = padded

        # Extract arm and base actions.
        arm_actions = action[:6].flatten()
        base_actions = action[6:].flatten()

        x_cmd_mm = base_actions[0].item()  # mm/s
        y_cmd_mm = base_actions[1].item()  # mm/s
        theta_cmd = base_actions[2].item()  # deg/s

        # Convert mm/s to m/s for the kinematics calculations.
        x_cmd = x_cmd_mm / 1000.0  # m/s
        y_cmd = y_cmd_mm / 1000.0  # m/s

        # Compute wheel commands from body commands.
        wheel_commands = self.body_to_wheel_raw(x_cmd, y_cmd, theta_cmd)

        arm_positions_list = arm_actions.tolist()

        message = {"raw_velocity": wheel_commands, "arm_positions": arm_positions_list}
        self.cmd_socket.send_string(json.dumps(message))

        return action

    def print_logs(self):
        pass

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected.")
        if self.cmd_socket:
            stop_cmd = {
                "raw_velocity": {"left_wheel": 0, "back_wheel": 0, "right_wheel": 0},
                "arm_positions": {},
            }
            self.cmd_socket.send_string(json.dumps(stop_cmd))
            self.cmd_socket.close()
        if self.video_socket:
            self.video_socket.close()
        if self.context:
            self.context.term()
        if PYNPUT_AVAILABLE:
            self.listener.stop()
        self.is_connected = False
        print("[INFO] Disconnected from remote robot.")

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
        if PYNPUT_AVAILABLE:
            self.listener.stop()

    @staticmethod
    def degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = abs(degps) * steps_per_deg
        speed_int = int(round(speed_in_steps))
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF
        if degps < 0:
            return speed_int | 0x8000
        else:
            return speed_int & 0x7FFF

    @staticmethod
    def raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed & 0x7FFF
        degps = magnitude / steps_per_deg
        if raw_speed & 0x8000:
            degps = -degps
        return degps

    def body_to_wheel_raw(
        self,
        x_cmd: float,
        y_cmd: float,
        theta_cmd: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert desired body-frame velocities into wheel raw commands.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the center of rotation to each wheel (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"left_wheel": value, "back_wheel": value, "right_wheel": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The raw command is computed from the wheels angular speed in deg/s
            using degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
        """
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta_cmd * (np.pi / 180.0)
        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([x_cmd, y_cmd, theta_rad])

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 120, 0]) - 90)
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel’s angular speed (deg/s) to a raw integer.
        wheel_raw = [MobileManipulator.degps_to_raw(deg) for deg in wheel_degps]

        return {"left_wheel": wheel_raw[0], "back_wheel": wheel_raw[1], "right_wheel": wheel_raw[2]}

    def wheel_raw_to_body(
        self, wheel_raw: dict, wheel_radius: float = 0.05, base_radius: float = 0.125
    ) -> tuple:
        """
        Convert wheel raw command feedback back into body-frame velocities.

        Parameters:
          wheel_raw   : Dictionary with raw wheel commands (keys: "left_wheel", "back_wheel", "right_wheel").
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the robot center to each wheel (meters).

        Returns:
          A tuple (x_cmd, y_cmd, theta_cmd) where:
             x_cmd      : Linear velocity in x (m/s).
             y_cmd      : Linear velocity in y (m/s).
             theta_cmd  : Rotational velocity in deg/s.
        """
        # Extract the raw values in order.
        raw_list = [
            int(wheel_raw.get("left_wheel", 0)),
            int(wheel_raw.get("back_wheel", 0)),
            int(wheel_raw.get("right_wheel", 0)),
        ]

        # Convert each raw command back to an angular speed in deg/s.
        wheel_degps = np.array([MobileManipulator.raw_to_degps(r) for r in raw_list])
        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel’s linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 120, 0]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Solve the inverse kinematics: body_velocity = M⁻¹ · wheel_linear_speeds.
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x_cmd, y_cmd, theta_rad = velocity_vector
        theta_cmd = theta_rad * (180.0 / np.pi)
        return (x_cmd, y_cmd, theta_cmd)


class LeKiwi:
    def __init__(self, motor_bus):
        """
        Initializes the LeKiwi with Feetech motors bus.
        """
        self.motor_bus = motor_bus
        self.motor_ids = ["left_wheel", "back_wheel", "right_wheel"]

        # Initialize motors in velocity mode.
        self.motor_bus.write("Lock", 0)
        self.motor_bus.write("Mode", [1, 1, 1], self.motor_ids)
        self.motor_bus.write("Lock", 1)
        print("Motors set to velocity mode.")

    def read_velocity(self):
        """
        Reads the raw speeds for all wheels. Returns a dictionary with motor names:
        """
        raw_speeds = self.motor_bus.read("Present_Speed", self.motor_ids)
        return {
            "left_wheel": int(raw_speeds[0]),
            "back_wheel": int(raw_speeds[1]),
            "right_wheel": int(raw_speeds[2]),
        }

    def set_velocity(self, command_speeds):
        """
        Sends raw velocity commands (16-bit encoded values) directly to the motor bus.
        The order of speeds must correspond to self.motor_ids.
        """
        self.motor_bus.write("Goal_Speed", command_speeds, self.motor_ids)

    def stop(self):
        """Stops the robot by setting all motor speeds to zero."""
        self.motor_bus.write("Goal_Speed", [0, 0, 0], self.motor_ids)
        print("Motors stopped.")
