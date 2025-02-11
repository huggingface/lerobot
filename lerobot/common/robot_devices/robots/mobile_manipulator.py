import base64
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import zmq
from pynput import keyboard

from lerobot.common.robot_devices.motors.feetech import TorqueMode
from lerobot.common.robot_devices.motors.utils import MotorsBus, make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.configs import MobileSO100RobotConfig
from lerobot.common.robot_devices.robots.feetech_calibration import run_arm_manual_calibration
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceNotConnectedError


class MobileCamera:
    def __init__(self, mobile_robot):
        self.mobile_robot = mobile_robot
        self.width = 640
        self.height = 480
        self.channels = 3
        self.logs = {}
        self.fps = 30

    def async_read(self):
        # The remote camera frame is received via the video socket.
        frame, _, _ = self.mobile_robot._get_video_frame(timeout=1)
        if frame is None:
            # Return a black frame if no frame was received.
            frame = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
        return frame


class MobileManipulator:
    """
    MobileManipulator is a class for connecting to and controlling a remote mobile manipulator robot.
    The robot includes a three omniwheel mobile base and a remote follower arm.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    def __init__(self, config: MobileSO100RobotConfig):
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

        # For teleoperation, the leader arm (local) is used to record the desired arm pose.
        self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)
        # Although follower arms are configured, on the laptop we do not read them locally.
        self.follower_arms = make_motors_buses_from_configs(self.config.follower_arms)

        # The mobile camera interface.
        self._mobile_camera = MobileCamera(self)
        self.is_connected = False

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
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        self.listener.start()

    def get_motor_names(self, arms: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arms.items() for motor in bus.motors]

    @property
    def cameras(self):
        # Provide a dictionary mimicking the camera interface expected by the recording pipeline.
        return {"mobile": self._mobile_camera}

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
        # For mobile teleop we assume the “action” has two parts:
        #   (i) Arm positions (to be set remotely)
        #  (ii) Wheel velocity commands (for wheel_1, wheel_2, wheel_3)
        follower_arm_names = self.get_motor_names(self.follower_arms)
        wheel_names = ["wheel_1", "wheel_2", "wheel_3"]
        combined_names = follower_arm_names + wheel_names
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
    def features(self) -> dict:
        return {**self.motor_features, **self.camera_features}

    @property
    def available_arms(self):
        available = []
        for name in self.follower_arms:
            available.append(get_arm_id(name, "follower"))
        for name in self.leader_arms:
            available.append(get_arm_id(name, "leader"))
        return available

    def on_press(self, key):
        try:
            if key.char == "w":
                self.pressed_keys["forward"] = True
            elif key.char == "s":
                self.pressed_keys["backward"] = True
            elif key.char == "a":
                self.pressed_keys["left"] = True
            elif key.char == "d":
                self.pressed_keys["right"] = True
            elif key.char == "z":
                self.pressed_keys["rotate_left"] = True
            elif key.char == "x":
                self.pressed_keys["rotate_right"] = True
            elif key.char == "q":
                self.running = False
                return False
        except AttributeError:
            if key == keyboard.Key.esc:
                self.running = False
                return False

    def on_release(self, key):
        try:
            if hasattr(key, "char"):
                if key.char == "w":
                    self.pressed_keys["forward"] = False
                elif key.char == "s":
                    self.pressed_keys["backward"] = False
                elif key.char == "a":
                    self.pressed_keys["left"] = False
                elif key.char == "d":
                    self.pressed_keys["right"] = False
                elif key.char == "z":
                    self.pressed_keys["rotate_left"] = False
                elif key.char == "x":
                    self.pressed_keys["rotate_right"] = False
        except AttributeError:
            pass

    def connect(self):
        if not self.leader_arms:
            raise ValueError("MobileManipulator has no leader arm to connect.")
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()
            self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        # Set up ZeroMQ sockets to communicate with the remote mobile robot.
        self.context = zmq.Context()
        self.cmd_socket = self.context.socket(zmq.PUSH)
        connection_string = f"tcp://{self.remote_ip}:{self.remote_port}"
        self.cmd_socket.connect(connection_string)
        self.cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.video_socket = self.context.socket(zmq.SUB)
        self.video_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        video_connection = f"tcp://{self.remote_ip}:{self.remote_port_video}"
        self.video_socket.connect(video_connection)
        self.video_socket.setsockopt(zmq.RCVHWM, 1)
        print(
            f"[INFO] Connected to remote robot at {connection_string} and video stream at {video_connection}."
        )
        self.is_connected = True

        self.activate_calibration()

    def activate_calibration(self):
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """

        def load_or_run_calibration_(name, arm, arm_type):
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

        for name, arm in self.leader_arms.items():
            calibration = load_or_run_calibration_(name, arm, "leader")
            arm.set_calibration(calibration)

    def _get_video_frame(self, timeout=1):
        """
        Receives messages from the video socket and extracts:
          - The JPEG image frame,
          - The present wheel speed (as provided by the remote),
          - The remote follower arm state (arm positions) if available.
        """
        frame, present_speed, remote_arm_state = None, {}, None
        while True:
            try:
                obs_string = self.video_socket.recv_string(flags=zmq.NOBLOCK)
                observation = json.loads(obs_string)
                image_b64 = observation.get("image", "")
                speed_data = observation.get("present_speed", {})
                remote_arm_state = observation.get("follower_arm_state", None)
                if image_b64:
                    jpg_original = base64.b64decode(image_b64)
                    np_arr = np.frombuffer(jpg_original, dtype=np.uint8)
                    frame_candidate = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame_candidate is not None:
                        frame = frame_candidate
                        # frame = cv2.rotate(frame, cv2.ROTATE_180)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                present_speed = speed_data
            except zmq.Again:
                break
            except Exception as e:
                print(f"[DEBUG] Error decoding video message: {e}")
                break
        return frame, present_speed, remote_arm_state

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
            x_cmd -= 0.4
        if self.pressed_keys["backward"]:
            x_cmd += 0.4
        if self.pressed_keys["left"]:
            y_cmd -= 0.4
        if self.pressed_keys["right"]:
            y_cmd += 0.4
        if self.pressed_keys["rotate_left"]:
            theta_cmd += 90
        if self.pressed_keys["rotate_right"]:
            theta_cmd -= 90

        wheel_commands = self.body_to_wheel_raw(x_cmd, y_cmd, theta_cmd)

        message = {"raw_velocity": wheel_commands, "arm_positions": arm_positions}

        # TODO(pepijn): Remove this
        self.cmd_socket.send_string(json.dumps(message))
        print(f"[DEBUG] Sent command: {message}")

        if not record_data:
            return

        obs_dict = self.capture_observation()

        arm_state_tensor = torch.tensor(arm_positions, dtype=torch.float32)
        wheel_tensor = torch.tensor([x_cmd, y_cmd, theta_cmd], dtype=torch.float32)
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

        frame, present_speed, remote_arm_state = self._get_video_frame()
        if remote_arm_state is not None:
            remote_arm_state_tensor = torch.tensor(remote_arm_state, dtype=torch.float32)
        else:
            # Instead of an empty tensor, create a default tensor of the expected shape.
            remote_arm_state_tensor = torch.zeros(6, dtype=torch.float32)

        present_speed_dict = {
            "wheel_1": int(present_speed.get("1", 0)),
            "wheel_2": int(present_speed.get("2", 0)),
            "wheel_3": int(present_speed.get("3", 0)),
        }

        body_state = self.wheel_raw_to_body(present_speed_dict)
        wheel_state_tensor = torch.tensor(body_state, dtype=torch.float32)
        combined_state_tensor = torch.cat((wheel_state_tensor, remote_arm_state_tensor), dim=0)

        obs_dict = {"observation.state": combined_state_tensor}

        if frame is not None:
            obs_dict["observation.images.mobile"] = torch.from_numpy(frame)

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        If an external policy sends a composite action containing body commands (x, y, theta)
        for the mobile base and optionally additional commands for follower arms,
        this method converts the body commands into raw wheel commands and sends the complete message.

        The action tensor is expected to have:
        - The first three values as [x_cmd, y_cmd, theta_cmd],
            where x_cmd and y_cmd are in m/s and theta_cmd is in deg/s.å
        - Any additional values are treated as follower arm commands.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected. Run `connect()` first.")

        x_cmd = action[0].item()  # m/s forward/backward
        y_cmd = action[1].item()  # m/s lateral
        theta_cmd = action[2].item()  # deg/s rotation

        wheel_commands = self.body_to_wheel_raw(x_cmd, y_cmd, theta_cmd)
        message = {"raw_velocity": wheel_commands}

        # Process arm commands if additional elements exist.
        if action.numel() > 3:
            arm_positions = {}
            extra = action[3:]
            motor_list = []
            for bus_key in sorted(self.follower_arms.keys()):
                bus = self.follower_arms[bus_key]
                motor_list.extend(bus.motors)
            for i, motor in enumerate(motor_list):
                if i < extra.numel():
                    arm_positions[motor] = int(extra[i].item())
            message["arm_positions"] = arm_positions

        self.cmd_socket.send_string(json.dumps(message))
        return action

    def print_logs(self):
        pass

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected.")
        if self.cmd_socket:
            # Send a “stop” command before disconnecting.
            stop_cmd = {"raw_velocity": {"wheel_1": 0, "wheel_2": 0, "wheel_3": 0}, "arm_positions": {}}
            self.cmd_socket.send_string(json.dumps(stop_cmd))
            self.cmd_socket.close()
        if self.video_socket:
            self.video_socket.close()
        if self.context:
            self.context.term()
        self.listener.stop()
        self.is_connected = False
        print("[INFO] Disconnected from remote robot.")

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
        if self.listener:
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
             {"wheel_1": value, "wheel_2": value, "wheel_3": value}.

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

        # Define the wheel mounting angles with a -45° offset.
        angles = np.radians(np.array([0, 120, 240]) - 30)
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # --- Scaling (Saturation) ---
        # Compute the tentative raw command for each wheel (before integer encoding).
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel’s (possibly scaled) angular speed (deg/s) to a raw integer.
        wheel_raw = [MobileManipulator.degps_to_raw(deg) for deg in wheel_degps]

        return {"wheel_1": wheel_raw[0], "wheel_2": wheel_raw[1], "wheel_3": wheel_raw[2]}

    def wheel_raw_to_body(
        self, wheel_raw: dict, wheel_radius: float = 0.05, base_radius: float = 0.125
    ) -> tuple:
        """
        Convert wheel raw command feedback back into body-frame velocities.

        Parameters:
          wheel_raw   : Dictionary with raw wheel commands (keys: "wheel_1", "wheel_2", "wheel_3").
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the robot center to each wheel (meters).

        Returns:
          A tuple (x_cmd, y_cmd, theta_cmd) where:
             x_cmd      : Linear velocity in x (m/s).
             y_cmd      : Linear velocity in y (m/s).
             theta_cmd  : Rotational velocity in deg/s.
        """
        # Extract the raw values in order.
        raw_list = [wheel_raw["wheel_1"], wheel_raw["wheel_2"], wheel_raw["wheel_3"]]

        # Convert each raw command back to an angular speed in deg/s.
        wheel_degps = np.array([MobileManipulator.raw_to_degps(r) for r in raw_list])
        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel’s linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Define the wheel mounting angles with a -45° offset.
        angles = np.radians(np.array([0, 120, 240]) - 30)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Solve the inverse kinematics: body_velocity = M⁻¹ · wheel_linear_speeds.
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x_cmd, y_cmd, theta_rad = velocity_vector
        theta_cmd = theta_rad * (180.0 / np.pi)
        return (x_cmd, y_cmd, theta_cmd)
