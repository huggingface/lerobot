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
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
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
            pos_tensor = torch.from_numpy(pos)
            # Instead of pos_tensor.item(), use tolist() to convert the entire tensor to a list
            arm_positions.extend(pos_tensor.tolist())

        # (The rest of your code for generating wheel commands remains unchanged)
        x_cmd = 0.0  # m/s forward/backward
        y_cmd = 0.0  # m/s lateral
        theta_cmd = 0.0  # deg/s rotation
        if self.pressed_keys["forward"]:
            x_cmd -= 0.1
        if self.pressed_keys["backward"]:
            x_cmd += 0.1
        if self.pressed_keys["left"]:
            y_cmd -= 0.1
        if self.pressed_keys["right"]:
            y_cmd += 0.1
        if self.pressed_keys["rotate_left"]:
            theta_cmd += 90
        if self.pressed_keys["rotate_right"]:
            theta_cmd -= 90

        theta_rad = theta_cmd * (np.pi / 180)
        wheel_radius = 0.05  # meters
        base_radius = 0.125  # meters
        angles = np.radians([0, 120, 240])
        kinematic_matrix = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])
        velocity_vector = np.array([x_cmd, y_cmd, theta_rad])
        wheel_linear_speeds = kinematic_matrix.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius
        ticks_per_second = wheel_angular_speeds * (4096 / (2 * np.pi))
        max_ticks = 3000
        max_abs_ticks = max(abs(tick) for tick in ticks_per_second)
        if max_abs_ticks > max_ticks:
            scale = max_ticks / max_abs_ticks
            ticks_limited = [tick * scale for tick in ticks_per_second]
        else:
            ticks_limited = ticks_per_second
        motor_ticks = [int(round(tick)) for tick in ticks_limited]
        command_speeds = [(abs(tick) | 0x8000) if tick < 0 else (tick & 0x7FFF) for tick in motor_ticks]
        raw_velocity_command = {
            "wheel_1": command_speeds[0],
            "wheel_2": command_speeds[1],
            "wheel_3": command_speeds[2],
        }

        message = {"raw_velocity": raw_velocity_command, "arm_positions": arm_positions}
        self.cmd_socket.send_string(json.dumps(message))
        print(f"[DEBUG] Sent command: {message}")

        if not record_data:
            return

        # Retrieve remote observation.
        frame, present_speed, remote_arm_state = self._get_video_frame()
        if remote_arm_state is not None:
            remote_arm_state_tensor = torch.tensor(remote_arm_state)
        else:
            remote_arm_state_tensor = torch.tensor([])

        state_tensor = self._process_present_speed(present_speed)
        decoded_action = [MobileManipulator.raw_to_degps(raw) for raw in command_speeds]
        action_tensor = torch.tensor(decoded_action, dtype=torch.int32)

        obs_dict = {"observation.state": state_tensor, "observation.arm_state": remote_arm_state_tensor}
        action_dict = {"action": action_tensor}
        if frame is not None:
            obs_dict["observation.images.mobile"] = torch.from_numpy(frame)
        return obs_dict, action_dict

    def capture_observation(self) -> dict:
        """
        Capture observations from the remote robot: current follower arm positions,
        present wheel speed, and a camera frame.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected. Run `connect()` first.")

        obs_dict = {}
        frame, present_speed, remote_arm_state = self._get_video_frame()

        state_tensor = self._process_present_speed(present_speed)

        if remote_arm_state is not None:
            obs_dict["observation.arm_state"] = torch.tensor(remote_arm_state)
        else:
            obs_dict["observation.arm_state"] = torch.tensor([])

        obs_dict["observation.state"] = state_tensor

        if frame is not None:
            obs_dict["observation.images.mobile"] = torch.from_numpy(frame)
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        If an external policy sends a composite action containing wheel commands (and possibly arm commands),
        here we assume that the action tensor contains three values for the wheels and optionally additional
        values for follower arm positions.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected. Run `connect()` first.")

        # Process wheel commands.
        motor_ticks = [int(action[i].item()) for i in range(3)]
        raw_velocity_command = {
            "wheel_1": self.degps_to_raw(motor_ticks[0]),
            "wheel_2": self.degps_to_raw(motor_ticks[1]),
            "wheel_3": self.degps_to_raw(motor_ticks[2]),
        }
        message = {"raw_velocity": raw_velocity_command}

        # If the action tensor contains more than three elements, assume they are follower arm commands.
        if action.numel() > 3:
            arm_positions = {}
            extra = action[3:]
            # Create a flat list of motor names from all follower arm buses.
            motor_list = []
            for bus_key in sorted(self.follower_arms.keys()):
                bus = self.follower_arms[bus_key]
                motor_list.extend(bus.motors)
            # Iterate over the flat motor list.
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
