import base64
import json

import cv2
import numpy as np
import torch
import zmq
from pynput import keyboard

from lerobot.common.robot_devices.robots.configs import MobileSO100RobotConfig
from lerobot.common.robot_devices.utils import RobotDeviceNotConnectedError


class MobileCamera:
    def __init__(self, mobile_robot):
        self.mobile_robot = mobile_robot
        # These values can be adjusted based on your remote video stream
        self.width = 640
        self.height = 480
        self.channels = 3
        self.logs = {}
        self.fps = 30

    def async_read(self):
        # Call the mobile robot’s helper method to get the latest video frame.
        frame, _ = self.mobile_robot._get_video_frame(timeout=1)
        if frame is None:
            # Return a black frame if no frame was received
            frame = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
        return frame


class MobileManipulator:
    """
    MobileManipulator is a new class for connecting to and controlling a remote robot such as Mobile-SO100.
    """

    def __init__(
        self, config: MobileSO100RobotConfig
    ):  # TODO: change this to generic MobileManipulator class
        """
        Initialize the MobileManipulator with the provided configuration.

        Expected keys in config:
          - "remote_ip": IP address of the remote robot.
          - "remote_port": Port number for the remote connection (e.g. 5555).

        Also, initializes a dictionary for tracking key states and starts
        a keyboard listener in a background thread.
        """
        self.robot_type = config.type
        self.config = config
        self.remote_ip = config.ip
        self.remote_port = config.port
        self.remote_port_video = config.video_port
        self.is_connected = False
        self.context = None
        self.cmd_socket = None
        self.video_socket = None

        # Flag to allow clean shutdown.
        self.running = True

        # Dictionary to track which keys are pressed.
        self.pressed_keys = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "rotate_left": False,
            "rotate_right": False,
        }

        # Start the keyboard listener.
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        self.listener.start()

        self._mobile_camera = MobileCamera(self)

    @property
    def cameras(self):
        """
        Provide a dictionary mimicking the camera interface expected by the recording pipeline.
        For a MobileManipulator, we assume there is a single remote video camera.
        """
        return {"mobile": self._mobile_camera}

    @property
    def camera_features(self) -> dict:
        """
        Similar to ManipulatorRobot, define a property for the camera features.
        """
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
        """
        Define motor features for the mobile manipulator.
        In this example, we assume the robot is controlled via three wheel motors.
        Adjust the names, shapes, and other details as needed.
        """
        motor_names = ["wheel_1", "wheel_2", "wheel_3"]
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(motor_names),),
                "names": motor_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(motor_names),),
                "names": motor_names,
            },
        }

    @property
    def features(self) -> dict:
        return {**self.motor_features, **self.camera_features}

    # --- Keyboard Event Handlers ---
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
                # Set the running flag to False and stop the listener.
                self.running = False
                return False
        except AttributeError:
            # In case of keys like ESC (which have no char attribute)
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
        """
        Connect to the remote robot.

        """
        self.context = zmq.Context()
        self.cmd_socket = self.context.socket(zmq.PUSH)
        connection_string = f"tcp://{self.remote_ip}:{self.remote_port}"
        self.cmd_socket.connect(connection_string)
        self.cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.is_connected = True
        print(f"[INFO] Connected to remote robot at {connection_string} using ZMQ.")

        self.video_socket = self.context.socket(zmq.SUB)
        self.video_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        video_connection = f"tcp://{self.remote_ip}:{self.remote_port_video}"
        self.video_socket.connect(video_connection)
        self.video_socket.setsockopt(zmq.CONFLATE, 1)
        print(f"[INFO] Connected to remote video stream at {video_connection}.")

    def _get_video_frame(self, timeout: int = 1):
        """
        Helper method to poll the video socket and decode the latest frame.

        Args:
            timeout (int): The poll timeout in milliseconds.

        Returns:
            tuple: (frame, present_speed) where 'frame' is a decoded image (or None)
                   and 'present_speed' is any accompanying speed data (or None).
        """
        if self.video_socket is not None:
            poller = zmq.Poller()
            poller.register(self.video_socket, zmq.POLLIN)
            socks = dict(poller.poll(timeout=timeout))
            if self.video_socket in socks and socks[self.video_socket] == zmq.POLLIN:
                try:
                    obs_string = self.video_socket.recv_string(zmq.NOBLOCK)
                    observation = json.loads(obs_string)
                    image_b64 = observation.get("image", "")
                    present_speed = observation.get("present_speed", None)
                    # Decode the base64 image.
                    jpg_original = base64.b64decode(image_b64)
                    np_arr = np.frombuffer(jpg_original, dtype=np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    return frame, present_speed
                except Exception as e:
                    print(f"[DEBUG] Error receiving video frame: {e}")
        return None, None

    def teleop_step(
        self, record_data: bool = False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Perform one teleoperation step using keyboard input.
        Instead of sending the intermediate x/y/θ command, this version converts the command
        into raw wheel velocities (encoded as 16-bit integers) and sends those to the remote robot.
        When record_data is True, the raw wheel commands are stored in the action dict.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "MobileManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # --- Teleop Command Processing ---
        # Teleoperation commands are given in physical units:
        #   x_cmd: forward/backward velocity in m/s (positive = backward, negative = forward)
        #   y_cmd: left/right velocity in m/s (positive = right, negative = left)
        #   theta_cmd: rotation rate in degrees/s (positive = counterclockwise)
        x_cmd = 0.0  # m/s
        y_cmd = 0.0  # m/s
        theta_cmd = 0.0  # deg/s

        # Swap x_cmd and y_cmd
        if self.pressed_keys["forward"]:  # w
            y_cmd -= 0.2  # move forward
        if self.pressed_keys["backward"]:  # s
            y_cmd += 0.2  # move backward
        if self.pressed_keys["left"]:  # a
            x_cmd -= 0.2  # move left
        if self.pressed_keys["right"]:  # d
            x_cmd += 0.2  # move right
        if self.pressed_keys["rotate_left"]:
            theta_cmd += 90  # Rotate counterclockwise (deg/s)
        if self.pressed_keys["rotate_right"]:
            theta_cmd -= 90  # Rotate clockwise (deg/s)

        # --- Convert rotational command from deg/s to rad/s ---
        theta_rad = theta_cmd * (np.pi / 180)  # rad/s

        # --- Omniwheel Kinematics Setup ---
        # Wheel and robot parameters:
        wheel_radius = 0.05  # in meters
        base_radius = 0.125  # distance from robot center to wheel (in meters)

        # The wheels are mounted at 0°, 120°, and 240° relative to the robot frame.
        angles = np.radians([0, 120, 240])

        # IMPORTANT: Use the standard inverse kinematics matrix:
        #   For each wheel i:
        #      wheel_speed_i (m/s) = cos(alpha_i)*v_x + sin(alpha_i)*v_y + base_radius*omega
        #
        # This replaces the previous definition which used [-sin(a), cos(a), base_radius].
        kinematic_matrix = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # The robot velocity vector:
        #   v_x is forward (m/s), v_y is lateral (m/s), and omega is rotation (rad/s)
        velocity_vector = np.array([x_cmd, y_cmd, theta_rad])

        # Compute the required wheel linear speeds (in m/s)
        wheel_linear_speeds = kinematic_matrix.dot(velocity_vector)

        # --- Convert Wheel Speeds to Motor Ticks/s ---
        # Step 1: Convert wheel linear speeds (m/s) to wheel angular speeds (rad/s)
        #         using: wheel_angular_speed = wheel_linear_speed / wheel_radius
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Step 3: Convert motor angular speed (rad/s) to ticks/s.
        #         One full rotation (2π rad) equals 4096 ticks.
        ticks_per_second = wheel_angular_speeds * (4096 / (2 * np.pi))

        # --- Limit the Motor Speed ---
        # Define the maximum allowed motor speed.
        max_ticks = 4096  # equals 4096 ticks/s

        # Scale the ticks/s values if any exceed the maximum.
        max_abs_ticks = max(abs(tick) for tick in ticks_per_second)
        if max_abs_ticks > max_ticks:
            scale = max_ticks / max_abs_ticks
            ticks_limited = [tick * scale for tick in ticks_per_second]
        else:
            ticks_limited = ticks_per_second

        # Round to integer tick values.
        motor_ticks = [int(round(tick)) for tick in ticks_limited]

        # --- Encode Each Tick Speed into a 16-bit Command ---
        # The encoding format is:
        #   - Lower 15 bits: absolute tick value.
        #   - MSB (bit 15): set (i.e. OR with 0x8000) if the tick value is negative.
        command_speeds = [(abs(tick) | 0x8000) if tick < 0 else (tick & 0x7FFF) for tick in motor_ticks]

        # --- Package and Send the Command ---
        raw_velocity_command = {
            "wheel_1": command_speeds[0],
            "wheel_2": command_speeds[1],
            "wheel_3": command_speeds[2],
        }
        message = {"raw_velocity": raw_velocity_command}
        self.cmd_socket.send_string(json.dumps(message))

        # Decode each wheel’s raw command back into deg/s for debugging.
        decoded_speed = {
            wheel: MobileManipulator.raw_to_degps(raw_value)
            for wheel, raw_value in raw_velocity_command.items()
        }
        print(f"[DEBUG] Sent raw velocity command (decoded): {decoded_speed} deg/s")

        # Use the helper method to receive a video frame.
        frame, present_speed = self._get_video_frame(timeout=1)

        # Early exit when recording data is not requested
        if not record_data:
            return

        # --- Recording Data ---
        obs_dict, action_dict = {}, {}

        if present_speed is None:
            state_tensor = torch.zeros(3, dtype=torch.int32)
        else:
            # Decode each raw speed in the present_speed observation.
            decoded_present_speed = [MobileManipulator.raw_to_degps(raw) for raw in present_speed]
            state_tensor = torch.tensor(decoded_present_speed, dtype=torch.int32)

        # Similarly, store the sent action in degrees per second.
        decoded_action = [MobileManipulator.raw_to_degps(raw) for raw in command_speeds]
        action_tensor = torch.tensor(decoded_action, dtype=torch.int32)

        obs_dict["observation.state"] = state_tensor
        action_dict["action"] = action_tensor

        if frame is not None:
            frame_tensor = torch.from_numpy(frame)
            obs_dict["observation.images.mobile"] = frame_tensor

        return obs_dict, action_dict

    def capture_observation(self) -> dict:
        """
        Retrieve sensor data (observations) from the remote robot.
        In addition to the state and camera images (if any), the camera stream observation is added:
        - "observation.image": a tensor containing the camera image from the video stream.
        - "observation.speed": the velocity information (in degrees/s) received along with the image.
        """
        obs_dict = {}

        # Use the helper method to get the latest video frame.
        frame, present_speed = self._get_video_frame(timeout=5)
        if frame is not None:
            # Convert the image from BGR to RGB and then to a torch tensor.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_img = torch.from_numpy(frame_rgb)
            obs_dict["observation.image"] = tensor_img

        if present_speed is not None:
            # Assume present_speed is an iterable of raw 16-bit values.
            decoded_speeds = [MobileManipulator.raw_to_degps(raw) for raw in present_speed]
            obs_dict["observation.speed"] = torch.tensor(decoded_speeds, dtype=torch.int32)

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Send a raw velocity command to the remote robot.
        Here we assume the action tensor contains three elements corresponding to
        the raw motor commands (already encoded as 16-bit integers) for:
          - wheel_1
          - wheel_2
          - wheel_3
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "MobileManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Construct the raw velocity command dictionary.
        # We assume that the action tensor has exactly three elements.
        raw_velocity_command = {
            "wheel_1": self.degps_to_raw(int(action[0].item())),
            "wheel_2": self.degps_to_raw(int(action[1].item())),
            "wheel_3": self.degps_to_raw(int(action[2].item())),
        }

        # Wrap it in a dictionary with key "raw_velocity" so the remote robot recognizes it.
        message = {"raw_velocity": raw_velocity_command}

        # Send the JSON-encoded command over the ZMQ socket.
        self.cmd_socket.send_string(json.dumps(message))
        print(f"[DEBUG] Sent raw velocity command: {raw_velocity_command} deg/s")

        action_sent = [action]
        return torch.cat(action_sent)

    def print_logs(self):
        pass

    def disconnect(self):
        """
        Disconnect from the remote robot.

        Steps:
          1. Optionally send a disconnect or shutdown command to the remote robot.
          2. Close the network socket.
          3. Update self.is_connected to False.
          4. Log the disconnection event.
        """
        if self.is_connected and self.cmd_socket:
            stop_cmd = {"x": 0, "y": 0, "theta": 0}
            self.cmd_socket.send_string(json.dumps(stop_cmd))
            self.cmd_socket.close()
            if self.video_socket:
                self.video_socket.close()
            self.context.term()
            self.is_connected = False
            print("[INFO] Disconnected from remote robot.")

        self.listener.stop()

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
        if self.listener:
            self.listener.stop()

    @staticmethod
    def degps_to_raw(degps: float) -> int:
        """
        Convert speed in degrees/s to a 16-bit signed raw command,
        where the lower 15 bits store the speed in steps/s (4096 steps per 360°),
        and bit 15 is the sign bit (1 = negative).
        """
        steps_per_deg = 4096.0 / 360.0  # ~11.3778 steps per degree
        # Convert deg/s to steps/s
        speed_in_steps = abs(degps) * steps_per_deg

        # Round and clamp to the 15-bit maximum (0x7FFF = 32767)
        speed_int = int(round(speed_in_steps))
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF

        # Set the sign bit if negative
        if degps < 0:
            return speed_int | 0x8000  # 0x8000 sets bit 15
        else:
            return speed_int & 0x7FFF  # Ensure bit 15 is cleared

    @staticmethod
    def raw_to_degps(raw_speed: int) -> float:
        """
        Convert a 16-bit signed raw speed (steps/s in lower 15 bits, sign in bit 15)
        back to degrees/s (°/s). 4096 steps = 360°.
        """
        steps_per_deg = 4096.0 / 360.0
        # Extract the magnitude (lower 15 bits)
        magnitude = raw_speed & 0x7FFF
        # Convert steps/s -> deg/s
        degps = magnitude / steps_per_deg

        # Check bit 15 for sign
        if raw_speed & 0x8000:  # negative speed
            degps = -degps

        return degps
