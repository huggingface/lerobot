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
            if key == keyboard.Key.up:
                self.pressed_keys["forward"] = True
            elif key == keyboard.Key.down:
                self.pressed_keys["backward"] = True
            elif key == keyboard.Key.left:
                self.pressed_keys["left"] = True
            elif key == keyboard.Key.right:
                self.pressed_keys["right"] = True
            elif key.char == "x":
                self.pressed_keys["rotate_left"] = True
            elif key.char == "c":
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
            if key == keyboard.Key.up:
                self.pressed_keys["forward"] = False
            elif key == keyboard.Key.down:
                self.pressed_keys["backward"] = False
            elif key == keyboard.Key.left:
                self.pressed_keys["left"] = False
            elif key == keyboard.Key.right:
                self.pressed_keys["right"] = False
            elif hasattr(key, "char"):
                if key.char == "x":
                    self.pressed_keys["rotate_left"] = False
                elif key.char == "c":
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

        # Process the current key states.
        x = 0.0
        y = 0.0
        theta = 0.0

        if self.pressed_keys["forward"]:
            x -= 0.2
        if self.pressed_keys["backward"]:
            x += 0.2
        if self.pressed_keys["left"]:
            y -= 0.2
        if self.pressed_keys["right"]:
            y += 0.2
        if self.pressed_keys["rotate_left"]:
            theta += 1.4
        if self.pressed_keys["rotate_right"]:
            theta -= 1.4

        # --- Convert Input Command to Raw Wheel Velocities ---
        # Define the kinematic parameters (these match those in your MobileSO100 class).
        wheel_radius = 0.05  # 5 cm wheel radius
        base_radius = 0.125  # 12.5 cm from center to wheel
        angles = np.radians([0, 120, 240])  # mounting angles in radians

        # Build the kinematic matrix:
        # For omni-wheels the contribution is: [-sin(theta), cos(theta), base_radius]
        kinematic_matrix = np.array([[-np.sin(a), np.cos(a), base_radius] for a in angles])

        # Form the desired velocity vector.
        velocity_vector = np.array([x, y, theta])
        # Compute individual wheel speeds (in m/s).
        wheel_speeds = kinematic_matrix.dot(velocity_vector)

        # Convert wheel speeds (m/s) to motor speeds (steps per second).
        # Conversion factor: first, linear speed -> RPM, then RPM -> steps/s.
        conversion_factor = (60 / (2 * np.pi)) * (50 / 0.732)  # approximately 652.0
        motor_speeds = (wheel_speeds / wheel_radius) * conversion_factor

        # Convert to a list of integers.
        motor_speeds = [int(round(speed)) for speed in motor_speeds]

        # If any absolute speed exceeds 3400, scale all speeds.
        max_speed_in_steps = 3400.0
        max_abs_speed = max(abs(speed) for speed in motor_speeds)
        if max_abs_speed > max_speed_in_steps:
            scalar = max_speed_in_steps / max_abs_speed
            motor_speeds = [int(round(speed * scalar)) for speed in motor_speeds]

        # Encode each speed into a 16-bit command:
        #   - The lower 15 bits hold the speed value.
        #   - Bit 15 is set (i.e. OR 0x8000) when the speed is negative.
        command_speeds = [(abs(speed) | 0x8000) if speed < 0 else (speed & 0x7FFF) for speed in motor_speeds]

        # --- Send the Raw Command via ZMQ ---
        # Build the message as a JSON dictionary. We assume the remote robot expects the raw speeds
        # under a key "raw_velocity" with subkeys for each wheel.
        raw_velocity_command = {
            "wheel_1": command_speeds[0],
            "wheel_2": command_speeds[1],
            "wheel_3": command_speeds[2],
        }
        message = {"raw_velocity": raw_velocity_command}
        self.cmd_socket.send_string(json.dumps(message))
        print(f"[DEBUG] Sent raw velocity command: {raw_velocity_command}")

        # Use the helper method to receive a video frame.
        frame, present_speed = self._get_video_frame(timeout=1)
        if frame is not None:
            # Overlay the present_speed if available.
            if present_speed is not None:
                text = f"Velocity: {present_speed}"
                cv2.putText(
                    frame, text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
            cv2.imshow("Robot Camera", frame)
            cv2.waitKey(1)
        else:
            print("[DEBUG] No video frame received.")

        # Early exit when recording data is not requested
        if not record_data:
            return

        # --- Recording Data ---
        obs_dict, action_dict = {}, {}

        # Create a tensor for state.
        # If present_speed is None, create a zero tensor.
        if present_speed is None:
            state_tensor = torch.zeros(3, dtype=torch.int32)
        else:
            # Ensure present_speed is an iterable of numbers.
            state_tensor = torch.tensor(present_speed, dtype=torch.int32)

        # Create a tensor for the action (raw command speeds).
        action_tensor = torch.tensor(command_speeds, dtype=torch.int32)

        obs_dict["observation.state"] = state_tensor
        action_dict["action"] = action_tensor

        # For cameras: assuming a single camera "mobile", convert the frame to a tensor.
        if frame is not None:
            # Convert the BGR image (NumPy array) to a tensor.
            frame_tensor = torch.from_numpy(frame)
            # Assuming the camera key is "mobile".
            obs_dict["observation.images.mobile"] = frame_tensor

        return obs_dict, action_dict

    def capture_observation(self) -> dict:
        """
        Retrieve sensor data (observations) from the remote robot.
        In addition to the state and camera images (if any), the camera stream observation is added:
          - "observation.image": a tensor containing the camera image from the video stream.
          - "observation.present_speed": the velocity information received along with the image.
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
            obs_dict["observation.present_speed"] = torch.tensor(present_speed)

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
            "wheel_1": int(action[0].item()),
            "wheel_2": int(action[1].item()),
            "wheel_3": int(action[2].item()),
        }

        # Wrap it in a dictionary with key "raw_velocity" so the remote robot recognizes it.
        message = {"raw_velocity": raw_velocity_command}

        # Send the JSON-encoded command over the ZMQ socket.
        self.cmd_socket.send_string(json.dumps(message))
        print(f"[DEBUG] Sent raw velocity command: {raw_velocity_command}")

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
