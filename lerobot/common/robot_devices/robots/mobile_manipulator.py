import base64
import json

import cv2
import numpy as np
import torch
import zmq
from pynput import keyboard

from lerobot.common.robot_devices.robots.configs import MobileSO100RobotConfig

# You may want to import or use a keyboard input library (e.g. 'keyboard' or 'pynput') in the future.
# import keyboard  # Uncomment if using a keyboard package for capturing key events.


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
        Additionally, receive and display the latest video frame.
        """
        # Process the current key states.
        x = 0.0
        y = 0.0
        theta = 0.0

        if self.pressed_keys["forward"]:
            x += 0.2
        if self.pressed_keys["backward"]:
            x -= 0.2
        if self.pressed_keys["left"]:
            y += 0.2
        if self.pressed_keys["right"]:
            y -= 0.2
        if self.pressed_keys["rotate_left"]:
            theta += 1.4
        if self.pressed_keys["rotate_right"]:
            theta -= 1.4

        # Build the command dictionary.
        command = {"x": x, "y": y, "theta": theta}

        # Send the command if connected.
        if self.is_connected and self.cmd_socket:
            self.cmd_socket.send_string(json.dumps(command))
            print(f"[DEBUG] Sent command: {command}")
        else:
            print("[WARNING] Not connected. Unable to send command.")

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

        # If record_data is enabled, simulate receiving sensor data.
        if record_data:
            # In a real application, you might wait for a sensor response here.
            # For this example we create dummy observation and command tensors.
            obs = {"observation.state": torch.tensor([0.0])}
            actions = {"action.command": torch.tensor([x, y, theta])}
            return obs, actions

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
        Send an action command to the remote robot.

        Steps:
          1. Convert the action (torch.Tensor) into a suitable format (e.g., JSON, bytes).
          2. Send the command over the established network connection.
          3. Optionally, wait for an acknowledgment or status feedback from the remote robot.
          4. Log the details of the sent action and any received response.
          5. Return the action that was sent (or the modified action if any limits are applied).
        """
        # Pseudo-code outline:
        # action_data = serialize_action(action)
        # self.socket.sendall(action_data)
        # Optionally, receive a confirmation message.
        # ack = self.socket.recv(1024)
        # return action  # or modified action based on remote feedback.
        pass

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
