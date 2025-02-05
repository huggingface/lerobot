import json

import torch
import zmq
from pynput import keyboard

from lerobot.common.robot_devices.robots.configs import MobileSO100RobotConfig

# You may want to import or use a keyboard input library (e.g. 'keyboard' or 'pynput') in the future.
# import keyboard  # Uncomment if using a keyboard package for capturing key events.


class MobileManipulator:
    """
    MobileManipulator is a new class for connecting to and controlling a remote robot.

    In contrast to the ManipulatorRobot that directly interfaces with hardware,
    MobileManipulator communicates over a network (e.g. via TCP sockets) and accepts keyboard input
    for teleoperation.

    All I/O operations (network communications and keyboard input) are described in the method comments.
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
        # Use port 5555 by default for ZMQ (adjust as needed)
        self.remote_port = config.port
        self.is_connected = False
        self.context = None
        self.cmd_socket = None

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

        Steps:
          1. Create a network socket (e.g., using the socket module).
          2. Connect to the remote IP and port specified in the configuration.
          3. Optionally perform a handshake or initialization sequence with the remote robot.
          4. Set self.is_connected to True and log the connection details.
        """
        self.context = zmq.Context()
        self.cmd_socket = self.context.socket(zmq.PUSH)
        connection_string = f"tcp://{self.remote_ip}:{self.remote_port}"
        self.cmd_socket.connect(connection_string)
        self.is_connected = True
        print(f"[INFO] Connected to remote robot at {connection_string} using ZMQ.")

    def teleop_step(
        self, record_data: bool = False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Perform one teleoperation step using keyboard input.

        Steps:
          1. Capture keyboard input:
             - Use a keyboard input library or a non-blocking I/O method.
             - Map specific keys (e.g., arrow keys, WASD) to robot commands.
          2. Process the input to create a control command for the robot.
          3. Send the command over the network connection to the remote robot.
          4. Optionally, if record_data is True:
             - Request and retrieve sensor data or feedback from the remote robot.
             - Convert received data to torch.Tensors.
             - Return a tuple (observations, actions) as dictionaries.
          5. Log the durations for input reading, command processing, and network I/O.
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
        xc
           Steps:
             1. Send a request (if needed) to the remote robot to get the current state/sensor data.
             2. Wait for and read the response data from the network.
             3. Parse and convert the data (e.g., positions, images) into a standardized format,
                such as torch.Tensors.
             4. Return the observations as a dictionary.
        """
        # Pseudo-code outline:
        # self.socket.sendall(b'GET_STATE')
        # response = self.socket.recv(8192)
        # observations = parse_observation_response(response)
        # Convert data to torch.Tensor as needed:
        # observations["observation.state"] = torch.tensor(observations["state_data"])
        # Return the dictionary.
        pass

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
            self.context.term()
            self.is_connected = False
            print("[INFO] Disconnected from remote robot.")

        self.listener.stop()

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
        if self.listener:
            self.listener.stop()
