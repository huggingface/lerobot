import json
import threading
import time
from dataclasses import dataclass, field, replace

import torch
import zmq

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError

# TODO use this class to receive from leader and forward to follower in order to sync data for the dataset


@dataclass
class Ned2RobotConfig:
    """
    Example of usage:
    ```python
    Ned2RobotConfig()
    ```
    """

    # Define all components of the robot
    robot_type: str = "ned2"
    leader_arms: dict[str, list] = field(default_factory=lambda: {})
    follower_arms: dict[str, list] = field(default_factory=lambda: {})
    cameras: dict[str, Camera] = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.robot_type not in ["ned2"]:
            raise ValueError(f"Provided robot type ({self.robot_type}) is not supported.")


class Ned2Robot:
    """Niryo Ned 2 robot interface"""

    def __init__(self, config: Ned2RobotConfig | None = None, **kwargs):
        if config is None:
            config = Ned2RobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)

        self.leader_arms = self.config.leader_arms
        self.follower_arms = self.config.follower_arms

        self.latest_follower_state = None
        self.latest_leader_state = None
        self.leader_lock = threading.Lock()
        self.follower_lock = threading.Lock()

        self.robot_type = self.config.robot_type
        self.cameras = self.config.cameras
        self.is_connected = False
        self.teleop = None
        self.logs = {}

        # Needed for Ned2
        context = zmq.Context()
        self.sub_follower_socket = context.socket(zmq.SUB)
        self.sub_leader_socket = context.socket(zmq.SUB)
        self.request_socket_leader = context.socket(zmq.REQ)
        self.request_socket_follower = context.socket(zmq.REQ)

        self.pub_follower_socket = context.socket(zmq.PUB)

        # Needed for dataset v2
        action_names = [
            f"{arm}_{motor}" for arm, bus in self.config.leader_arms.items() for motor in bus.motors
        ]
        state_names = [
            f"{arm}_{motor}" for arm, bus in self.config.follower_arms.items() for motor in bus.motors
        ]
        self.names = {
            "action": action_names,
            "observation.state": state_names,
        }

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.config.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.config.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    # ---------------------------- Niryo Methods ----------------------------

    def sub_follower_state(self) -> None:
        try:
            self.sub_follower_socket.connect(f"{self.config.follower_arms.main.port}:5555")
            self.sub_follower_socket.setsockopt_string(
                zmq.SUBSCRIBE, self.config.follower_arms.main.state_topic
            )
            while True:
                message = self.sub_follower_socket.recv_multipart()
                data = json.loads(message[1])
                with self.follower_lock:
                    self.latest_follower_state = data
        except Exception as e:
            print(f"Error while getting follower states: {e}")
        finally:
            print("Closing follower states connection...")

    def sub_leader_state(self) -> None:
        try:
            self.sub_leader_socket.connect(f"{self.config.leader_arms.main.port}:5555")
            self.sub_leader_socket.setsockopt_string(zmq.SUBSCRIBE, self.config.leader_arms.main.state_topic)
            while True:
                message = self.sub_leader_socket.recv_multipart()
                data = json.loads(message[1])
                with self.leader_lock:
                    self.latest_leader_state = data
        except Exception as e:
            print(f"Error while getting leader states: {e}")
        finally:
            print("Closing leader states connection...")

    @property
    def follower_state(self) -> list:
        with self.follower_lock:
            return self.latest_follower_state

    @property
    def leader_state(self) -> list:
        with self.leader_lock:
            return self.latest_leader_state

    def trigger_freemotion(self, activate: bool) -> None:
        freemotion_msg = {"action": "freemotion", "payload": {"enable": f"{activate}"}}
        self.request_socket_leader.send_string(json.dumps(freemotion_msg))

    # ---------------------------- LeRobot Methods ----------------------------

    def connect(self) -> None:
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "Ned2Robot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.config.leader_arms and not self.config.follower_arms and not self.config.cameras:
            raise ValueError(
                "Ned2Robot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        self.request_socket_follower.connect(f"{self.config.follower_arms.main.port}:6666")
        self.request_socket_leader.connect(f"{self.config.leader_arms.main.port}:6666")
        self.pub_follower_socket.bind("tcp://0.0.0.0:5555")

        # Start subscription threads for follower and leader states
        threading.Thread(target=self.sub_follower_state, daemon=True).start()
        threading.Thread(target=self.sub_leader_state, daemon=True).start()

        # Wait until both states are populated at least once
        print("Waiting for follower and leader states")
        while self.latest_follower_state is None or self.latest_leader_state is None:
            # print(str(self.latest_follower_state))
            # print(str(self.latest_leader_state))
            time.sleep(0.01)  # Small sleep to avoid busy waiting
        print("Follower and leader states initialized")

        self.is_connected = True

        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected

        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        # self.run_calibration()

        # self.trigger_freemotion(True)

    def run_calibration(self):
        calibrate_msg = {"action": "calibrate", "payload": {}}
        self.request_socket_leader.send_string(json.dumps(calibrate_msg))
        self.request_socket_follower.send_string(json.dumps(calibrate_msg))

    def teleop_step(self, record_data=False):
        # TODO request to enable teleop

        if not self.is_connected:
            raise ConnectionError()

        # Check that both states are available
        if self.latest_follower_state is None or self.latest_leader_state is None:
            raise ValueError("Follower or leader state is not initialized.")

        state = self.follower_state
        action = self.leader_state

        if not record_data:
            return

        state = torch.as_tensor(state)
        action = torch.as_tensor(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])

        # Populate output dictionnaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self):
        state = self.follower_state
        state = torch.as_tensor(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])

        # Populate output dictionnaries
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict

    def send_action(self, action: torch.Tensor):
        if not self.is_connected:
            raise ConnectionError()

        action_string = str(action.tolist())
        self.pub_follower_socket.send_string(f"{self.config.leader_arms.main.state_topic}  {action_string}")

    def disconnect(self):
        # self.trigger_freemotion(False)

        # self.sub_follower_socket.disconnect(f"{self.config.follower_arms.main.port}")
        # self.request_socket_follower.disconnect(f"{self.config.follower_arms.main.port}")
        # self.sub_leader_socket.disconnect(f"{self.config.leader_arms.main.port}")
        # self.request_socket_leader.disconnect(f"{self.config.leader_arms.main.port}")

        self.is_connected = False
