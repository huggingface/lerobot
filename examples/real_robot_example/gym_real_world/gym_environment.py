import time
from unittest.mock import MagicMock

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dynamixel import pos2pwm, pwm2pos
from .robot import Robot

FPS = 30

CAMERAS_SHAPES = {
    "images.high": (480, 640, 3),
    "images.low": (480, 640, 3),
}

CAMERAS_PORTS = {
    "images.high": "/dev/video6",
    "images.low": "/dev/video0",
}

LEADER_PORT = "/dev/ttyACM1"
FOLLOWER_PORT = "/dev/ttyACM0"

MockRobot = MagicMock()
MockRobot.read_position = MagicMock()
MockRobot.read_position.return_value = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

MockCamera = MagicMock()
MockCamera.isOpened = MagicMock(return_value=True)
MockCamera.read = MagicMock(return_value=(True, np.zeros((480, 640, 3), dtype=np.uint8)))


def capture_image(cam, cam_width, cam_height):
    # Capture a single frame
    _, frame = cam.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # # Define your crop coordinates (top left corner and bottom right corner)
    # x1, y1 = 400, 0  # Example starting coordinates (top left of the crop rectangle)
    # x2, y2 = 1600, 900  # Example ending coordinates (bottom right of the crop rectangle)
    # # Crop the image
    # image = image[y1:y2, x1:x2]
    # Resize the image
    image = cv2.resize(image, (cam_width, cam_height), interpolation=cv2.INTER_AREA)

    return image


class RealEnv(gym.Env):
    metadata = {}

    def __init__(
        self,
        record: bool = False,
        num_joints: int = 6,
        cameras_shapes: dict = CAMERAS_SHAPES,
        cameras_ports: dict = CAMERAS_PORTS,
        follower_port: str = FOLLOWER_PORT,
        leader_port: str = LEADER_PORT,
        warmup_steps: int = 100,
        trigger_torque=70,
        fps: int = FPS,
        fps_tolerance: float = 0.1,
        mock: bool = False,
    ):
        self.num_joints = num_joints
        self.cameras_shapes = cameras_shapes
        self.cameras_ports = cameras_ports
        self.warmup_steps = warmup_steps
        assert len(self.cameras_shapes) == len(self.cameras_ports), "Number of cameras and shapes must match."

        self.follower_port = follower_port
        self.leader_port = leader_port
        self.record = record
        self.fps = fps
        self.fps_tolerance = fps_tolerance

        # Initialize the robot
        self.follower = Robot(device_name=self.follower_port) if not mock else MockRobot
        if self.record:
            self.leader = Robot(device_name=self.leader_port) if not mock else MockRobot
            self.leader.set_trigger_torque(trigger_torque)

        # Initialize the cameras - sorted by camera names
        self.cameras = {}
        for cn, p in sorted(self.cameras_ports.items()):
            self.cameras[cn] = cv2.VideoCapture(p) if not mock else MockCamera
            if not self.cameras[cn].isOpened():
                raise OSError(
                    f"Cannot open camera port {p} for {cn}."
                    f" Make sure the camera is connected and the port is correct."
                    f"Also check you are not spinning several instances of the same environment (eval.batch_size)"
                )

        # Specify gym action and observation spaces
        observation_space = {}

        if self.num_joints > 0:
            observation_space["agent_pos"] = spaces.Box(
                low=-1000.0,
                high=1000.0,
                shape=(num_joints,),
                dtype=np.float64,
            )
        if self.record:
            observation_space["leader_pos"] = spaces.Box(
                low=-1000.0,
                high=1000.0,
                shape=(num_joints,),
                dtype=np.float64,
            )

        if self.cameras_shapes:
            for cn, hwc_shape in self.cameras_shapes.items():
                # Assumes images are unsigned int8 in [0,255]
                observation_space[cn] = spaces.Box(
                    low=0,
                    high=255,
                    # height x width x channels (e.g. 480 x 640 x 3)
                    shape=hwc_shape,
                    dtype=np.uint8,
                )

        self.observation_space = spaces.Dict(observation_space)
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_joints,), dtype=np.float32)

        self._observation = {}
        self._terminated = False
        self.timestamps = []

    def _get_obs(self):
        qpos = self.follower.read_position()
        self._observation["agent_pos"] = pwm2pos(qpos)
        for cn, c in self.cameras.items():
            self._observation[cn] = capture_image(c, self.cameras_shapes[cn][1], self.cameras_shapes[cn][0])

        if self.record:
            action = self.leader.read_position()
            self._observation["leader_pos"] = pwm2pos(action)

    def reset(self, seed: int | None = None):
        # Reset the robot and sync the leader and follower if we are recording
        for _ in range(self.warmup_steps):
            self._get_obs()
            if self.record:
                self.follower.set_goal_pos(pos2pwm(self._observation["leader_pos"]))
        self._terminated = False
        info = {}
        self.timestamps = []
        return self._observation, info

    def step(self, action: np.ndarray = None):
        if self.timestamps:
            # wait the right amount of time to stay at the desired fps
            time.sleep(max(0, 1 / self.fps - (time.time() - self.timestamps[-1])))

        self.timestamps.append(time.time())

        # Get the observation
        self._get_obs()
        if self.record:
            # Teleoperate the leader
            self.follower.set_goal_pos(pos2pwm(self._observation["leader_pos"]))
        else:
            # Apply the action to the follower
            self.follower.set_goal_pos(pos2pwm(action))

        reward = 0
        terminated = truncated = self._terminated
        info = {"timestamp": self.timestamps[-1] - self.timestamps[0], "fps_error": False}

        # Check if we are able to keep up with the desired fps
        if len(self.timestamps) > 1 and (self.timestamps[-1] - self.timestamps[-2]) > 1 / (
            self.fps - self.fps_tolerance
        ):
            print(
                f"Error: recording fps {1 / (self.timestamps[-1] - self.timestamps[-2]):.5f} is lower"
                f" than min admited fps {(self.fps - self.fps_tolerance):.5f}"
                f" at frame {len(self.timestamps)}"
            )
            info["fps_error"] = True

        return self._observation, reward, terminated, truncated, info

    def render(self): ...

    def close(self):
        self.follower._disable_torque()
        if self.record:
            self.leader._disable_torque()
