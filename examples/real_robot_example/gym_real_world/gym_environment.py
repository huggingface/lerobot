import time

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dynamixel import pos2pwm, pwm2pos
from .robot import Robot

FPS = 30

CAMERAS_SHAPES = {
    "observation.images.high": (480, 640, 3),
    "observation.images.low": (480, 640, 3),
}

CAMERAS_PORTS = {
    "observation.images.high": "/dev/video6",
    "observation.images.low": "/dev/video0",
}

LEADER_PORT = "/dev/ttyACM1"
FOLLOWER_PORT = "/dev/ttyACM0"


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
    ):
        self.num_joints = num_joints
        self.cameras_shapes = cameras_shapes
        self.cameras_ports = cameras_ports
        self.warmup_steps = warmup_steps
        assert len(self.cameras_shapes) == len(self.cameras_ports), "Number of cameras and shapes must match."

        self.follower_port = follower_port
        self.leader_port = leader_port
        self.record = record

        # Initialize the robot
        self.follower = Robot(device_name=self.follower_port)
        if self.record:
            self.leader = Robot(device_name=self.leader_port)
            self.leader.set_trigger_torque(trigger_torque)

        # Initialize the cameras - sorted by camera names
        self.cameras = {}
        for cn, p in sorted(self.cameras_ports.items()):
            assert cn.startswith("observation.images."), "Camera names must start with 'observation.images.'."
            self.cameras[cn] = cv2.VideoCapture(p)
        if not all(c.isOpened() for c in self.cameras.values()):
            raise OSError("Cannot open all camera ports.")

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
                observation_space[f"images.{cn}"] = spaces.Box(
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
        self._action_time = time.time()

    def _get_obs(self):
        qpos = self.follower.read_position()
        self._observation["agent_pos"] = pwm2pos(qpos)
        for cn, c in self.cameras.items():
            self._observation[f"images.{cn}"] = capture_image(
                c, self.cameras_shapes[cn][1], self.cameras_shapes[cn][0]
            )

        if self.record:
            leader_pos = self.leader.read_position()
            self._observation["leader_pos"] = pwm2pos(leader_pos)

    def reset(self, seed: int | None = None):
        del seed
        # Reset the robot and sync the leader and follower if we are recording
        for _ in range(self.warmup_steps):
            self._get_obs()
            if self.record:
                self.follower.set_goal_pos(pos2pwm(self._observation["leader_pos"]))
        self._terminated = False
        info = {}
        return self._observation, info

    def step(self, action: np.ndarray = None):
        # Reset the observation
        self._get_obs()
        if self.record:
            # Teleoperate the leader
            self.follower.set_goal_pos(pos2pwm(self._observation["leader_pos"]))
        else:
            # Apply the action to the follower
            self.follower.set_goal_pos(pos2pwm(action))
        reward = 0
        terminated = truncated = self._terminated
        info = {}
        return self._observation, reward, terminated, truncated, info

    def render(self): ...

    def close(self):
        self.follower._disable_torque()
        if self.record:
            self.leader._disable_torque()
