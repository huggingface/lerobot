"""
RC10 robot integration for HIL-SERL.

Wraps TaskSpaceJogController + Gripper + OpenCV cameras into the lerobot
Robot-like interface and provides a Gym environment for the training pipeline.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import gymnasium as gym
import numpy as np

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep


if TYPE_CHECKING:
    from rc10_api.controller import TaskSpaceJogController
    from rc10_api.gripper import Gripper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@RobotConfig.register_subclass("rc10")
@dataclass
class RC10RobotConfig(RobotConfig):
    """Connection and hardware configuration for the RC10 robot."""

    ip: str = "10.10.10.10"
    rate_hz: int = 100
    velocity: float = 1.0
    acceleration: float = 1.0
    threshold_position: float = 0.001
    threshold_angle: float = 1.0

    gripper_port: str = "/dev/ttyUSB0"
    gripper_baudrate: int = 115200

    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@dataclass
class RC10RobotEnvConfig:
    """Task-space control parameters for the RC10 environment."""

    # End-effector step sizes in metres per unit action (action range is [-1, 1])
    ee_step_sizes: dict[str, float] = field(
        default_factory=lambda: {"x": 0.002, "y": 0.002, "z": 0.002}
    )

    # Cartesian workspace bounds [x, y, z]
    ee_bounds_min: list[float] = field(default_factory=lambda: [-0.0771, 0.2554, 0.2296])
    ee_bounds_max: list[float] = field(default_factory=lambda: [0.2836, 0.6417, 0.4079])

    # Fixed orientation (pick tasks typically dont need orientation control)
    fixed_roll: float = 3.14159
    fixed_pitch: float = 0.0
    fixed_yaw: float = 0.0

    # Home TCP pose for reset [x, y, z, roll, pitch, yaw]
    home_tcp: list[float] = field(
        default_factory=lambda: [0.095, 0.35, 0.28, 3.14159, 0.0, 0.0]
    )
    reset_time_s: float = 7.0

    use_gripper: bool = True

    # Start-position randomization (metres). At each reset the home x,y,z are
    # perturbed uniformly in [-rand, +rand]. Set to 0.0 to disable.
    randomization_xy: float = 0.02   # 2 cm in x and y (matching paper)
    randomization_z: float = 0.0     # no z randomization by default


# ---------------------------------------------------------------------------
# RC10 Robot
# ---------------------------------------------------------------------------


class RC10Robot:
    """Interface for RC10, gripper, and camera system.

    Attributes:
        config (RC10RobotConfig): Config params for the robot and camera
        controller (TaskSpaceJogController | None): The jog controller instance
        gripper (Gripper | None): Gripper instance
        cameras (dict[str, OpenCVCamera]): Mapping of camera names to instances
    """

    def __init__(self, config: RC10RobotConfig):
        self.config = config
        self.controller: TaskSpaceJogController | None = None
        self.gripper: Gripper | None = None
        self.cameras: dict[str, OpenCVCamera] = {}
        self._connected = False

    # -- lifecycle ----------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        from rc10_api.controller import TaskSpaceJogController
        from rc10_api.gripper import Gripper

        logger.info("Connecting to RC10 at %s ...", self.config.ip)

        self.controller = TaskSpaceJogController(
            ip=self.config.ip,
            rate_hz=self.config.rate_hz,
            velocity=self.config.velocity,
            acceleration=self.config.acceleration,
            treshold_position=self.config.threshold_position,
            treshold_angel=self.config.threshold_angle,
        )
        self.controller.start()

        self.gripper = Gripper(
            device=self.config.gripper_port,
            baudrate=self.config.gripper_baudrate,
        )

        self.cameras = make_cameras_from_configs(self.config.cameras)
        for cam in self.cameras.values():
            cam.connect()

        self._connected = True
        logger.info("RC10 connected.")

    def disconnect(self) -> None:
        if self.controller is not None:
            self.controller.stop()
        if self.gripper is not None:
            self.gripper.close()
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        self._connected = False
        logger.info("RC10 disconnected.")

    # -- observation --------------------------------------------------------

    def get_observation(self) -> dict:
        """Collect and format the current environment state following the lerobot convention.
        Returns:
            dict: A dict containing
                - agent_pos (np.ndarray): A (16,) vector consisting of:
                    - joint_pos (6): Current joint positions
                    - joint_vel (6): Current joint velocities
                    - gripper_state (1): 1.0 if open, 0.0 if closed
                    - tcp_xyz (3): Tool center point [x, y, z] in meters
                - pixels (dict[str, np.ndarray]): A mapping of camera names to thier latest (H, W, 3) RGB image arrays
        """
        joint_pos = self.controller.get_current_joint()       # (6,)
        joint_vel = self.controller.get_current_joint_vel()    # (6,)
        joint_torques = self.controller.get_current_torque()   # (6,)
        tcp = self.controller.get_current_tcp()                # (6,) [x,y,z,r,p,y]
        gripper_state = float(self.gripper.is_open)            # 0.0 or 1.0

        agent_pos = np.concatenate([
            joint_pos,              # 6
            joint_vel,              # 6
            joint_torques,          # 6
            [gripper_state],        # 1
            tcp[:3],                # 3  (x, y, z)
        ]).astype(np.float32)       # total: 22

        pixels = {}
        for name, cam in self.cameras.items():
            pixels[name] = cam.async_read()

        return {"agent_pos": agent_pos, "pixels": pixels}

    # -- commands -----------------------------------------------------------

    def get_current_tcp(self) -> np.ndarray:
        """Current TCP pose.
        Returns:
            np.ndarray: A (6,) float array for pose: [x, y, z, roll, pitch, yaw]
                - x, y, z: Cartesian coorindnates in meters
                - roll, pitch, yaw: Euler angles in radians
        """
        return self.controller.get_current_tcp()

    def send_target(self, x: float, y: float, z: float,
                    roll: float, pitch: float, yaw: float) -> None:
        """Set absolute Cartesian target for the jog controller.
        Args:
            x, y, z (float): Target coordinates in meters
            roll, pitch, yaw (float): Target Euler angles in radians
        """
        self.controller.set_target(x, y, z, roll, pitch, yaw)

    def send_gripper(self, command: int) -> None:
        """Send discrete gripper command.
        Args:
            command (int):
                - 0 = close
                - 1 = stay (no-op)
                - 2 = open
        """
        if command == 0:
            self.gripper.send(-1)   # close
        elif command == 2:
            self.gripper.send(1)    # open
        # command == 1: do nothing


# ---------------------------------------------------------------------------
# RC10 Gym Environment
# ---------------------------------------------------------------------------


class RC10RobotEnv(gym.Env):
    """Gym environment for the RC10 robot with task-space delta control.

    Converts normalised delta actions [-1, 1] to absolute Cartesian targets
    and sends them to the RC10's TaskSpaceJogController.
    """

    def __init__(self, robot: RC10Robot, config: RC10RobotEnvConfig):
        super().__init__()
        self.robot = robot
        self.config = config

        self.ee_step = np.array([
            config.ee_step_sizes["x"],
            config.ee_step_sizes["y"],
            config.ee_step_sizes["z"],
        ], dtype=np.float32)
        self.ee_min = np.array(config.ee_bounds_min, dtype=np.float32)
        self.ee_max = np.array(config.ee_bounds_max, dtype=np.float32)

        self.use_gripper = config.use_gripper
        self.current_step = 0

        # -- spaces ---------------------------------------------------------
        # Action: 3D continuous (dx, dy, dz).
        # If use_gripper, the env also accepts a 4th element for discrete gripper.
        action_dim = 4 if self.use_gripper else 3
        low = np.concatenate([-np.ones(3), [0.0]]) if self.use_gripper else -np.ones(3)
        high = np.concatenate([np.ones(3), [2.0]]) if self.use_gripper else np.ones(3)
        self.action_space = gym.spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            shape=(action_dim,),
            dtype=np.float32,
        )

        # Observation: built from a sample
        if self.robot.is_connected:
            sample = self.robot.get_observation()
            obs_spaces = {
                OBS_STATE: gym.spaces.Box(
                    low=-10, high=10,
                    shape=sample["agent_pos"].shape,
                    dtype=np.float32,
                ),
            }
            for cam_name, img in sample["pixels"].items():
                obs_spaces[f"{OBS_IMAGES}.{cam_name}"] = gym.spaces.Box(
                    low=0, high=255, shape=img.shape, dtype=np.uint8,
                )
            self.observation_space = gym.spaces.Dict(obs_spaces)

    # -- gym interface ------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict, dict]:
        """Reset the environment and move the robot to the home pose

        Args:
            seed (int | None, optional): _description_. Defaults to None.
            options (dict[str, Any] | None, optional): _description_. Defaults to None.

        Returns:
            tuple[dict, dict]: _description_
        """


        super().reset(seed=seed, options=options)

        logger.info("Resetting RC10 to home TCP pose …")
        self.robot.send_gripper(2)  # open gripper

        home = list(self.config.home_tcp)  # copy so we don't mutate config

        # Randomize start position (paper: 2cm in x,y)
        rng = self.np_random  # seeded RNG from gym.Env
        rand_xy = self.config.randomization_xy
        rand_z = self.config.randomization_z
        if rand_xy > 0:
            home[0] += rng.uniform(-rand_xy, rand_xy)
            home[1] += rng.uniform(-rand_xy, rand_xy)
        if rand_z > 0:
            home[2] += rng.uniform(-rand_z, rand_z)

        # Clip to workspace bounds so randomization can't push outside
        home[0] = np.clip(home[0], self.ee_min[0], self.ee_max[0])
        home[1] = np.clip(home[1], self.ee_min[1], self.ee_max[1])
        home[2] = np.clip(home[2], self.ee_min[2], self.ee_max[2])

        logger.info(f"  Randomized start: x={home[0]:.4f}, y={home[1]:.4f}, z={home[2]:.4f}")
        self.robot.send_target(*home)
        precise_sleep(self.config.reset_time_s)

        self.current_step = 0
        obs = self.robot.get_observation()
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one environment step using the given action.

        Args:
            action (np.ndarray): The action to be take by the robot.
                Shape (3,): [dx, dy, dz] normalised to [-1, 1].
                Shape (4,): [dx, dy, dz, gripper_cmd] where gripper ∈ {0, 1, 2}.

        Returns:
            tuple: A tuple containing:
                - observation (dict): Current state of the environment:
                    - agent_pos (np.ndarray): A (16,) vector consisting of:
                        - joint_pos (6): Current joint positions
                        - joint_vel (6): Current joint velocities
                        - gripper_state (1): 1.0 if open, 0.0 if closed
                        - tcp_xyz (3): Tool center point [x, y, z] in meters
                    - pixels (dict[str, np.ndarray]): A mapping of camera names to thier latest (H, W, 3) RGB image arrays (pixels are synced/fresh frames)
                - reward (float): Amount of reward returned after previous action
                - terminated (bool): Whether the robot reached a terminal state
                - truncated (bool): Whether the episode was cutoff (for eg. time limit)
                - info (dict): Auxillary diagnostic information for debugging.

        """
        continuous = action[:3]
        gripper_cmd = int(round(action[3])) if len(action) > 3 else 1

        # Delta to absolute TCP
        tcp = self.robot.get_current_tcp()
        current_xyz = tcp[:3]
        delta_xyz = continuous * self.ee_step
        new_xyz = np.clip(current_xyz + delta_xyz, self.ee_min, self.ee_max)

        # Update: removed for usb insertion task 
        # Safety: hardcoded for avoid colliding with the sorter (we can use the accurate urdf of the env later to avoid collision with the physical env objects)
        # if new_xyz[1] >=0.535:
        #     new_xyz[2] = 0.25

        self.robot.send_target(
            float(new_xyz[0]), float(new_xyz[1]), float(new_xyz[2]),
            self.config.fixed_roll, self.config.fixed_pitch, self.config.fixed_yaw,
        )
        self.robot.send_gripper(gripper_cmd)

        obs = self.robot.get_observation()
        self.current_step += 1

        return obs, 0.0, False, False, {TeleopEvents.IS_INTERVENTION: False}

    def close(self):
        self.robot.disconnect()
