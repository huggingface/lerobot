import logging
import threading
import time
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..config import RobotConfig
from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_moveit2 import MoveIt2Config
from .moveit2_interface import MoveIt2Interface

if TYPE_CHECKING:
    from rclpy.executors import Executor

logger = logging.getLogger(__name__)


class MoveIt2(Robot):

    config_class = cast(MoveIt2Config, RobotConfig)
    name = "moveit2"

    def __init__(self, config: MoveIt2Config):
        super().__init__(config)
        self.config = config
        self.moveit2_interface = MoveIt2Interface(config.moveit2_interface)
        self.cameras = make_cameras_from_configs(config.cameras)
        

        self.executor: Executor | None = None
        self.executor_thread: threading.Thread | None = None

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        all_joint_names = self.config.moveit2_interface.arm_joint_names + [self.config.moveit2_interface.gripper_joint_name]
        motor_state_ft = {
            f"{motor}.pos": float for motor in all_joint_names
        }
        return {**motor_state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "linear_vel_x": float,
            "linear_vel_y": float,
            "linear_vel_z": float,
            "angular_vel_x": float,
            "angular_vel_y": float,
            "angular_vel_z": float,
            "gripper_pos": float,
        }

    @property
    def is_connected(self) -> bool:
        return self.moveit2_interface.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        for cam in self.cameras.values():
            cam.connect()

        # TODO (Yifei): should this be done in the MoveIt2Interface?
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        rclpy.init()
        # Create and start the executor in a separate thread
        self.executor = MultiThreadedExecutor()
        self.moveit2_interface.connect(self.executor)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        time.sleep(3)  # Give some time to connect to services and receive messages

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass  # robot must be calibrated before running LeRobot

    def configure(self) -> None:
        pass # robot must be configured before running LeRobot

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}
        arm_obs = self._get_arm_observation()
        obs_dict.update({f"{joint}.pos": pos for joint, pos in arm_obs.items()})

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def _get_arm_observation(self) -> dict[str, float]:
        positions = {}
        arm = self.moveit2_interface
        if not arm.joint_state:
            raise ValueError("Joint state is not available yet.")

        name_to_index = {name: i for i, name in enumerate(arm.joint_state.name)}
        for joint_name in arm.config.arm_joint_names:
            idx = name_to_index.get(joint_name)
            if idx is None:
                raise ValueError(f"Joint '{joint_name}' not found in joint state.")
            positions[joint_name] = arm.joint_state.position[idx]

        idx = name_to_index.get(arm.config.gripper_joint_name)
        if idx is None:
            raise ValueError(f"Gripper joint '{arm.config.gripper_joint_name}' not found in joint state.")
        positions[arm.config.gripper_joint_name] = arm.joint_state.position[idx]
        return positions

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors.

        Raises:
            DeviceNotConnectedError: if robot is not connected.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.max_relative_target is not None:
            # We don't have the current velocity of the arm, so set it to 0.0
            # Effectively the goal velocity gets clipped by max_relative_target
            goal_present_vel = {key: (act, 0.0) for key, act in action.items()}
            action = ensure_safe_goal_position(goal_present_vel, self.config.max_relative_target)

        linear_vel = (
            action["linear_vel_x"],
            action["linear_vel_y"],
            action["linear_vel_z"],
        )
        angular_vel = (
            action["angular_vel_x"],
            action["angular_vel_y"],
            action["angular_vel_z"],
        )
        gripper_pos = action["gripper_pos"]
        self.moveit2_interface.servo(linear=linear_vel, angular=angular_vel)
        self.moveit2_interface.send_gripper_command(gripper_pos)
        return action

    def from_keyboard_to_action(self, pressed_keys: dict[str, Any]) -> dict[str, float]:
        """Convert pressed keys to action commands for teleop."""
        lin_vel_x = 0.0
        if "a" in pressed_keys:
            lin_vel_x -= 1.0
        if "d" in pressed_keys:
            lin_vel_x += 1.0
        lin_vel_y = 0.0
        if "w" in pressed_keys:
            lin_vel_y += 1.0
        if "s" in pressed_keys:
            lin_vel_y -= 1.0
        lin_vel_z = 0.0
        if "q" in pressed_keys:
            lin_vel_z -= 1.0
        if "e" in pressed_keys:
            lin_vel_z += 1.0

        ang_vel_x = 0.0
        if "j" in pressed_keys:
            ang_vel_x -= 1.0
        if "l" in pressed_keys:
            ang_vel_x += 1.0
        
        ang_vel_y = 0.0
        if "i" in pressed_keys:
            ang_vel_y += 1.0
        if "k" in pressed_keys:
            ang_vel_y -= 1.0
        ang_vel_z = 0.0
        if "u" in pressed_keys:
            ang_vel_z -= 1.0
        if "o" in pressed_keys:
            ang_vel_z += 1.0

        gripper_pos = 0.0
        if "g" in pressed_keys:
            gripper_pos = 1.0

        return {
            "linear_vel_x": lin_vel_x,
            "linear_vel_y": lin_vel_y,
            "linear_vel_z": lin_vel_z,
            "angular_vel_x": ang_vel_x,
            "angular_vel_y": ang_vel_y,
            "angular_vel_z": ang_vel_z,
            "gripper_pos": gripper_pos,
        }

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.moveit2_interface.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
