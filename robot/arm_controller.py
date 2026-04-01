"""
High-level arm controller that wraps LeRobot's robot + kinematics + motion executor
into a clean, task-oriented interface.

Usage::

    from robot.arm_controller import ArmController

    arm = ArmController.from_config("config.yaml")
    arm.connect()

    arm.home()
    arm.move_to_pose([0.2, 0.0, 0.15], speed=0.08)
    arm.set_gripper(width_mm=40)
    joints = arm.get_joint_positions()

    arm.disconnect()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# SO-100/101 arm joint names (excluding gripper)
SO100_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Default home joint angles (degrees) — safe stowed position for SO-100/101
DEFAULT_HOME_JOINTS_DEG = [0.0, -90.0, 90.0, -90.0, 0.0]

# SO-100 gripper physical range
GRIPPER_MAX_WIDTH_MM = 80.0


@dataclass
class ArmControllerConfig:
    """Configuration for ArmController."""

    urdf_path: str = "SO101/so101_new_calib.urdf"
    ee_frame: str = "gripper_frame_link"
    motor_names: list[str] | None = None
    home_joints_deg: list[float] | None = None
    gripper_max_width_mm: float = GRIPPER_MAX_WIDTH_MM
    # Motion parameters
    cartesian_step_m: float = 0.003
    min_steps_per_segment: int = 12
    inter_step_sleep_s: float = 0.05
    settle_timeout_s: float = 4.5
    settle_threshold_deg: float = 2.0


class ArmController:
    """High-level interface for commanding a serial robot arm.

    Wraps LeRobot's Robot, RobotKinematics, and MotionExecutionConfig
    into the simple interface:
        move_to_pose, set_gripper, get_joint_positions, home, emergency_stop
    """

    def __init__(
        self,
        robot: Any,
        kinematics: Any | None = None,
        config: ArmControllerConfig | None = None,
    ):
        self.robot = robot
        self.config = config or ArmControllerConfig()
        self.motor_names = self.config.motor_names or list(SO100_MOTOR_NAMES)
        self.home_joints = np.array(
            self.config.home_joints_deg or DEFAULT_HOME_JOINTS_DEG,
            dtype=np.float64,
        )
        self._kinematics = kinematics
        self._connected = False

    @staticmethod
    def from_config(config_path: str) -> "ArmController":
        """Create an ArmController from a YAML config file.

        Args:
            config_path: Path to config.yaml.

        Returns:
            An unconnected ArmController ready for ``connect()``.
        """
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        robot_cfg = cfg.get("robot", {})
        arm_cfg = ArmControllerConfig(
            urdf_path=cfg.get("robot", {}).get("urdf", "SO101/so101_new_calib.urdf"),
            ee_frame=cfg.get("robot", {}).get("ee_frame", "gripper_frame_link"),
            home_joints_deg=robot_cfg.get("home_joints"),
            gripper_max_width_mm=robot_cfg.get("gripper_max_width_mm", GRIPPER_MAX_WIDTH_MM),
        )

        # Build the LeRobot robot instance
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower import SOFollowerRobotConfig

        cameras_raw = robot_cfg.get("cameras", {})
        cameras: dict = {}
        for cam_name, cam_conf in cameras_raw.items():
            cam_type = cam_conf.pop("type", "oakd")
            if cam_type == "oakd":
                from lerobot.cameras.oakd.configuration_oakd import OAKDCameraConfig
                cameras[cam_name] = OAKDCameraConfig(**cam_conf)
            elif cam_type == "opencv":
                from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
                cameras[cam_name] = OpenCVCameraConfig(**cam_conf)
            elif cam_type == "realsense":
                from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
                cameras[cam_name] = RealSenseCameraConfig(**cam_conf)

        robot_config = SOFollowerRobotConfig(
            port=robot_cfg.get("port", ""),
            cameras=cameras,
            use_degrees=robot_cfg.get("use_degrees", True),
        )
        robot = make_robot_from_config(robot_config)

        return ArmController(robot=robot, config=arm_cfg)

    @property
    def kinematics(self):
        """Lazy-load kinematics (requires placo)."""
        if self._kinematics is None:
            from lerobot.model.kinematics import RobotKinematics

            self._kinematics = RobotKinematics(
                urdf_path=self.config.urdf_path,
                target_frame_name=self.config.ee_frame,
            )
        return self._kinematics

    @property
    def motion_config(self):
        from lerobot.utils.motion_executor import MotionExecutionConfig

        return MotionExecutionConfig(
            use_cartesian_interp=True,
            cartesian_step_m=self.config.cartesian_step_m,
            min_steps_per_segment=self.config.min_steps_per_segment,
            inter_step_sleep_s=self.config.inter_step_sleep_s,
            settle_last_step=True,
            settle_timeout_s=self.config.settle_timeout_s,
            settle_threshold_deg=self.config.settle_threshold_deg,
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the robot arm and cameras."""
        self.robot.connect(calibrate=calibrate)
        self._connected = True
        logger.info("ArmController connected.")

    def disconnect(self) -> None:
        """Disconnect from the robot arm."""
        if self._connected:
            self.robot.disconnect()
            self._connected = False
            logger.info("ArmController disconnected.")

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def move_to_pose(
        self,
        position: list[float] | np.ndarray,
        orientation: list[float] | np.ndarray | None = None,
        speed: float = 0.1,
    ) -> bool:
        """Move end-effector to a target pose in robot base frame.

        Args:
            position: [x, y, z] in meters (robot base frame).
            orientation: Quaternion [qx, qy, qz, qw] or None for top-down.
            speed: Fraction of max speed (0.01 = slow, 1.0 = fast).
                   Maps to cartesian_step_m scaling.

        Returns:
            True if motion completed without timeout.
        """
        from lerobot.perception.grasp_planner import Waypoint
        from lerobot.utils.motion_executor import execute_waypoint

        pos = np.asarray(position, dtype=np.float64)

        # Build 4x4 target pose
        T_target = np.eye(4, dtype=np.float64)
        T_target[:3, 3] = pos

        if orientation is not None:
            quat = np.asarray(orientation, dtype=np.float64)
            # scipy expects [x, y, z, w]
            T_target[:3, :3] = Rotation.from_quat(quat).as_matrix()
        else:
            # Default: top-down (gripper pointing down)
            T_target[:3, :3] = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ])

        wp = Waypoint(
            pose_4x4=T_target,
            gripper_open=True,
            label="move",
            gripper_width_pct=100.0,
        )

        # Scale motion config by speed factor
        motion = self.motion_config
        motion.cartesian_step_m = self.config.cartesian_step_m * max(0.1, min(speed * 3.0, 3.0))
        motion.inter_step_sleep_s = self.config.inter_step_sleep_s / max(0.1, speed)

        current_joints = self.get_joint_positions()
        execute_waypoint(
            self.robot,
            self.kinematics,
            wp,
            self.motor_names,
            motion,
            current_joints=current_joints,
        )
        return True

    def set_gripper(self, width_mm: float) -> None:
        """Set gripper opening width in millimeters.

        Args:
            width_mm: 0 = fully closed, gripper_max_width_mm = fully open.
        """
        pct = np.clip(width_mm / self.config.gripper_max_width_mm * 100.0, 0.0, 100.0)
        action = {f"{m}.pos": float(self.get_joint_positions()[i]) for i, m in enumerate(self.motor_names)}
        action["gripper.pos"] = float(pct)
        self.robot.send_action(action)
        time.sleep(0.3)  # allow gripper to reach target

    def get_joint_positions(self) -> np.ndarray:
        """Return current joint angles in degrees (arm joints only, no gripper).

        Returns:
            numpy array of shape (n_joints,) in degrees.
        """
        obs = self.robot.get_observation()
        return np.array(
            [float(obs[f"{m}.pos"]) for m in self.motor_names],
            dtype=np.float64,
        )

    def get_gripper_position(self) -> float:
        """Return current gripper opening as percentage (0=closed, 100=open)."""
        obs = self.robot.get_observation()
        return float(obs.get("gripper.pos", 0.0))

    def get_ee_pose(self) -> np.ndarray:
        """Return current end-effector pose as 4x4 matrix in robot base frame."""
        joints = self.get_joint_positions()
        return self.kinematics.forward_kinematics(joints)

    def home(self) -> None:
        """Move arm to the home/stowed position."""
        logger.info("Moving to home position...")
        action = {}
        for i, m in enumerate(self.motor_names):
            action[f"{m}.pos"] = float(self.home_joints[i])
        action["gripper.pos"] = 100.0  # open gripper
        self.robot.send_action(action)

        # Wait for convergence
        from lerobot.utils.motion_executor import wait_for_convergence

        wait_for_convergence(
            self.robot,
            action,
            timeout_s=5.0,
            threshold_deg=3.0,
        )
        logger.info("Home position reached.")

    def emergency_stop(self) -> None:
        """Immediately disable torque on all motors.

        This causes the arm to go limp — ensure it won't fall on anything.
        """
        logger.warning("EMERGENCY STOP — disabling torque on all motors.")
        try:
            if hasattr(self.robot, "bus"):
                self.robot.bus.disable_torque()
        except Exception as e:
            logger.error("Failed to disable torque: %s", e)
        try:
            self.robot.disconnect()
        except Exception:
            pass
        self._connected = False

    # ------------------------------------------------------------------
    # Higher-level manipulation
    # ------------------------------------------------------------------

    def pick(
        self,
        position: list[float] | np.ndarray,
        object_size: list[float] | np.ndarray | None = None,
        object_label: str = "object",
    ) -> bool:
        """Execute a full pick sequence at the given position.

        Uses the GraspPlanner to generate waypoints, then executes them.

        Args:
            position: [x, y, z] target in robot base frame (meters).
            object_size: [sx, sy, sz] object dimensions in meters.
            object_label: Label for grasp strategy selection.

        Returns:
            True if pick sequence completed.
        """
        from lerobot.perception.grasp_planner import GraspPlanner
        from lerobot.utils.motion_executor import execute_waypoints

        size = np.asarray(object_size or [0.04, 0.04, 0.04], dtype=np.float64)
        center = np.asarray(position, dtype=np.float64)

        planner = GraspPlanner()
        waypoints = planner.plan_pick(
            object_center_xyz=center,
            object_size_xyz=size,
            object_label=object_label,
        )

        current_joints = self.get_joint_positions()
        execute_waypoints(
            self.robot,
            self.kinematics,
            waypoints,
            self.motor_names,
            self.motion_config,
        )
        return True

    def place(self, position: list[float] | np.ndarray) -> bool:
        """Execute a place sequence at the given position.

        Args:
            position: [x, y, z] target in robot base frame (meters).

        Returns:
            True if place sequence completed.
        """
        from lerobot.perception.grasp_planner import GraspPlanner
        from lerobot.utils.motion_executor import execute_waypoints

        target = np.asarray(position, dtype=np.float64)

        planner = GraspPlanner()
        waypoints = planner.plan_place(target_xyz=target)

        execute_waypoints(
            self.robot,
            self.kinematics,
            waypoints,
            self.motor_names,
            self.motion_config,
        )
        return True
