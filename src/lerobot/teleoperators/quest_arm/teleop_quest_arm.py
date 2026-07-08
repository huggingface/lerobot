#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quest VR arm teleoperator — bridges QuestArmTeleop into LeRobot."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .configuration_quest_arm import QuestArmTeleopConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure helpers (reuses logic from QuestArmTeleop without modifying that repo)
# ---------------------------------------------------------------------------


def _xyzrpy_to_mat(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    mat[:3, 3] = np.array([x, y, z])
    return mat


_ADJ_MAT = np.array(
    [[0.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=float,
)
# Hand controller orientation correction (matches pub_pose.py r_adj)
_R_ADJ = _xyzrpy_to_mat(0.0, 0.0, 0.0, -np.pi, 0.0, -np.pi / 2)


def _mat2xyzquat(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = matrix[:3, 3]
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()  # x, y, z, w
    return pos, quat


def _calc_pose_incre(
    start_pose_matrix: np.ndarray,
    current_pose_xyzrpy: np.ndarray,
    zero_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Delta pose: zero @ inv(start) @ current  →  (xyz, xyzquat)."""
    end_matrix = _xyzrpy_to_mat(*current_pose_xyzrpy)
    result_matrix = zero_matrix @ np.linalg.inv(start_pose_matrix) @ end_matrix
    return _mat2xyzquat(result_matrix)


def _correct_to_ros(transform: np.ndarray, ros_to_arm_mat: np.ndarray) -> np.ndarray:
    """OpenXR → ROS → arm coordinate frame."""
    transform = _ADJ_MAT @ transform
    transform = transform @ _R_ADJ
    transform = transform @ ros_to_arm_mat
    return transform


# ---------------------------------------------------------------------------
# QuestArmTeleop
# ---------------------------------------------------------------------------

_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


class QuestArmTeleop(Teleoperator):
    """Meta Quest VR arm teleoperator for NERO via QuestArmTeleop.

    Reads Quest controller poses over ADB, computes delta EE targets,
    solves IK with Pinocchio/CasADi, and outputs joint position commands.
    """

    config_class = QuestArmTeleopConfig
    name = "quest_arm"

    def __init__(self, config: QuestArmTeleopConfig):
        super().__init__(config)
        self.config = config

        self._oculus_reader = None
        self._arm_ik = None
        self._connected = False

        # State machine
        self._teleop_active = False
        self._zero_matrix = np.eye(4)
        self._start_pose_matrix = np.eye(4)

        # Current VR pose (after coordinate correction)
        self._current_pose_xyzrpy: np.ndarray | None = None

        # Build ros_to_arm transform matrix
        self._ros_to_arm_mat = _xyzrpy_to_mat(
            float(config.ros_to_arm_xyz[0]),
            float(config.ros_to_arm_xyz[1]),
            float(config.ros_to_arm_xyz[2]),
            float(config.ros_to_arm_rpy[0]),
            float(config.ros_to_arm_rpy[1]),
            float(config.ros_to_arm_rpy[2]),
        )

        # Resolve QuestArmTeleop scripts path
        self._scripts_path = self._resolve_scripts_path(config)

        # Pre-import (lazy — actual OculusReader created in connect())
        self._setup_imports()

    def _resolve_scripts_path(self, config: QuestArmTeleopConfig) -> Path:
        if config.quest_arm_teleop_path:
            return Path(config.quest_arm_teleop_path)
        # Auto-detect from standard location relative to this repo
        candidate = Path(__file__).resolve().parents[4] / "nero" / "QuestArmTeleop" / "src" / "oculus_reader" / "scripts"
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(
            "Cannot auto-detect QuestArmTeleop scripts path. "
            "Please set --teleop.quest_arm_teleop_path=/path/to/QuestArmTeleop/src/oculus_reader/scripts"
        )

    def _setup_imports(self) -> None:
        scripts_str = str(self._scripts_path)
        if scripts_str not in sys.path:
            sys.path.insert(0, scripts_str)

    # -- Teleoperator interface ------------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in _JOINT_NAMES + ["gripper"]}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in _JOINT_NAMES}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True  # VR controller — no calibration needed

    def calibrate(self) -> None:
        pass  # No calibration for VR

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        from oculus_reader import OculusReader

        logger.info("Connecting to Quest VR headset...")
        self._oculus_reader = OculusReader(ip_address=self.config.ip_address, run=True)
        time.sleep(0.5)  # Wait for first data

        self._init_ik_solver()
        self._connected = True
        logger.info("QuestArmTeleop connected.")

    def _init_ik_solver(self) -> None:
        if self.config.urdf_path is None:
            raise ValueError(
                "QuestArmTeleop requires --teleop.urdf_path=<path/to/nero.urdf>. "
                "e.g. /home/yuhang/projects/lerobot/nero/QuestArmTeleop/src/agx_arm_ros/"
                "src/agx_arm_description/agx_arm_urdf/nero/urdf/nero_with_gripper_description.urdf"
            )

        # arm_ik_pose_node imports rclpy (ROS2) at the top level, but ArmIK itself
        # only needs pinocchio + casadi. Mock out ROS2 modules to avoid the C extension error.
        from unittest.mock import MagicMock

        _ros2_modules_to_mock = [
            "rclpy", "rclpy.node", "rclpy.executors",
            "rclpy.executors.multi_threaded_executor",
            "rcl_interfaces", "rcl_interfaces.msg",
            "geometry_msgs", "geometry_msgs.msg",
            "sensor_msgs", "sensor_msgs.msg",
            "std_msgs", "std_msgs.msg",
            "ament_index_python", "ament_index_python.packages",
        ]
        saved = {}
        for mod_name in _ros2_modules_to_mock:
            saved[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = MagicMock()

        try:
            from arm_ik_pose_node import ArmIK
        finally:
            # Restore original modules
            for mod_name in _ros2_modules_to_mock:
                if saved[mod_name] is None:
                    sys.modules.pop(mod_name, None)
                else:
                    sys.modules[mod_name] = saved[mod_name]

        urdf_path = str(self.config.urdf_path)
        package_dirs = [self.config.package_dirs] if self.config.package_dirs else []

        self._arm_ik = ArmIK(
            urdf_path=urdf_path,
            package_dirs=package_dirs,
            locked_joints=list(self.config.locked_joints),
            ee_parent_joint=self.config.ee_parent_joint,
            ee_frame_name=self.config.ee_frame_name,
            tool_pre_rot_rpy=list(self.config.tool_pre_rot_rpy),
            tool_translation_xyz=list(self.config.tool_translation_xyz),
            collision_pairs_flat=[],
            w_pos=self.config.w_pos,
            w_ori=self.config.w_ori,
            w_reg=self.config.w_reg,
            w_smooth=self.config.w_smooth,
            ipopt_max_iter=self.config.ipopt_max_iter,
            ipopt_tol=self.config.ipopt_tol,
            enable_visualization=False,
            viewer_open_browser=False,
            viewer_model_name="pinocchio",
            viewer_target_frame_name="ee_target",
            viewer_axis_length=0.1,
            viewer_axis_width=10.0,
        )
        logger.info(f"ArmIK initialized: nq={self._arm_ik.nq}, joints={self._arm_ik.active_joint_names()}")

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        transforms, buttons = self._oculus_reader.get_transformations_and_buttons()

        if not transforms or not buttons:
            return self._hold_action()

        # Select hand
        hand_key = "r" if self.config.hand_name == "right" else "l"
        raw_transform = transforms.get(hand_key)
        if raw_transform is None:
            return self._hold_action()

        # Coordinate transform: OpenXR → ROS → arm
        corrected = _correct_to_ros(raw_transform, self._ros_to_arm_mat)
        xyz = corrected[:3, 3]
        rpy = Rotation.from_matrix(corrected[:3, :3]).as_euler("xyz")
        self._current_pose_xyzrpy = np.array([xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]])

        # Button state machine
        if buttons.get(self.config.start_button, False):
            if not self._teleop_active:
                logger.info(f"[{self.config.hand_name}] VR teleop started")
            self._start_pose_matrix = _xyzrpy_to_mat(*self._current_pose_xyzrpy)
            self._teleop_active = True

        if buttons.get(self.config.stop_button, False):
            if self._teleop_active:
                logger.info(f"[{self.config.hand_name}] VR teleop stopped")
            self._teleop_active = False
            # Reset zero to current pose so next start is stable
            self._zero_matrix = _xyzrpy_to_mat(*self._current_pose_xyzrpy)
            self._start_pose_matrix = self._zero_matrix

        # Gripper from trigger
        trigger_raw = buttons.get(self.config.trigger_axis, [0.0])
        trigger_value = float(trigger_raw[0]) if isinstance(trigger_raw, (list, tuple)) and trigger_raw else 0.0
        gripper_value = max(self.config.gripper_min, min(trigger_value, 1.0) * self.config.gripper_max)

        # If not active, hold current joint positions (return empty → robot holds)
        if not self._teleop_active:
            return self._hold_action()

        # Compute delta pose
        _xyz, _quat = _calc_pose_incre(self._start_pose_matrix, self._current_pose_xyzrpy, self._zero_matrix)

        # Build 4x4 target matrix for IK
        target_pose = np.eye(4)
        target_pose[:3, 3] = _xyz
        target_pose[:3, :3] = Rotation.from_quat(_quat).as_matrix()

        # IK solve
        try:
            joint_angles = self._arm_ik.solve(target_pose)
        except Exception as e:
            logger.warning(f"IK solve failed: {e}")
            return self._hold_action()

        # Build action dict
        action: RobotAction = {}
        for i, name in enumerate(_JOINT_NAMES):
            action[f"{name}.pos"] = float(joint_angles[i])
        action["gripper.pos"] = float(gripper_value)

        return action

    def _hold_action(self) -> RobotAction:
        """Return empty action (robot holds current position)."""
        return {}

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Sync actual robot joint state into IK solver for better continuity."""
        if self._arm_ik is None:
            return
        q_current = []
        for name in _JOINT_NAMES:
            key = f"{name}.pos"
            if key in feedback:
                q_current.append(float(feedback[key]))
        if len(q_current) == self._arm_ik.nq:
            self._arm_ik.sync_state(q_current)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._oculus_reader is not None:
            self._oculus_reader.stop()
            self._oculus_reader = None
        self._connected = False
        self._teleop_active = False
        logger.info("QuestArmTeleop disconnected.")
