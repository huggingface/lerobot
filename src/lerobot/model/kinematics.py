# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils import require_package

_placo_runtime_error: ImportError | None = None

if TYPE_CHECKING:
    import placo
else:
    try:
        import placo  # type: ignore[import-not-found]
    except ImportError as _placo_import_err:
        placo = None
        _placo_runtime_error = _placo_import_err


def _raise_if_placo_unusable() -> None:
    if placo is None and _placo_runtime_error is not None:
        raise ImportError(
            f"placo is installed but failed to import: {_placo_runtime_error!s}"
        ) from _placo_runtime_error


def _load_robot(urdf_path: str | Path) -> placo.RobotWrapper:
    require_package("placo", extra="placo-dep")
    _raise_if_placo_unusable()
    # Kinematics and gravity don't need collision geometry, whose loading also logs every colliding pair.
    return placo.RobotWrapper(str(urdf_path), placo.Flags.ignore_collisions)


class RobotKinematics:
    """Forward and inverse kinematics of one or several end-effector frames, computed with Placo."""

    def __init__(
        self,
        urdf_path: str | Path,
        target_frame_name: str | Sequence[str] = "gripper_frame_link",
        joint_names: Sequence[str] | None = None,
        posture_weight: float = 0.0,
    ):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path (`str | Path`):
                Path to the robot URDF file.
            target_frame_name (`str | Sequence[str]`, *optional*, defaults to `"gripper_frame_link"`):
                End-effector frame, or several frames solved together, e.g. both hands of a humanoid. The
                first one is the default frame of `forward_kinematics`.
            joint_names (`Sequence[str]`, *optional*):
                Joints that the IK moves, in the order of joint positions; other joints keep their position.
                Defaults to every joint of the model.
            posture_weight (`float`, *optional*, defaults to 0.0):
                Weight, relative to the position targets, of a pull of the joints toward zero. It picks the
                posture closest to zero among the solutions of a redundant arm; 0.0 disables it.
        """
        self.robot = _load_robot(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base

        self.joint_names = list(self.robot.joint_names()) if joint_names is None else list(joint_names)
        for joint_name in self.robot.joint_names():
            if joint_name not in self.joint_names:
                self.solver.mask_dof(joint_name)

        frames = [target_frame_name] if isinstance(target_frame_name, str) else list(target_frame_name)
        self.target_frame_name = frames[0]
        self.frame_tasks = {frame: self.solver.add_frame_task(frame, np.eye(4)) for frame in frames}
        self.tip_frame = self.frame_tasks[self.target_frame_name]

        if posture_weight > 0:
            posture = self.solver.add_joints_task()
            posture.set_joints(dict.fromkeys(self.joint_names, 0.0))
            posture.configure("posture", "soft", posture_weight)

    def forward_kinematics(self, joint_pos_deg: np.ndarray, frame_name: str | None = None) -> np.ndarray:
        """
        Compute the pose of an end-effector frame for a joint configuration.

        Args:
            joint_pos_deg (`np.ndarray`):
                Positions of `joint_names` in degrees; extra trailing values, such as a gripper, are ignored.
            frame_name (`str`, *optional*):
                Frame to return; defaults to the first target frame.

        Returns:
            `np.ndarray`: 4x4 transformation matrix of the frame in the world.
        """
        self._set_joints(joint_pos_deg)
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(frame_name or self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray | Mapping[str, np.ndarray],
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        max_iters: int = 8,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using placo solver.

        Args:
            current_joint_pos (`np.ndarray`):
                Current joint positions in degrees, used as the initial guess.
            desired_ee_pose (`np.ndarray | Mapping[str, np.ndarray]`):
                Target pose of the end-effector frame as a 4x4 transformation matrix, or one target pose per
                frame when several frames are solved together.
            position_weight (`float`, *optional*, defaults to 1.0):
                Weight of the position targets.
            orientation_weight (`float`, *optional*, defaults to 0.01):
                Weight of the orientation targets; 0.0 constrains only the positions.
            max_iters (`int`, *optional*, defaults to 8):
                Number of placo Newton steps to run.

        Returns:
            `np.ndarray`: Joint positions in degrees that reach the target poses.

        Raises:
            ValueError: If the target poses don't match the target frames.
        """
        targets = (
            desired_ee_pose
            if isinstance(desired_ee_pose, Mapping)
            else {self.target_frame_name: desired_ee_pose}
        )
        if set(targets) != set(self.frame_tasks):
            raise ValueError(f"Expected target poses for {list(self.frame_tasks)}; got {list(targets)}")

        self._set_joints(current_joint_pos)
        for frame, pose in targets.items():
            task = self.frame_tasks[frame]
            task.T_world_frame = pose
            task.configure(frame, "soft", position_weight, orientation_weight)

        # Solve IK.
        for _ in range(max_iters):
            self.solver.solve(True)
            self.robot.update_kinematics()

        # Extract joint positions
        joint_pos_rad = []
        for joint_name in self.joint_names:
            joint = self.robot.get_joint(joint_name)
            joint_pos_rad.append(joint)

        # Convert back to degrees
        joint_pos_deg = np.rad2deg(joint_pos_rad)

        # Preserve gripper position if present in current_joint_pos
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return joint_pos_deg

    def _set_joints(self, joint_pos_deg: np.ndarray) -> None:
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])
        for joint_name, position in zip(self.joint_names, joint_pos_rad, strict=True):
            self.robot.set_joint(joint_name, float(position))


class GravityCompensation:
    """Joint torques that hold a fixed-base URDF model against gravity, computed with Placo.

    Safe to call from several threads.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        joint_names: Sequence[str],
        base_frame: str,
        scale: Sequence[float] | None = None,
        max_torque: float | None = None,
    ) -> None:
        """
        Load a robot model for gravity compensation.

        Args:
            urdf_path (`str | Path`):
                Path to the URDF model.
            joint_names (`Sequence[str]`):
                Model joints to compensate, in the order torques are returned.
            base_frame (`str`):
                Name of the fixed base frame in the model.
            scale (`Sequence[float]`, *optional*):
                Per-joint multipliers of the model's torques, to correct for payloads or model errors.
            max_torque (`float`, *optional*):
                Cap on each joint's torque, in Nm, applied after `scale`.
        """
        self.robot = _load_robot(urdf_path)
        self.joint_names = list(joint_names)
        self.base_frame = base_frame
        self.scale = None if scale is None else np.asarray(scale, dtype=float)
        if self.scale is not None and self.scale.shape != (len(self.joint_names),):
            raise ValueError(f"scale must have one value per joint; got {self.scale.shape}")
        self.max_torque = max_torque
        self._lock = threading.Lock()

    def torques(
        self, joint_pos_rad: np.ndarray, extra_joint_pos: dict[str, float] | None = None
    ) -> np.ndarray:
        """
        Compute the torques that hold a configuration against gravity.

        Args:
            joint_pos_rad (`np.ndarray`):
                Positions of `joint_names` in radians.
            extra_joint_pos (`dict[str, float]`, *optional*):
                Positions of other model joints whose mass still loads the compensated joints (e.g. gripper
                fingers), in model units.

        Returns:
            `np.ndarray`: Torques in Nm for each joint in `joint_names`, scaled and capped.
        """
        with self._lock:
            for joint_name, position in zip(self.joint_names, joint_pos_rad, strict=True):
                self.robot.set_joint(joint_name, float(position))
            for joint_name, position in (extra_joint_pos or {}).items():
                self.robot.set_joint(joint_name, float(position))
            self.robot.update_kinematics()
            torques_by_joint = self.robot.static_gravity_compensation_torques_dict(self.base_frame)
        torques = np.asarray([torques_by_joint[joint_name] for joint_name in self.joint_names])
        if self.scale is not None:
            torques = torques * self.scale
        if self.max_torque is not None:
            torques = np.clip(torques, -self.max_torque, self.max_torque)
        return torques
