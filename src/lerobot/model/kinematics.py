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

from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import require_package

_placo_runtime_error: ImportError | None = None

if TYPE_CHECKING:
    import placo  # type: ignore[import-not-found]
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


class RobotKinematics:
    """Robot kinematics using placo library for forward and inverse kinematics."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path (str): Path to the robot URDF file
            target_frame_name (str): Name of the end-effector frame in the URDF
            joint_names (list[str] | None): List of joint names to use for the kinematics solver
        """
        require_package("placo", extra="placo-dep")
        _raise_if_placo_unusable()

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base

        self.target_frame_name = target_frame_name

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration given the target frame name in the constructor.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """

        # Convert degrees to radians
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])

        # Update joint positions in placo robot
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

        # Update kinematics
        self.robot.update_kinematics()

        # Get the transformation matrix
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using placo solver.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK, set to 0.0 to only constrain position

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """

        # Convert current joint positions to radians for initial guess
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])

        # Set current joint positions as initial guess
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_rad[i])

        # Update the target pose for the frame task
        self.tip_frame.T_world_frame = desired_ee_pose

        # Configure the task based on position_only flag
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)

        # Solve IK
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


class RobotKinematicsDLS:
    """Robot kinematics using Damped Least Squares (DLS) for inverse kinematics.

    Uses pinocchio for forward kinematics and Jacobian computation.
    Drop-in replacement for RobotKinematics (placo-based), same interface.

    The DLS method adds a damping factor to the pseudo-inverse to handle
    singularities more robustly than standard Newton-based IK:

        Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ · e

    where J is the Jacobian, λ is the damping factor, and e is the 6D pose error.
    """

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        """
        Initialize pinocchio-based DLS kinematics solver.

        Args:
            urdf_path: Path to the robot URDF file.
            target_frame_name: Name of the end-effector frame in the URDF.
            joint_names: List of joint names to use. If None, uses all revolute joints.
        """
        try:
            import pinocchio as pin  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "pinocchio is required for RobotKinematicsDLS. "
                "Install it with: pip install pin"
            ) from e

        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.target_frame_id = self.model.getFrameId(target_frame_name)

        if joint_names is not None:
            self.joint_names = list(joint_names)
        else:
            # Auto-detect all joints (skip "universe" at index 0)
            self.joint_names = [
                self.model.names[i]
                for i in range(1, self.model.njoints)
                if self.model.joints[i].nq > 0
            ]

        # Map joint names → velocity indices for Jacobian column extraction
        self._joint_v_indices = []
        for name in self.joint_names:
            jid = self.model.getJointId(name)
            self._joint_v_indices.append(self.model.joints[jid].idx_v)

        self._q = pin.neutral(self.model)

    def _set_joints(self, q: np.ndarray, joint_pos_deg: np.ndarray) -> None:
        """Write joint angles (degrees) into the q vector for our selected joints."""
        rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])
        for i, name in enumerate(self.joint_names):
            jid = self.model.getJointId(name)
            q[self.model.joints[jid].idx_q] = rad[i]

    def _get_joints(self, q: np.ndarray) -> np.ndarray:
        """Read joint angles from q vector, return in degrees."""
        rad = np.array([q[self.model.joints[self.model.getJointId(n)].idx_q] for n in self.joint_names])
        return np.rad2deg(rad)

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration.

        Args:
            joint_pos_deg: Joint positions in degrees.

        Returns:
            4x4 transformation matrix of the end-effector pose.
        """
        self._set_joints(self._q, joint_pos_deg)
        self._pin.forwardKinematics(self.model, self.data, self._q)
        self._pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.target_frame_id].homogeneous

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        damping: float = 0.02,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using Damped Least Squares.

        Args:
            current_joint_pos: Current joint positions in degrees (initial guess).
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix.
            position_weight: Weight for position error. Default 1.0.
            orientation_weight: Weight for orientation error. Default 0.01.
            max_iter: Maximum DLS iterations. Default 100.
            tolerance: Convergence threshold on the 6D error norm. Default 1e-6.
            damping: Damping factor λ (higher = more stable, slower convergence).
                     Default 0.1.

        Returns:
            Joint positions in degrees that achieve (or approximate) the desired EE pose.
        """
        self._set_joints(self._q, current_joint_pos)

        for _ in range(max_iter):
            self._pin.forwardKinematics(self.model, self.data, self._q)
            self._pin.updateFramePlacements(self.model, self.data)

            current_pose = self.data.oMf[self.target_frame_id]

            # Position error in world frame
            pos_err = desired_ee_pose[:3, 3] - current_pose.translation

            # Orientation error in world frame
            R_des = desired_ee_pose[:3, :3]
            R_cur = current_pose.rotation
            R_err = R_des @ R_cur.T  # rotation from current → desired in world frame
            ori_err = self._pin.log3(R_err)

            # Weighted error vector
            err = np.empty(6)
            err[:3] = pos_err * position_weight
            err[3:] = ori_err * orientation_weight

            if np.linalg.norm(err) < tolerance:
                break

            # Jacobian of the target frame, world-aligned
            J_full = self._pin.computeFrameJacobian(
                self.model,
                self.data,
                self._q,
                self.target_frame_id,
                self._pin.LOCAL_WORLD_ALIGNED,
            )

            # Extract columns for our selected joints only
            J = J_full[:, self._joint_v_indices]

            # DLS: Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
            n_joints = len(self._joint_v_indices)
            JJt_damped = J @ J.T + (damping * damping) * np.eye(6)
            try:
                delta_q = J.T @ np.linalg.solve(JJt_damped, err)
            except np.linalg.LinAlgError:
                # Fallback to svd-based pseudo-inverse if solve fails
                delta_q = np.linalg.lstsq(JJt_damped, err, rcond=None)[0]
                delta_q = J.T @ delta_q

            # Apply delta to our joints
            for i, v_idx in enumerate(self._joint_v_indices):
                self._q[v_idx] += delta_q[i]

        joint_pos_deg = self._get_joints(self._q)

        # Preserve non-arm joints (e.g. gripper) from input if present
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return joint_pos_deg
