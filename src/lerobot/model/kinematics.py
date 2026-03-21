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

import numpy as np


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
        try:
            import placo  # type: ignore[import-not-found] # C++ library with Python bindings, no type stubs available. TODO: Create stub file or request upstream typing support.
        except ImportError as e:
            raise ImportError(
                "placo is required for RobotKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base

        self.target_frame_name = target_frame_name

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    def get_joint_v_offsets(self, joint_names: list[str] | None = None) -> list[int]:
        """Get the velocity-vector column indices for the given joints.

        Placo's frame_jacobian returns a 6×nv matrix where nv includes 6 floating-base
        DOFs followed by the actuated joints. Each joint's column is at
        ``robot.get_joint_v_offset(name)``.

        Args:
            joint_names: Joint names to look up. Defaults to ``self.joint_names``.

        Returns:
            List of column indices into the full Jacobian.
        """
        names = joint_names if joint_names is not None else self.joint_names
        return [self.robot.get_joint_v_offset(n) for n in names]

    def compute_frame_jacobian(
        self,
        joint_pos_deg: np.ndarray,
        frame_name: str | None = None,
        joint_names: list[str] | None = None,
    ) -> np.ndarray:
        """
        Compute the 6×n_joints frame Jacobian at the given frame, restricted to the
        requested joints only.

        Placo returns a 6×nv Jacobian over the full velocity vector (floating base +
        all joints). This method extracts only the columns corresponding to the
        requested joints using ``get_joint_v_offset``.

        The reference frame is ``local_world_aligned`` (origin at the frame, axes
        aligned with the world frame). Top 3 rows = linear velocity, bottom 3 =
        angular velocity.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array).
            frame_name: Frame to compute Jacobian at. Defaults to target_frame_name.
            joint_names: Joints whose columns to extract. Defaults to self.joint_names.

        Returns:
            6×n_joints Jacobian matrix (numpy array).
        """
        self.forward_kinematics(joint_pos_deg)
        frame = frame_name if frame_name is not None else self.target_frame_name
        j_full = self.robot.frame_jacobian(frame, "local_world_aligned")
        col_indices = self.get_joint_v_offsets(joint_names)
        return j_full[:, col_indices]

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
