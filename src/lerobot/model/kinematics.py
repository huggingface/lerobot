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

    def __init__(self, urdf_path: str, ee_frame_name: str = "gripper_tip", joint_names: list[str] = None):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path: Path to the robot URDF file
            ee_frame_name: Name of the end-effector frame in the URDF
            joint_names: List of joint names to control (if None, will use default naming)
        """
        try:
            import placo
        except ImportError as e:
            raise ImportError(
                "placo is required for RobotKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base

        self.ee_frame_name = ee_frame_name

        # Set joint names
        if joint_names is None:
            # Default joint names for SO-ARM100
            self.joint_names = ["1", "2", "3", "4", "5"]
        else:
            self.joint_names = joint_names

        # Initialize frame task for IK
        self.tip_starting_pose = np.eye(4)
        self.tip_frame = self.solver.add_frame_task(self.ee_frame_name, self.tip_starting_pose)
        self.tip_frame.configure(self.ee_frame_name, "soft", 1.0, 1.0)

    def forward_kinematics(self, robot_pos_deg, frame=None):
        """
        Compute forward kinematics for given joint configuration.

        Args:
            robot_pos_deg: Joint positions in degrees (numpy array)
            frame: Target frame name (if None, uses ee_frame_name)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """
        if frame is None:
            frame = self.ee_frame_name

        # Convert degrees to radians
        robot_pos_rad = np.deg2rad(robot_pos_deg[: len(self.joint_names)])

        # Update joint positions in placo robot
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, robot_pos_rad[i])

        # Update kinematics
        self.robot.update_kinematics()

        # Get the transformation matrix
        return self.robot.get_T_world_frame(frame)

    def ik(self, current_joint_pos, desired_ee_pose, position_only=True, frame=None):
        """
        Compute inverse kinematics using placo solver.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_only: If True, only match position (not orientation)
            frame: Target frame name (if None, uses ee_frame_name)

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """
        if frame is None:
            frame = self.ee_frame_name

        # Convert current joint positions to radians for initial guess
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])

        # Set current joint positions as initial guess
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_rad[i])

        # Update the target pose for the frame task
        self.tip_frame.T_world_frame = desired_ee_pose

        # Configure the task based on position_only flag
        if position_only:
            # Only constrain position, not orientation
            self.tip_frame.configure(self.ee_frame_name, "soft", 1.0, 0.0)
        else:
            # Constrain both position and orientation
            self.tip_frame.configure(self.ee_frame_name, "soft", 1.0, 1.0)

        # Solve IK
        self.solver.solve(True)
        self.robot.update_kinematics()

        # Extract joint positions
        joint_positions_rad = []
        for joint_name in self.joint_names:
            joint = self.robot.get_joint(joint_name)
            joint_positions_rad.append(joint)

        # Convert back to degrees
        joint_positions_deg = np.rad2deg(joint_positions_rad)

        # Preserve gripper position if present in current_joint_pos
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_positions_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return joint_positions_deg
