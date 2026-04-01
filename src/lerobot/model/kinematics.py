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

import os

import numpy as np


def _resolve_urdf_path(urdf_path: str) -> str:
    """Resolve to a path placo can load (directory containing robot.urdf).

    placo.RobotWrapper always looks for robot.urdf inside the given path (it
    appends /robot.urdf), so we must pass a directory that contains a file
    named robot.urdf. If the user passes a path to a .urdf file (e.g. so101.urdf),
    we pass its parent directory and ensure robot.urdf exists there (symlink to
    the given file if needed).
    """
    path = os.path.abspath(os.path.expanduser(urdf_path))
    if os.path.isfile(path):
        parent = os.path.dirname(path)
        robot_urdf = os.path.join(parent, "robot.urdf")
        if not os.path.isfile(robot_urdf):
            try:
                os.symlink(os.path.basename(path), robot_urdf)
            except OSError:
                pass  # e.g. permission or already exists as broken link
        if os.path.isfile(robot_urdf):
            return parent
        raise ValueError(
            f"Could not use {path}: placo requires a directory containing robot.urdf. "
            f"Rename to {os.path.join(parent, 'robot.urdf')} or pass the directory: --urdf={parent}"
        )
    if os.path.isdir(path):
        robot_urdf = os.path.join(path, "robot.urdf")
        if os.path.isfile(robot_urdf):
            return path
        for name in ("so101.urdf", "so101_new_calib.urdf", "model.urdf"):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                try:
                    os.symlink(name, robot_urdf)
                except OSError:
                    pass
                if os.path.isfile(robot_urdf):
                    return path
        raise ValueError(
            f"URDF path is a directory but no robot.urdf / so101.urdf / model.urdf found: {path}. "
            "Add a .urdf file or pass a path to an existing .urdf file, e.g. --urdf=./SO101/so101.urdf"
        )
    # Path is missing (e.g. user passed ./SO101/so101.urdf but only so101_new_calib.urdf exists).
    # Try parent directory with known URDF names.
    parent = os.path.dirname(path)
    if os.path.isdir(parent):
        robot_urdf = os.path.join(parent, "robot.urdf")
        for name in ("so101.urdf", "so101_new_calib.urdf", "model.urdf"):
            candidate = os.path.join(parent, name)
            if os.path.isfile(candidate):
                try:
                    os.symlink(name, robot_urdf)
                except OSError:
                    pass
                if os.path.isfile(robot_urdf):
                    return parent
    hint = ""
    if "SO101" in path or "so101" in path.lower():
        hint = " For SO101, download so101_new_calib.urdf from https://github.com/TheRobotStudio/SO-ARM100 and put it in SO101/."
    raise ValueError(
        f"URDF path does not exist: {path}. "
        f"Pass a path to an existing .urdf file or a directory containing robot.urdf.{hint}"
    )


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

        resolved = _resolve_urdf_path(urdf_path)
        self.urdf_dir = resolved
        self.robot = placo.RobotWrapper(resolved)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base

        self.target_frame_name = target_frame_name

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

        self._viz_chain: list[str] = [target_frame_name]
        try:
            from lerobot.utils.urdf_chain import kinematic_chain_links, robot_urdf_file_in_dir

            urdf_file = robot_urdf_file_in_dir(resolved)
            chain = kinematic_chain_links(urdf_file, target_frame_name)
            if chain:
                self._viz_chain = chain
        except Exception:
            pass

    def set_joint_positions(self, joint_pos_deg: np.ndarray) -> None:
        """Apply arm joint angles (degrees) and update placo kinematics."""
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])
        self.robot.update_kinematics()

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration given the target frame name in the constructor.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """
        self.set_joint_positions(joint_pos_deg)
        return self.robot.get_T_world_frame(self.target_frame_name)

    def get_link_transforms_chain(self, joint_pos_deg: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """World poses for the URDF chain from base to end-effector (for 3D debug viz)."""
        self.set_joint_positions(joint_pos_deg)
        out: list[tuple[str, np.ndarray]] = []
        for name in self._viz_chain:
            try:
                T = self.robot.get_T_world_frame(name)
                out.append((name, np.asarray(T, dtype=np.float64)))
            except Exception:
                continue
        return out

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
