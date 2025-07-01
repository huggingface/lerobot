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
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def skew_symmetric(w: NDArray[np.float32]) -> NDArray[np.float32]:
    """Creates the skew-symmetric matrix from a 3D vector."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def rodrigues_rotation(w: NDArray[np.float32], theta: float) -> NDArray[np.float32]:
    """Computes the rotation matrix using Rodrigues' formula."""
    w_hat = skew_symmetric(w)
    return np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat


def screw_axis_to_transform(s: NDArray[np.float32], theta: float) -> NDArray[np.float32]:
    """Converts a screw axis to a 4x4 transformation matrix."""
    screw_axis_rot = s[:3]
    screw_axis_trans = s[3:]

    # Pure translation
    if np.allclose(screw_axis_rot, 0) and np.linalg.norm(screw_axis_trans) == 1:
        transform = np.eye(4)
        transform[:3, 3] = screw_axis_trans * theta

    # Rotation (and potentially translation)
    elif np.linalg.norm(screw_axis_rot) == 1:
        w_hat = skew_symmetric(screw_axis_rot)
        rot_mat = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat
        t = (
            np.eye(3) * theta + (1 - np.cos(theta)) * w_hat + (theta - np.sin(theta)) * w_hat @ w_hat
        ) @ screw_axis_trans
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = t
    else:
        raise ValueError("Invalid screw axis parameters")
    return transform


def pose_difference_se3(pose1: NDArray[np.float32], pose2: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Calculates the SE(3) difference between two 4x4 homogeneous transformation matrices.
    SE(3) (Special Euclidean Group) represents rigid body transformations in 3D space,
    combining rotation (SO(3)) and translation.

    Each 4x4 matrix has the following structure:
    [R11 R12 R13 tx]
    [R21 R22 R23 ty]
    [R31 R32 R33 tz]
    [ 0   0   0   1]

    where R is the 3x3 rotation matrix and [tx,ty,tz] is the translation vector.

    Args:
        pose1: A 4x4 numpy array representing the first pose.
        pose2: A 4x4 numpy array representing the second pose.

    Returns:
        A 6D numpy array concatenating translation and rotation differences.
        First 3 elements are the translational difference (position).
        Last 3 elements are the rotational difference in axis-angle representation.
    """
    rot1 = pose1[:3, :3]
    rot2 = pose2[:3, :3]

    translation_diff = pose1[:3, 3] - pose2[:3, 3]

    # Calculate rotational difference using scipy's Rotation library
    rot_diff = Rotation.from_matrix(rot1 @ rot2.T)
    rotation_diff = rot_diff.as_rotvec()  # Axis-angle representation

    return np.concatenate([translation_diff, rotation_diff])


def se3_error(target_pose: NDArray[np.float32], current_pose: NDArray[np.float32]) -> NDArray[np.float32]:
    pos_error = target_pose[:3, 3] - current_pose[:3, 3]

    rot_target = target_pose[:3, :3]
    rot_current = current_pose[:3, :3]
    rot_error_mat = rot_target @ rot_current.T
    rot_error = Rotation.from_matrix(rot_error_mat).as_rotvec()

    return np.concatenate([pos_error, rot_error])


class RobotKinematics:
    """Robot kinematics class supporting multiple robot models."""

    # Robot measurements dictionary
    ROBOT_MEASUREMENTS = {
        "koch": {
            "gripper": [0.239, -0.001, 0.024],
            "wrist": [0.209, 0, 0.024],
            "forearm": [0.108, 0, 0.02],
            "humerus": [0, 0, 0.036],
            "shoulder": [0, 0, 0],
            "base": [0, 0, 0.02],
        },
        "moss": {
            "gripper": [0.246, 0.013, 0.111],
            "wrist": [0.245, 0.002, 0.064],
            "forearm": [0.122, 0, 0.064],
            "humerus": [0.001, 0.001, 0.063],
            "shoulder": [0, 0, 0],
            "base": [0, 0, 0.02],
        },
        "so_old_calibration": {
            "gripper": [0.320, 0, 0.050],
            "wrist": [0.278, 0, 0.050],
            "forearm": [0.143, 0, 0.044],
            "humerus": [0.031, 0, 0.072],
            "shoulder": [0, 0, 0],
            "base": [0, 0, 0.02],
        },
        "so_new_calibration": {
            "gripper": [0.33, 0.0, 0.285],
            "wrist": [0.30, 0.0, 0.267],
            "forearm": [0.25, 0.0, 0.266],
            "humerus": [0.06, 0.0, 0.264],
            "shoulder": [0.0, 0.0, 0.238],
            "base": [0.0, 0.0, 0.12],
        },
    }

    def __init__(self, robot_type: str = "so100"):
        """Initialize kinematics for the specified robot type.

        Args:
            robot_type: String specifying the robot model ("koch", "so100", or "moss")
        """
        if robot_type not in self.ROBOT_MEASUREMENTS:
            raise ValueError(
                f"Unknown robot type: {robot_type}. Available types: {list(self.ROBOT_MEASUREMENTS.keys())}"
            )

        self.robot_type = robot_type
        self.measurements = self.ROBOT_MEASUREMENTS[robot_type]

        # Initialize all transformation matrices and screw axes
        self._setup_transforms()

    def _create_translation_matrix(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> NDArray[np.float32]:
        """Create a 4x4 translation matrix."""
        return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

    def _setup_transforms(self):
        """Setup all transformation matrices and screw axes for the robot."""
        # Set up rotation matrices (constant across robot types)

        # Gripper orientation
        self.gripper_X0 = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Wrist orientation
        self.wrist_X0 = np.array(
            [
                [0, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Base orientation
        self.base_X0 = np.array(
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Gripper
        # Screw axis of gripper frame wrt base frame
        self.S_BG = np.array(
            [
                1,
                0,
                0,
                0,
                self.measurements["gripper"][2],
                -self.measurements["gripper"][1],
            ],
            dtype=np.float32,
        )

        # Gripper origin to centroid transform
        self.X_GoGc = self._create_translation_matrix(x=0.07)

        # Gripper origin to tip transform
        self.X_GoGt = self._create_translation_matrix(x=0.12)

        # 0-position gripper frame pose wrt base
        self.X_BoGo = self._create_translation_matrix(
            x=self.measurements["gripper"][0],
            y=self.measurements["gripper"][1],
            z=self.measurements["gripper"][2],
        )

        # Wrist
        # Screw axis of wrist frame wrt base frame
        self.S_BR = np.array(
            [0, 1, 0, -self.measurements["wrist"][2], 0, self.measurements["wrist"][0]], dtype=np.float32
        )

        # 0-position origin to centroid transform
        self.X_RoRc = self._create_translation_matrix(x=0.0035, y=-0.002)

        # 0-position wrist frame pose wrt base
        self.X_BR = self._create_translation_matrix(
            x=self.measurements["wrist"][0],
            y=self.measurements["wrist"][1],
            z=self.measurements["wrist"][2],
        )

        # Forearm
        # Screw axis of forearm frame wrt base frame
        self.S_BF = np.array(
            [
                0,
                1,
                0,
                -self.measurements["forearm"][2],
                0,
                self.measurements["forearm"][0],
            ],
            dtype=np.float32,
        )

        # Forearm origin + centroid transform
        self.X_ForearmFc = self._create_translation_matrix(x=0.036)

        # 0-position forearm frame pose wrt base
        self.X_BF = self._create_translation_matrix(
            x=self.measurements["forearm"][0],
            y=self.measurements["forearm"][1],
            z=self.measurements["forearm"][2],
        )

        # Humerus
        # Screw axis of humerus frame wrt base frame
        self.S_BH = np.array(
            [
                0,
                -1,
                0,
                self.measurements["humerus"][2],
                0,
                -self.measurements["humerus"][0],
            ],
            dtype=np.float32,
        )

        # Humerus origin to centroid transform
        self.X_HoHc = self._create_translation_matrix(x=0.0475)

        # 0-position humerus frame pose wrt base
        self.X_BH = self._create_translation_matrix(
            x=self.measurements["humerus"][0],
            y=self.measurements["humerus"][1],
            z=self.measurements["humerus"][2],
        )

        # Shoulder
        # Screw axis of shoulder frame wrt Base frame
        self.S_BS = np.array([0, 0, -1, 0, 0, 0], dtype=np.float32)

        # Shoulder origin to centroid transform
        self.X_SoSc = self._create_translation_matrix(x=-0.017, z=0.0235)

        # 0-position shoulder frame pose wrt base
        self.X_BS = self._create_translation_matrix(
            x=self.measurements["shoulder"][0],
            y=self.measurements["shoulder"][1],
            z=self.measurements["shoulder"][2],
        )

        # Base
        # Base origin to centroid transform
        self.X_BoBc = self._create_translation_matrix(y=0.015)

        # World to base transform
        self.X_WoBo = self._create_translation_matrix(
            x=self.measurements["base"][0],
            y=self.measurements["base"][1],
            z=self.measurements["base"][2],
        )

        # Pre-compute gripper post-multiplication matrix
        self._fk_gripper_post = self.X_GoGc @ self.X_BoGo @ self.gripper_X0

    def forward_kinematics(
        self,
        robot_pos_deg: NDArray[np.float32],
        frame: str = "gripper_tip",
    ) -> NDArray[np.float32]:
        """Generic forward kinematics.

        Args:
            robot_pos_deg: Joint positions in degrees. Can be ``None`` when
                computing the *base* frame as it does not depend on joint
                angles.
            frame: Target frame. One of
                ``{"base", "shoulder", "humerus", "forearm", "wrist", "gripper", "gripper_tip"}``.

        Returns
        -------
        NDArray[np.float32]
            4×4 homogeneous transformation matrix of the requested frame
            expressed in the world coordinate system.
        """
        frame = frame.lower()
        if frame not in {
            "base",
            "shoulder",
            "humerus",
            "forearm",
            "wrist",
            "gripper",
            "gripper_tip",
        }:
            raise ValueError(
                f"Unknown frame '{frame}'. Valid options are base, shoulder, humerus, forearm, wrist, gripper, gripper_tip."
            )

        # Base frame does not rely on joint angles.
        if frame == "base":
            return self.X_WoBo @ self.X_BoBc @ self.base_X0

        robot_pos_rad = robot_pos_deg / 180 * np.pi

        # Extract joint angles (note the sign convention for shoulder lift).
        theta_shoulder_pan = robot_pos_rad[0]
        theta_shoulder_lift = -robot_pos_rad[1]
        theta_elbow_flex = robot_pos_rad[2]
        theta_wrist_flex = robot_pos_rad[3]
        theta_wrist_roll = robot_pos_rad[4]

        # Start with the world-to-base transform; incrementally add successive links.
        transformation_matrix = self.X_WoBo @ screw_axis_to_transform(self.S_BS, theta_shoulder_pan)
        if frame == "shoulder":
            return transformation_matrix @ self.X_SoSc @ self.X_BS

        transformation_matrix = transformation_matrix @ screw_axis_to_transform(
            self.S_BH, theta_shoulder_lift
        )
        if frame == "humerus":
            return transformation_matrix @ self.X_HoHc @ self.X_BH

        transformation_matrix = transformation_matrix @ screw_axis_to_transform(self.S_BF, theta_elbow_flex)
        if frame == "forearm":
            return transformation_matrix @ self.X_ForearmFc @ self.X_BF

        transformation_matrix = transformation_matrix @ screw_axis_to_transform(self.S_BR, theta_wrist_flex)
        if frame == "wrist":
            return transformation_matrix @ self.X_RoRc @ self.X_BR @ self.wrist_X0

        transformation_matrix = transformation_matrix @ screw_axis_to_transform(self.S_BG, theta_wrist_roll)
        if frame == "gripper":
            return transformation_matrix @ self._fk_gripper_post
        else:  # frame == "gripper_tip"
            return transformation_matrix @ self.X_GoGt @ self.X_BoGo @ self.gripper_X0

    def compute_jacobian(
        self, robot_pos_deg: NDArray[np.float32], frame: str = "gripper_tip"
    ) -> NDArray[np.float32]:
        """Finite differences to compute the Jacobian.
        J(i, j) represents how the ith component of the end-effector's velocity changes wrt a small change
        in the jth joint's velocity.

        Args:
            robot_pos_deg: Current joint positions in degrees
            fk_func: Forward kinematics function to use (defaults to fk_gripper)
        """

        eps = 1e-8
        jac = np.zeros(shape=(6, 5))
        delta = np.zeros(len(robot_pos_deg[:-1]), dtype=np.float64)
        for el_ix in range(len(robot_pos_deg[:-1])):
            delta *= 0
            delta[el_ix] = eps / 2
            sdot = (
                pose_difference_se3(
                    self.forward_kinematics(robot_pos_deg[:-1] + delta, frame),
                    self.forward_kinematics(robot_pos_deg[:-1] - delta, frame),
                )
                / eps
            )
            jac[:, el_ix] = sdot
        return jac

    def compute_positional_jacobian(
        self, robot_pos_deg: NDArray[np.float32], frame: str = "gripper_tip"
    ) -> NDArray[np.float32]:
        """Finite differences to compute the positional Jacobian.
        J(i, j) represents how the ith component of the end-effector's position changes wrt a small change
        in the jth joint's velocity.

        Args:
            robot_pos_deg: Current joint positions in degrees
            fk_func: Forward kinematics function to use (defaults to fk_gripper)
        """
        eps = 1e-8
        jac = np.zeros(shape=(3, 5))
        delta = np.zeros(len(robot_pos_deg[:-1]), dtype=np.float64)
        for el_ix in range(len(robot_pos_deg[:-1])):
            delta *= 0
            delta[el_ix] = eps / 2
            sdot = (
                self.forward_kinematics(robot_pos_deg[:-1] + delta, frame)[:3, 3]
                - self.forward_kinematics(robot_pos_deg[:-1] - delta, frame)[:3, 3]
            ) / eps
            jac[:, el_ix] = sdot
        return jac

    def ik(
        self,
        current_joint_pos: NDArray[np.float32],
        desired_ee_pose: NDArray[np.float32],
        position_only: bool = True,
        frame: str = "gripper_tip",
        max_iterations: int = 5,
        learning_rate: float = 1,
    ) -> NDArray[np.float32]:
        """Inverse kinematics using gradient descent.

        Args:
            current_joint_state: Initial joint positions in degrees
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_only: If True, only match end-effector position, not orientation
            frame: Target frame. One of
                ``{"base", "shoulder", "humerus", "forearm", "wrist", "gripper", "gripper_tip"}``.
            max_iterations: Maximum number of iterations to run
            learning_rate: Learning rate for gradient descent

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """
        # Do gradient descent.
        current_joint_state = current_joint_pos.copy()
        for _ in range(max_iterations):
            current_ee_pose = self.forward_kinematics(current_joint_state, frame)
            if not position_only:
                error = se3_error(desired_ee_pose, current_ee_pose)
                jac = self.compute_jacobian(current_joint_state, frame)
            else:
                error = desired_ee_pose[:3, 3] - current_ee_pose[:3, 3]
                jac = self.compute_positional_jacobian(current_joint_state, frame)
            delta_angles = np.linalg.pinv(jac) @ error
            current_joint_state[:-1] += learning_rate * delta_angles

            if np.linalg.norm(error) < 5e-3:
                return current_joint_state
        return current_joint_state
