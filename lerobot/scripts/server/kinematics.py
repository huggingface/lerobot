# ruff: noqa: N806, N815, N803

import numpy as np
from scipy.spatial.transform import Rotation


def skew_symmetric(w):
    """Creates the skew-symmetric matrix from a 3D vector."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def rodrigues_rotation(w, theta):
    """Computes the rotation matrix using Rodrigues' formula."""
    w_hat = skew_symmetric(w)
    return np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat


def screw_axis_to_transform(S, theta):
    """Converts a screw axis to a 4x4 transformation matrix."""
    S_w = S[:3]
    S_v = S[3:]
    if np.allclose(S_w, 0) and np.linalg.norm(S_v) == 1:  # Pure translation
        T = np.eye(4)
        T[:3, 3] = S_v * theta
    elif np.linalg.norm(S_w) == 1:  # Rotation and translation
        w_hat = skew_symmetric(S_w)
        R = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat
        t = (np.eye(3) * theta + (1 - np.cos(theta)) * w_hat + (theta - np.sin(theta)) * w_hat @ w_hat) @ S_v
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
    else:
        raise ValueError("Invalid screw axis parameters")
    return T


def pose_difference_se3(pose1, pose2):
    """
    Calculates the SE(3) difference between two 4x4 homogeneous transformation matrices.

    pose1 - pose2

    Args:
        pose1: A 4x4 numpy array representing the first pose.
        pose2: A 4x4 numpy array representing the second pose.

    Returns:
        A tuple (translation_diff, rotation_diff) where:
        - translation_diff is a 3x1 numpy array representing the translational difference.
        - rotation_diff is a 3x1 numpy array representing the rotational difference in axis-angle representation.
    """

    # Extract rotation matrices from poses
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]

    # Calculate translational difference
    translation_diff = pose1[:3, 3] - pose2[:3, 3]

    # Calculate rotational difference using scipy's Rotation library
    R_diff = Rotation.from_matrix(R1 @ R2.T)
    rotation_diff = R_diff.as_rotvec()  # Convert to axis-angle representation

    return np.concatenate([translation_diff, rotation_diff])


def se3_error(target_pose, current_pose):
    pos_error = target_pose[:3, 3] - current_pose[:3, 3]
    R_target = target_pose[:3, :3]
    R_current = current_pose[:3, :3]
    R_error = R_target @ R_current.T
    rot_error = Rotation.from_matrix(R_error).as_rotvec()
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
        "so100": {
            "gripper": [0.320, 0, 0.050],
            "wrist": [0.278, 0, 0.050],
            "forearm": [0.143, 0, 0.044],
            "humerus": [0.031, 0, 0.072],
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
    }

    def __init__(self, robot_type="so100"):
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

    def _create_translation_matrix(self, x=0, y=0, z=0):
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
            ]
        )

        # Wrist orientation
        self.wrist_X0 = np.array(
            [
                [0, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Base orientation
        self.base_X0 = np.array(
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
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
            ]
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
        self.S_BR = np.array([0, 1, 0, -self.measurements["wrist"][2], 0, self.measurements["wrist"][0]])

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
            ]
        )

        # Forearm origin + centroid transform
        self.X_FoFc = self._create_translation_matrix(x=0.036)  # spellchecker:disable-line

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
            ]
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
        self.S_BS = np.array([0, 0, -1, 0, 0, 0])

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

    def fk_base(self):
        """Forward kinematics for the base frame."""
        return self.X_WoBo @ self.X_BoBc @ self.base_X0

    def fk_shoulder(self, robot_pos_deg):
        """Forward kinematics for the shoulder frame."""
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return self.X_WoBo @ screw_axis_to_transform(self.S_BS, robot_pos_rad[0]) @ self.X_SoSc @ self.X_BS

    def fk_humerus(self, robot_pos_deg):
        """Forward kinematics for the humerus frame."""
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            self.X_WoBo
            @ screw_axis_to_transform(self.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(self.S_BH, robot_pos_rad[1])
            @ self.X_HoHc
            @ self.X_BH
        )

    def fk_forearm(self, robot_pos_deg):
        """Forward kinematics for the forearm frame."""
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            self.X_WoBo
            @ screw_axis_to_transform(self.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(self.S_BH, robot_pos_rad[1])
            @ screw_axis_to_transform(self.S_BF, robot_pos_rad[2])
            @ self.X_FoFc  # spellchecker:disable-line
            @ self.X_BF
        )

    def fk_wrist(self, robot_pos_deg):
        """Forward kinematics for the wrist frame."""
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            self.X_WoBo
            @ screw_axis_to_transform(self.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(self.S_BH, robot_pos_rad[1])
            @ screw_axis_to_transform(self.S_BF, robot_pos_rad[2])
            @ screw_axis_to_transform(self.S_BR, robot_pos_rad[3])
            @ self.X_RoRc
            @ self.X_BR
            @ self.wrist_X0
        )

    def fk_gripper(self, robot_pos_deg):
        """Forward kinematics for the gripper frame."""
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            self.X_WoBo
            @ screw_axis_to_transform(self.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(self.S_BH, robot_pos_rad[1])
            @ screw_axis_to_transform(self.S_BF, robot_pos_rad[2])
            @ screw_axis_to_transform(self.S_BR, robot_pos_rad[3])
            @ screw_axis_to_transform(self.S_BG, robot_pos_rad[4])
            @ self._fk_gripper_post
        )

    def fk_gripper_tip(self, robot_pos_deg):
        """Forward kinematics for the gripper tip frame."""
        robot_pos_rad = robot_pos_deg / 180 * np.pi
        return (
            self.X_WoBo
            @ screw_axis_to_transform(self.S_BS, robot_pos_rad[0])
            @ screw_axis_to_transform(self.S_BH, robot_pos_rad[1])
            @ screw_axis_to_transform(self.S_BF, robot_pos_rad[2])
            @ screw_axis_to_transform(self.S_BR, robot_pos_rad[3])
            @ screw_axis_to_transform(self.S_BG, robot_pos_rad[4])
            @ self.X_GoGt
            @ self.X_BoGo
            @ self.gripper_X0
        )

    def compute_jacobian(self, robot_pos_deg, fk_func=None):
        """Finite differences to compute the Jacobian.
        J(i, j) represents how the ith component of the end-effector's velocity changes wrt a small change
        in the jth joint's velocity.

        Args:
            robot_pos_deg: Current joint positions in degrees
            fk_func: Forward kinematics function to use (defaults to fk_gripper)
        """
        if fk_func is None:
            fk_func = self.fk_gripper

        eps = 1e-8
        jac = np.zeros(shape=(6, 5))
        delta = np.zeros(len(robot_pos_deg[:-1]), dtype=np.float64)
        for el_ix in range(len(robot_pos_deg[:-1])):
            delta *= 0
            delta[el_ix] = eps / 2
            Sdot = (
                pose_difference_se3(
                    fk_func(robot_pos_deg[:-1] + delta),
                    fk_func(robot_pos_deg[:-1] - delta),
                )
                / eps
            )
            jac[:, el_ix] = Sdot
        return jac

    def compute_positional_jacobian(self, robot_pos_deg, fk_func=None):
        """Finite differences to compute the positional Jacobian.
        J(i, j) represents how the ith component of the end-effector's position changes wrt a small change
        in the jth joint's velocity.

        Args:
            robot_pos_deg: Current joint positions in degrees
            fk_func: Forward kinematics function to use (defaults to fk_gripper)
        """
        if fk_func is None:
            fk_func = self.fk_gripper

        eps = 1e-8
        jac = np.zeros(shape=(3, 5))
        delta = np.zeros(len(robot_pos_deg[:-1]), dtype=np.float64)
        for el_ix in range(len(robot_pos_deg[:-1])):
            delta *= 0
            delta[el_ix] = eps / 2
            Sdot = (
                fk_func(robot_pos_deg[:-1] + delta)[:3, 3] - fk_func(robot_pos_deg[:-1] - delta)[:3, 3]
            ) / eps
            jac[:, el_ix] = Sdot
        return jac

    def ik(self, current_joint_state, desired_ee_pose, position_only=True, fk_func=None):
        """Inverse kinematics using gradient descent.

        Args:
            current_joint_state: Initial joint positions in degrees
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_only: If True, only match end-effector position, not orientation
            fk_func: Forward kinematics function to use (defaults to fk_gripper)

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """
        if fk_func is None:
            fk_func = self.fk_gripper

        # Do gradient descent.
        max_iterations = 5
        learning_rate = 1
        for _ in range(max_iterations):
            current_ee_pose = fk_func(current_joint_state)
            if not position_only:
                error = se3_error(desired_ee_pose, current_ee_pose)
                jac = self.compute_jacobian(current_joint_state, fk_func)
            else:
                error = desired_ee_pose[:3, 3] - current_ee_pose[:3, 3]
                jac = self.compute_positional_jacobian(current_joint_state, fk_func)
            delta_angles = np.linalg.pinv(jac) @ error
            current_joint_state[:-1] += learning_rate * delta_angles

            if np.linalg.norm(error) < 5e-3:
                return current_joint_state
        return current_joint_state


if __name__ == "__main__":
    import time

    def run_test(robot_type):
        """Run test suite for a specific robot type."""
        print(f"\n--- Testing {robot_type.upper()} Robot ---")

        # Initialize kinematics for this robot
        robot = RobotKinematics(robot_type)

        # Test 1: Forward kinematics consistency
        print("Test 1: Forward kinematics consistency")
        test_angles = np.array([30, 45, -30, 20, 10, 0])  # Example joint angles in degrees

        # Calculate FK for different joints
        shoulder_pose = robot.fk_shoulder(test_angles)
        humerus_pose = robot.fk_humerus(test_angles)
        forearm_pose = robot.fk_forearm(test_angles)
        wrist_pose = robot.fk_wrist(test_angles)
        gripper_pose = robot.fk_gripper(test_angles)
        gripper_tip_pose = robot.fk_gripper_tip(test_angles)

        # Check that poses form a consistent kinematic chain (positions should be progressively further from origin)
        distances = [
            np.linalg.norm(shoulder_pose[:3, 3]),
            np.linalg.norm(humerus_pose[:3, 3]),
            np.linalg.norm(forearm_pose[:3, 3]),
            np.linalg.norm(wrist_pose[:3, 3]),
            np.linalg.norm(gripper_pose[:3, 3]),
            np.linalg.norm(gripper_tip_pose[:3, 3]),
        ]

        # Check if distances generally increase along the chain
        is_consistent = all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))
        print(f"  Pose distances from origin: {[round(d, 3) for d in distances]}")
        print(f"  Kinematic chain consistency: {'PASSED' if is_consistent else 'FAILED'}")

        # Test 2: Jacobian computation
        print("Test 2: Jacobian computation")
        jacobian = robot.compute_jacobian(test_angles)
        positional_jacobian = robot.compute_positional_jacobian(test_angles)

        # Check shapes
        jacobian_shape_ok = jacobian.shape == (6, 5)
        pos_jacobian_shape_ok = positional_jacobian.shape == (3, 5)

        print(f"  Jacobian shape: {'PASSED' if jacobian_shape_ok else 'FAILED'}")
        print(f"  Positional Jacobian shape: {'PASSED' if pos_jacobian_shape_ok else 'FAILED'}")

        # Test 3: Inverse kinematics
        print("Test 3: Inverse kinematics (position only)")

        # Generate target pose from known joint angles
        original_angles = np.array([10, 20, 30, -10, 5, 0])
        target_pose = robot.fk_gripper(original_angles)

        # Start IK from a different position
        initial_guess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Measure IK performance
        start_time = time.time()
        computed_angles = robot.ik(initial_guess.copy(), target_pose)
        ik_time = time.time() - start_time

        # Compute resulting pose from IK solution
        result_pose = robot.fk_gripper(computed_angles)

        # Calculate position error
        pos_error = np.linalg.norm(target_pose[:3, 3] - result_pose[:3, 3])
        passed = pos_error < 0.01  # Accept errors less than 1cm

        print(f"  IK computation time: {ik_time:.4f} seconds")
        print(f"  Position error: {pos_error:.4f}")
        print(f"  IK position accuracy: {'PASSED' if passed else 'FAILED'}")

        return is_consistent and jacobian_shape_ok and pos_jacobian_shape_ok and passed

    # Run tests for all robot types
    results = {}
    for robot_type in ["koch", "so100", "moss"]:
        results[robot_type] = run_test(robot_type)

    # Print overall summary
    print("\n=== Test Summary ===")
    all_passed = all(results.values())
    for robot_type, passed in results.items():
        print(f"{robot_type.upper()}: {'PASSED' if passed else 'FAILED'}")
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
