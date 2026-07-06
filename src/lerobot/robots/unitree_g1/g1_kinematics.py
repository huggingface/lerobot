#!/usr/bin/env python

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

import logging
import os
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class WeightedMovingFilter:
    def __init__(self, weights, data_size=14):
        self._window_size = len(weights)
        self._weights = np.array(weights)
        self._data_size = data_size
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = deque(maxlen=self._window_size)

    def _apply_filter(self):
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]

        data_array = np.array(self._data_queue)
        return data_array.T @ self._weights

    def add_data(self, new_data):
        assert len(new_data) == self._data_size

        if len(self._data_queue) > 0 and np.array_equal(
            new_data, self._data_queue[-1]
        ):  # skip duplicate data
            return

        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    @property
    def filtered_data(self):
        return self._filtered_data


class G1_29_ArmIK:  # noqa: N801
    def __init__(self, unit_test=False):
        import casadi
        import pinocchio as pin
        from huggingface_hub import snapshot_download
        from pinocchio import casadi as cpin

        self._pin = pin
        self.unit_test = unit_test

        self.repo_path = snapshot_download("lerobot/unitree-g1-mujoco")
        urdf_path = os.path.join(self.repo_path, "assets", "g1_body29_hand14.urdf")
        mesh_dir = os.path.join(self.repo_path, "assets")

        self.robot = self._pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)

        self.mixed_jointsToLockIDs = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        # Arm joint names in G1 motor order (G1_29_JointArmIndex)
        self._arm_joint_names_g1 = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        # Pinocchio uses its own joint order in q; build index mapping.
        self._arm_joint_names_pin = sorted(
            self._arm_joint_names_g1,
            key=lambda name: self.reduced_robot.model.idx_qs[self.reduced_robot.model.getJointId(name)],
        )
        logger.info(f"Pinocchio arm joint order: {self._arm_joint_names_pin}")
        self._arm_reorder_g1_to_pin = [
            self._arm_joint_names_g1.index(name) for name in self._arm_joint_names_pin
        ]
        # Inverse mapping to return tau in G1 motor order.
        self._arm_reorder_pin_to_g1 = np.argsort(self._arm_reorder_g1_to_pin)

        self.reduced_robot.model.addFrame(
            self._pin.Frame(
                "L_ee",
                self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                self._pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                self._pin.FrameType.OP_FRAME,
            )
        )

        self.reduced_robot.model.addFrame(
            self._pin.Frame(
                "R_ee",
                self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                self._pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                self._pin.FrameType.OP_FRAME,
            )
        )

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(
            50 * self.translational_cost
            + self.rotation_cost
            + 0.02 * self.regularization_cost
            + 0.1 * self.smooth_cost
        )

        opts = {
            "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-6},
            "print_time": False,  # print or not
            "calc_lam_p": False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)  # for smooth

        converged = True
        try:
            self.opti.solve()
            sol_q = self.opti.value(self.var_q)
        except Exception as e:
            converged = False
            logger.error(f"IK convergence error: {e}")
            sol_q = self.opti.debug.value(self.var_q)

        self.smooth_filter.add_data(sol_q)
        sol_q = self.smooth_filter.filtered_data
        self.init_data = sol_q

        if not converged:
            logger.error(
                f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}"
            )
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)

        sol_tauff = self._pin.rnea(
            self.reduced_robot.model,
            self.reduced_robot.data,
            sol_q,
            np.zeros(self.reduced_robot.model.nv),
            np.zeros(self.reduced_robot.model.nv),
        )

        return sol_q, sol_tauff

    def solve_tau(self, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        try:
            q_g1 = np.array(current_lr_arm_motor_q, dtype=float)
            if q_g1.shape[0] != len(self._arm_joint_names_g1):
                raise ValueError(f"Expected {len(self._arm_joint_names_g1)} arm joints, got {q_g1.shape[0]}")
            q_pin = q_g1[self._arm_reorder_g1_to_pin]
            sol_tauff = self._pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                q_pin,
                np.zeros(self.reduced_robot.model.nv),
                np.zeros(self.reduced_robot.model.nv),
            )
            return sol_tauff[self._arm_reorder_pin_to_g1]

        except Exception as e:
            logger.error(f"ERROR in convergence, plotting debug info.{e}")
            return np.zeros(self.reduced_robot.model.nv)


_LEG_JOINT_NAMES_G1 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

_LEFT_FOOT_FRAME = "left_ankle_roll_link"
_RIGHT_FOOT_FRAME = "right_ankle_roll_link"


def _homogeneous_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rotation
    mat[:3, 3] = translation
    return mat


class G1_29_LegIK:  # noqa: N801
    """12-DOF leg IK (pelvis frame) targeting ankle roll link positions."""

    def __init__(
        self,
        unit_test: bool = False,
        max_iter: int = 50,
        tol: float = 1e-6,
        smoothing_weights: np.ndarray | None = None,
    ) -> None:
        import casadi
        import pinocchio as pin
        from huggingface_hub import snapshot_download
        from pinocchio import casadi as cpin

        self._pin = pin
        self.unit_test = unit_test

        self.repo_path = snapshot_download("lerobot/unitree-g1-mujoco")
        urdf_path = os.path.join(self.repo_path, "assets", "g1_body29_hand14.urdf")
        mesh_dir = os.path.join(self.repo_path, "assets")

        self.robot = self._pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)

        joints_to_lock = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=joints_to_lock,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        self._leg_joint_names_g1 = list(_LEG_JOINT_NAMES_G1)
        self._leg_joint_names_pin = sorted(
            self._leg_joint_names_g1,
            key=lambda name: self.reduced_robot.model.idx_qs[self.reduced_robot.model.getJointId(name)],
        )
        self._leg_reorder_g1_to_pin = [
            self._leg_joint_names_g1.index(name) for name in self._leg_joint_names_pin
        ]
        self._leg_reorder_pin_to_g1 = np.argsort(self._leg_reorder_g1_to_pin)

        self.left_foot_id = self.reduced_robot.model.getFrameId(_LEFT_FOOT_FRAME)
        self.right_foot_id = self.reduced_robot.model.getFrameId(_RIGHT_FOOT_FRAME)

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.translational_error = casadi.Function(
            "leg_translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.left_foot_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.right_foot_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "leg_rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.left_foot_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.right_foot_id].rotation @ self.cTf_r[:3, :3].T),
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(
            50 * self.translational_cost
            + 0.5 * self.rotation_cost
            + 0.02 * self.regularization_cost
            + 0.1 * self.smooth_cost
        )

        opts = {
            "ipopt": {"print_level": 0, "max_iter": max_iter, "tol": tol},
            "print_time": False,
            "calc_lam_p": False,
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        if smoothing_weights is None:
            smoothing_weights = np.array([0.4, 0.3, 0.2, 0.1])
        self.smooth_filter = WeightedMovingFilter(np.asarray(smoothing_weights, dtype=float), 12)
        self._default_foot_rot_l: np.ndarray | None = None
        self._default_foot_rot_r: np.ndarray | None = None

    def _g1_leg_to_pin(self, q_g1: np.ndarray) -> np.ndarray:
        q = np.asarray(q_g1, dtype=np.float64).reshape(12)
        return q[self._leg_reorder_g1_to_pin]

    def _pin_leg_to_g1(self, q_pin: np.ndarray) -> np.ndarray:
        q = np.asarray(q_pin, dtype=np.float64).reshape(len(self._leg_joint_names_pin))
        return q[self._leg_reorder_pin_to_g1]

    def foot_poses(self, q_leg_g1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return 4x4 foot poses in the pelvis frame."""
        q_pin = self._g1_leg_to_pin(q_leg_g1)
        self._pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q_pin)
        self._pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
        left = self.reduced_robot.data.oMf[self.left_foot_id].homogeneous
        right = self.reduced_robot.data.oMf[self.right_foot_id].homogeneous
        return left.copy(), right.copy()

    def foot_positions(self, q_leg_g1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left, right = self.foot_poses(q_leg_g1)
        return left[:3, 3].copy(), right[:3, 3].copy()

    def default_foot_state(
        self, q_leg_g1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Positions (3,) and rotations (3,3) for both feet at the given leg configuration."""
        left, right = self.foot_poses(q_leg_g1)
        return left[:3, 3], left[:3, :3], right[:3, 3], right[:3, :3]

    def targets_from_xyz(
        self,
        left_xyz: np.ndarray,
        right_xyz: np.ndarray,
        left_rot: np.ndarray | None = None,
        right_rot: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if left_rot is None:
            if self._default_foot_rot_l is None:
                raise RuntimeError("default foot orientation not set; call cache_default_orientation first")
            left_rot = self._default_foot_rot_l
        if right_rot is None:
            if self._default_foot_rot_r is None:
                raise RuntimeError("default foot orientation not set; call cache_default_orientation first")
            right_rot = self._default_foot_rot_r
        return (
            _homogeneous_matrix(left_rot, np.asarray(left_xyz, dtype=np.float64)),
            _homogeneous_matrix(right_rot, np.asarray(right_xyz, dtype=np.float64)),
        )

    def cache_default_orientation(self, q_leg_g1: np.ndarray) -> None:
        _, rot_l, _, rot_r = self.default_foot_state(q_leg_g1)
        self._default_foot_rot_l = rot_l
        self._default_foot_rot_r = rot_r

    def solve_ik(
        self,
        left_xyz: np.ndarray,
        right_xyz: np.ndarray,
        current_leg_q_g1: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve for 12 leg joint angles (G1 motor order) from foot positions in pelvis frame."""
        if current_leg_q_g1 is not None:
            self.init_data = self._g1_leg_to_pin(current_leg_q_g1)

        left_tf, right_tf = self.targets_from_xyz(left_xyz, right_xyz)
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_tf_l, left_tf)
        self.opti.set_value(self.param_tf_r, right_tf)
        self.opti.set_value(self.var_q_last, self.init_data)

        fallback = (
            self._pin_leg_to_g1(self.init_data)
            if current_leg_q_g1 is None
            else np.asarray(current_leg_q_g1, dtype=np.float64)
        )
        converged = True
        try:
            self.opti.solve()
            sol_q = self.opti.value(self.var_q)
        except Exception as e:
            converged = False
            logger.error(f"Leg IK convergence error: {e}")
            sol_q = self.opti.debug.value(self.var_q)

        self.smooth_filter.add_data(sol_q)
        sol_q = self.smooth_filter.filtered_data
        self.init_data = sol_q

        if not converged:
            logger.warning("Leg IK did not converge; returning last solution")
            return fallback

        return self._pin_leg_to_g1(sol_q)

    def solve_ik_dls(
        self,
        left_xyz: np.ndarray,
        right_xyz: np.ndarray,
        current_leg_q_g1: np.ndarray,
        left_rot: np.ndarray | None = None,
        right_rot: np.ndarray | None = None,
        iters: int = 100,
        damping: float = 1e-2,
        max_step: float = 0.4,
        pos_weight: float = 1.0,
        rot_weight: float = 0.3,
        tol: float = 1e-4,
    ) -> np.ndarray:
        """Fast damped-least-squares leg IK (sub-ms), warm-started from the current pose.

        Iteratively Newton-steps ``q`` toward foot pose targets using the frame
        Jacobian, instead of solving a full NLP. Ideal for interactive/real-time use
        where the target moves in small increments each step.
        """
        pin = self._pin
        model = self.reduced_robot.model
        data = self.reduced_robot.data

        if left_rot is None:
            left_rot = self._default_foot_rot_l
        if right_rot is None:
            right_rot = self._default_foot_rot_r
        if left_rot is None or right_rot is None:
            raise RuntimeError("default foot orientation not set; call cache_default_orientation first")

        q = self._g1_leg_to_pin(np.asarray(current_leg_q_g1, dtype=np.float64))
        lower = model.lowerPositionLimit
        upper = model.upperPositionLimit
        weights = np.tile(
            np.array([pos_weight] * 3 + [rot_weight] * 3, dtype=np.float64), 2
        )  # 12-vector

        targets = (
            (self.left_foot_id, np.asarray(left_xyz, dtype=np.float64), np.asarray(left_rot)),
            (self.right_foot_id, np.asarray(right_xyz, dtype=np.float64), np.asarray(right_rot)),
        )

        err = np.zeros(12)
        jac = np.zeros((12, model.nv))
        eye = np.eye(12)
        for _ in range(iters):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            for k, (fid, pos, rot) in enumerate(targets):
                target_se3 = pin.SE3(rot, pos)
                # Pose error expressed in the foot's LOCAL frame: log( oMf^{-1} * target ).
                local_err = pin.log6(data.oMf[fid].actInv(target_se3)).vector
                err[6 * k : 6 * k + 6] = local_err
                jac[6 * k : 6 * k + 6, :] = pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL)

            if np.linalg.norm(err) < tol:
                break
            we = weights * err
            wj = weights[:, None] * jac
            # dq = Jw^T (Jw Jw^T + λ² I)^{-1} e_w
            dq = wj.T @ np.linalg.solve(wj @ wj.T + damping**2 * eye, we)
            step = np.linalg.norm(dq)
            if step > max_step:
                dq *= max_step / step
            q = pin.integrate(model, q, dq)
            q = np.clip(q, lower, upper)

        return self._pin_leg_to_g1(q)
