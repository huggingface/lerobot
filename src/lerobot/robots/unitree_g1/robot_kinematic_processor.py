import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer
import os
import sys

import logging_mp

logger_mp = logging_mp.get_logger(__name__)
parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

from huggingface_hub import snapshot_download

import numpy as np
import matplotlib.pyplot as plt




class WeightedMovingFilter:
    def __init__(self, weights, data_size=14):
        self._window_size = len(weights)
        self._weights = np.array(weights)
        # assert np.isclose(np.sum(self._weights), 1.0), "[WeightedMovingFilter] the sum of weights list must be 1.0!"
        self._data_size = data_size
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = []

    def _apply_filter(self):
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]

        data_array = np.array(self._data_queue)
        temp_filtered_data = np.zeros(self._data_size)
        for i in range(self._data_size):
            temp_filtered_data[i] = np.convolve(data_array[:, i], self._weights, mode="valid")[-1]

        return temp_filtered_data

    def add_data(self, new_data):
        assert len(new_data) == self._data_size

        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return  # skip duplicate data

        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)

        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    @property
    def filtered_data(self):
        return self._filtered_data


class G1_29_ArmIK:
    def __init__(self, Unit_Test=False, Visualization=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        self.repo_path = snapshot_download("lerobot/unitree-g1-mujoco")
        urdf_path = os.path.join(self.repo_path, "assets", "g1_body29_hand14.urdf")
        mesh_dir = os.path.join(self.repo_path, "assets")
        
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)

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
        print(f"Pinocchio arm joint order: {self._arm_joint_names_pin}")
        self._arm_reorder_g1_to_pin = [
            self._arm_joint_names_g1.index(name) for name in self._arm_joint_names_pin
        ]
        # Inverse mapping to return tau in G1 motor order.
        self._arm_reorder_pin_to_g1 = np.argsort(self._arm_reorder_g1_to_pin)

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "L_ee",
                self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "R_ee",
                self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        # for i in range(self.reduced_robot.model.nframes):
        #     frame = self.reduced_robot.model.frames[i]
        #     frame_id = self.reduced_robot.model.getFrameId(frame.name)
        #     logger_mp.debug(f"Frame ID: {frame_id}, Name: {frame.name}")
        # for idx, name in enumerate(self.reduced_robot.model.names):
        #     logger_mp.debug(f"{idx}: {name}")

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
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit, self.var_q, self.reduced_robot.model.upperPositionLimit
            )
        )
        self.opti.minimize(
            50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost
        )

        opts = {
            "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-6},
            "print_time": False,  # print or not
            "calc_lam_p": False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.displayFrames(True, frame_ids=[107, 108], axis_length=0.15, axis_width=5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ["L_ee_target", "R_ee_target"]
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 20
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    # If the robot arm is not the same size as your arm :)
    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        # left_wrist, right_wrist = self.scale_arms(left_wrist, right_wrist)
        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(left_wrist)  # for visualization
            self.vis.viewer["R_ee_target"].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)  # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(
                self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv)
            )

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff

        except Exception as e:
            logger_mp.error(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(
                self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv)
            )

            logger_mp.error(
                f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}"
            )
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)

    def solve_tau(self, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        try:
            q_g1 = np.array(current_lr_arm_motor_q, dtype=float)
            if q_g1.shape[0] != len(self._arm_joint_names_g1):
                raise ValueError(
                    f"Expected {len(self._arm_joint_names_g1)} arm joints, got {q_g1.shape[0]}"
                )
            q_pin = q_g1[self._arm_reorder_g1_to_pin]
            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                q_pin,
                np.zeros(14),
                np.zeros(self.reduced_robot.model.nv),
            )
            return sol_tauff[self._arm_reorder_pin_to_g1]

        except Exception as e:
            logger_mp.error(f"ERROR in convergence, plotting debug info.{e}")
            return np.zeros(self.reduced_robot.model.nv)
