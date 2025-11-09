# !/usr/bin/env python

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

import time
from dataclasses import dataclass

import numpy as np

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    RobotProcessorPipeline,
    TransitionKey,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.rotation import Rotation
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_target_action")
@dataclass
class ForwardKinematicsJointsToEETargetAction(RobotActionProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    end_effector_step_sizes: dict
    max_gripper_pos: float

    def action(self, action: RobotAction) -> RobotAction:
        # return compute_forward_kinematics_joints_to_ee(action, self.kinematics, self.motor_names)
        teleop_action = action
        raw_joint_pos = self.transition.get(TransitionKey.OBSERVATION)

        leader_pos = np.array([teleop_action[f"{motor}.pos"] for motor in self.motor_names])
        follower_pos = np.array([raw_joint_pos[f"{motor}.pos"] for motor in self.motor_names])

        leader_ee = self.kinematics.forward_kinematics(leader_pos)
        follower_ee = self.kinematics.forward_kinematics(follower_pos)
        follower_gripper_pos = raw_joint_pos["gripper.pos"]

        leader_ee_pos = leader_ee[:3, 3]
        leader_ee_rvec = Rotation.from_matrix(leader_ee[:3, :3]).as_rotvec()
        leader_gripper_pos = teleop_action["gripper.pos"]

        follower_ee_pos = follower_ee[:3, 3]
        # follower_ee_rvec = Rotation.from_matrix(follower_ee[:3, :3]).as_rotvec()

        delta_pos = leader_ee_pos - follower_ee_pos

        # For rotation: compute relative rotation from follower to leader
        # R_leader = R_follower * R_delta  =>  R_delta = R_follower^T * R_leader
        r_delta = follower_ee[:3, :3].T @ leader_ee[:3, :3]
        delta_rvec = Rotation.from_matrix(r_delta).as_rotvec()
        delta_gripper = leader_gripper_pos - follower_gripper_pos

        desired = np.eye(4, dtype=float)
        desired[:3, :3] = follower_ee[:3, :3] @ r_delta
        desired[:3, 3] = follower_ee[:3, 3] + delta_pos

        pos = desired[:3, 3]
        tw = Rotation.from_matrix(desired[:3, :3]).as_rotvec()

        assert np.allclose(pos, leader_ee_pos), "Position delta computation error"
        assert np.allclose(tw, leader_ee_rvec), "Orientation delta computation error"
        assert np.isclose(follower_gripper_pos + delta_gripper, leader_gripper_pos), (
            "Gripper delta computation error"
        )

        new_action = {}
        new_action["enabled"] = True
        new_action["target_x"] = float(delta_pos[0] / self.end_effector_step_sizes["x"])
        new_action["target_y"] = float(delta_pos[1] / self.end_effector_step_sizes["y"])
        new_action["target_z"] = float(delta_pos[2] / self.end_effector_step_sizes["z"])
        new_action["target_wx"] = float(delta_rvec[0] / self.end_effector_step_sizes["wx"])
        new_action["target_wy"] = float(delta_rvec[1] / self.end_effector_step_sizes["wy"])
        new_action["target_wz"] = float(delta_rvec[2] / self.end_effector_step_sizes["wz"])
        new_action["gripper_vel"] = float(
            np.clip(delta_gripper, -self.max_gripper_pos, self.max_gripper_pos) / self.max_gripper_pos
        )
        return new_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # TODO: implement feature transformation
        return features


FPS = 20

# Initialize the robot and teleoperator config
follower_config = SO100FollowerConfig(port="/dev/usb_follower_arm", id="main_follower", use_degrees=True)
leader_config = SO101LeaderConfig(port="/dev/usb_leader_arm", id="main_leader", use_degrees=True)

# Initialize the robot and teleoperator
follower = SO100Follower(follower_config)
leader = SO101Leader(leader_config)

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
follower_kinematics_solver = RobotKinematics(
    urdf_path="../SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(follower.bus.motors.keys()),
)

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
leader_kinematics_solver = RobotKinematics(
    urdf_path="../SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(leader.bus.motors.keys()),
)

# Build pipeline to convert teleop joints to EE action
leader_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
    steps=[
        ForwardKinematicsJointsToEETargetAction(
            kinematics=leader_kinematics_solver,
            motor_names=list(leader.bus.motors.keys()),
            end_effector_step_sizes={
                "x": 0.006,
                "y": 0.01,
                "z": 0.005,
                "wx": 0.03490658503988659,
                "wy": 0.05235987755982988,
                "wz": 0.08726646259971647,
            },
            max_gripper_pos=30.0,
        )
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# build pipeline to convert EE action to robot joints
ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    [
        EEReferenceAndDelta(
            kinematics=follower_kinematics_solver,
            # end_effector_step_sizes={"x": 0.006, "y": 0.01, "z": 0.005},
            end_effector_step_sizes={
                "x": 0.006,
                "y": 0.01,
                "z": 0.005,
                "wx": 0.03490658503988659,
                "wy": 0.05235987755982988,
                "wz": 0.08726646259971647,
            },
            motor_names=list(follower.bus.motors.keys()),
            use_latched_reference=False,
            use_ik_solution=False,
        ),
        EEBoundsAndSafety(
            end_effector_bounds={
                "min": [0.115, -0.165, -0.018],
                "max": [0.28, 0.16, 0.2],
            },
            # end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            max_ee_step_m=0.05,
        ),
        GripperVelocityToJoint(
            clip_max=30.0,
            speed_factor=0.2,
            discrete_gripper=False,
            scale_velocity=True,
        ),
        InverseKinematicsEEToJoints(
            kinematics=follower_kinematics_solver,
            motor_names=list(follower.bus.motors.keys()),
            initial_guess_current_joints=True,
        ),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Connect to the robot and teleoperator
follower.connect()
leader.connect()

# Init rerun viewer
init_rerun(session_name="so100_so100_EE_teleop")

print("Starting teleop loop...")
while True:
    t0 = time.perf_counter()

    # Get robot observation
    robot_obs = follower.get_observation()

    # Get teleop observation
    leader_joints_obs = leader.get_action()

    # teleop joints -> teleop EE action
    leader_ee_act = leader_to_ee((leader_joints_obs, robot_obs))

    # teleop EE -> robot joints
    follower_joints_act = ee_to_follower_joints((leader_ee_act, robot_obs))

    # r_ee = follower_kinematics_solver.forward_kinematics(
    #     np.array(
    #         [robot_obs[f"{n}.pos"] for n in follower_kinematics_solver.joint_names],
    #         dtype=float,
    #     )
    # )
    # l_ee = leader_kinematics_solver.forward_kinematics(
    #     np.array(
    #         [leader_joints_obs[f"{n}.pos"] for n in leader_kinematics_solver.joint_names],
    #         dtype=float,
    #     )
    # )

    # r_curr = np.array(
    #     [robot_obs[f"{n}.pos"] for n in follower_kinematics_solver.joint_names],
    #     dtype=float,
    # )
    # # q_target = follower_kinematics_solver.inverse_kinematics(
    # #     r_curr,
    # #     l_ee,
    # # )
    # print(f"r_curr: {r_curr}")
    # # print(f"Leader EE pose for follower IK target: {l_ee}")
    # # print(f"Follower IK target joints to reach leader EE: {q_target}")

    # r_pos = r_ee[:3, 3].tolist()
    # r_tw = Rotation.from_matrix(r_ee[:3, :3]).as_rotvec().tolist()
    # l_pos = l_ee[:3, 3].tolist()
    # l_tw = Rotation.from_matrix(l_ee[:3, :3]).as_rotvec().tolist()

    # t_des = np.eye(4, dtype=float)
    # # t_des[:3, :3] = l_ee[:3, :3]
    # # t_des[:3, 3] = l_ee[:3, 3]
    # t_des = np.eye(4, dtype=float)
    # t_des[:3, :3] = Rotation.from_rotvec(l_tw).as_matrix()
    # t_des[:3, 3] = l_pos

    # q_target = follower_kinematics_solver.inverse_kinematics(r_curr, t_des)
    # print(f"Follower EE pos: {r_pos}, tw: {r_tw}")
    # print(f"Leader EE pos: {l_pos}, tw: {l_tw}")
    # # print(f"Desired EE pose for IK: {t_des}")
    # print(f"Follower IK target joints: {q_target.tolist()}")

    # Send action to robot
    _ = follower.send_action(follower_joints_act)

    # Visualize
    log_rerun_data(observation=leader_ee_act, action=follower_joints_act)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
