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
import torch

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
    create_transition,
    identity_transition,
)
from lerobot.robots.robot import Robot
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.rotation import Rotation


def reset_follower_position(robot_arm: Robot, target_position: np.ndarray) -> None:
    """Reset robot arm to target position using smooth trajectory."""
    current_position_dict = robot_arm.bus.sync_read("Present_Position")
    current_position = np.array(
        [current_position_dict[name] for name in current_position_dict],
        dtype=np.float32,
    )
    trajectory = torch.from_numpy(
        np.linspace(current_position, target_position, 50)
    )  # NOTE: 30 is just an arbitrary number
    for pose in trajectory:
        action_dict = dict(zip(current_position_dict, pose, strict=False))
        robot_arm.bus.sync_write("Goal_Position", action_dict)
        busy_wait(0.015)


@dataclass
class LogRobotAction(RobotActionProcessorStep):
    def action(self, action: RobotAction) -> RobotAction:
        print(f"Robot action: {action}")
        return action

    def transform_features(self, features):
        # features[PipelineFeatureType.ACTION][ACTION] = PolicyFeature(
        #     type=FeatureType.ACTION, shape=(len(self.motor_names),)
        # )
        return features


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
    use_ik_solution: bool = False

    def action(self, action: RobotAction) -> RobotAction:
        # return compute_forward_kinematics_joints_to_ee(action, self.kinematics, self.motor_names)
        teleop_action = action
        raw_joint_pos = self.transition.get(TransitionKey.OBSERVATION)

        leader_pos = np.array([teleop_action[f"{motor}.pos"] for motor in self.motor_names])
        follower_pos = np.array([raw_joint_pos[f"{motor}.pos"] for motor in self.motor_names])

        leader_ee = self.kinematics.forward_kinematics(leader_pos)

        if self.use_ik_solution and "IK_solution" in self.transition.get(TransitionKey.COMPLEMENTARY_DATA):
            q_raw = self.transition.get(TransitionKey.COMPLEMENTARY_DATA)["IK_solution"]
            follower_ee = self.kinematics.forward_kinematics(q_raw)
        else:
            follower_ee = self.kinematics.forward_kinematics(follower_pos)

        follower_ee_pos = follower_ee[:3, 3]
        follower_ee_rvec = Rotation.from_matrix(follower_ee[:3, :3]).as_rotvec()
        follower_gripper_pos = raw_joint_pos["gripper.pos"]

        leader_ee_pos = leader_ee[:3, 3]
        leader_ee_rvec = Rotation.from_matrix(leader_ee[:3, :3]).as_rotvec()
        leader_gripper_pos = teleop_action["gripper.pos"]

        print("f pos:", follower_ee_pos)
        print("l pos:", leader_ee_pos)

        print("f rvec:", follower_ee_rvec)
        print("l rvec:", leader_ee_rvec)

        # follower_ee_pos = follower_ee[:3, 3]
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

        # Normalize the action to the range [-1, 1]
        delta_pos = delta_pos / np.array(
            [
                self.end_effector_step_sizes["x"],
                self.end_effector_step_sizes["y"],
                self.end_effector_step_sizes["z"],
            ]
        )
        delta_rvec = delta_rvec / np.array(
            [
                self.end_effector_step_sizes["wx"],
                self.end_effector_step_sizes["wy"],
                self.end_effector_step_sizes["wz"],
            ]
        )

        # Check if any of the normalized deltas exceed 1.0

        max_normalized_pos = max(
            abs(delta_pos[0]),
            abs(delta_pos[1]),
            abs(delta_pos[2]),
        )

        # max_normalized_rot = max(
        #     # abs(delta_rvec[0]),
        #     abs(delta_rvec[1]),
        #     abs(delta_rvec[2]),
        # )

        # Use the same scaling factor for both position and rotation
        # max_normalized = max(max_normalized_pos, max_normalized_rot)
        if max_normalized_pos > 1.0:
            print(f"Warning: EE delta too large, scaling. Max normalized delta: {max_normalized_pos}")
            print(f"Original delta_pos: {delta_pos}, delta_rvec: {delta_rvec}")
            # Scale proportionally
            delta_pos = delta_pos / max_normalized_pos
            delta_rvec = delta_rvec / max_normalized_pos

        new_action = {}
        new_action["enabled"] = True
        new_action["target_x"] = float(delta_pos[0])
        new_action["target_y"] = float(delta_pos[1])
        new_action["target_z"] = float(delta_pos[2])
        new_action["target_wx"] = float(delta_rvec[0])
        new_action["target_wy"] = float(delta_rvec[1])
        new_action["target_wz"] = float(delta_rvec[2])
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
follower_config = SO101FollowerConfig(port="/dev/usb_follower_arm_a", id="follower_arm_a", use_degrees=True)
leader_config = SO101LeaderConfig(port="/dev/usb_leader_arm_a", id="leader_arm_a", use_degrees=True)

# Initialize the robot and teleoperator
follower = SO101Follower(follower_config)
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

end_effector_step_sizes = {
    "x": 0.01,
    "y": 0.01,
    "z": 0.01,
    "wx": 30 * np.pi / 180,
    "wy": 30 * np.pi / 180,
    "wz": 30 * np.pi / 180,
}


# Build pipeline to convert teleop joints to EE action
leader_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
    steps=[
        LogRobotAction(),
        ForwardKinematicsJointsToEETargetAction(
            kinematics=leader_kinematics_solver,
            motor_names=list(leader.bus.motors.keys()),
            end_effector_step_sizes=end_effector_step_sizes,
            max_gripper_pos=30.0,
            use_ik_solution=True,
        ),
        LogRobotAction(),
    ],
    to_transition=identity_transition,
    to_output=identity_transition,
)

# build pipeline to convert EE action to robot joints
ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    [
        LogRobotAction(),
        EEReferenceAndDelta(
            kinematics=follower_kinematics_solver,
            # end_effector_step_sizes={"x": 0.006, "y": 0.01, "z": 0.005},
            end_effector_step_sizes=end_effector_step_sizes,
            motor_names=list(follower.bus.motors.keys()),
            use_latched_reference=False,
            use_ik_solution=True,
        ),
        LogRobotAction(),
        EEBoundsAndSafety(
            end_effector_bounds={
                "min": [-0.05, -0.55, -0.0075],
                "max": [0.55, 0.55, 0.55],
            },
            # end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            max_ee_step_m=0.05,
        ),
        LogRobotAction(),
        GripperVelocityToJoint(
            clip_max=30.0,
            speed_factor=0.2,
            discrete_gripper=False,
            scale_velocity=True,
        ),
        LogRobotAction(),
        InverseKinematicsRLStep(
            kinematics=follower_kinematics_solver,
            motor_names=list(follower.bus.motors.keys()),
            initial_guess_current_joints=False,
        ),
        LogRobotAction(),
    ],
    to_transition=identity_transition,
    to_output=identity_transition,
)

# Connect to the robot and teleoperator
follower.connect()
leader.connect()

reset_pose = [0.0, 10, 20, 60.00, 90.00, 10.00]

start_time = time.perf_counter()
reset_follower_position(follower, np.array(reset_pose))
reset_follower_position(leader, np.array(reset_pose))
busy_wait(5.0 - (time.perf_counter() - start_time))
# time.sleep(10)
leader.bus.sync_write("Torque_Enable", 0)

# Init rerun viewer
# init_rerun(session_name="so100_so100_EE_teleop")

transition = None

print("Starting teleop loop...")
while True:
    print("New loop iteration")
    t0 = time.perf_counter()

    # Get robot observation
    robot_obs = follower.get_observation()

    # Get teleop observation
    leader_joints_obs = leader.get_action()

    # teleop joints -> teleop EE action
    if transition is None:
        transition = create_transition(action=leader_joints_obs, observation=robot_obs)
    else:
        transition = create_transition(
            action=leader_joints_obs,
            observation=robot_obs,
            complementary_data=transition.get(TransitionKey.COMPLEMENTARY_DATA),
        )

    transition = leader_to_ee(transition)
    leader_ee_act = transition[TransitionKey.ACTION]

    # teleop EE -> robot joints
    transition = create_transition(
        action=leader_ee_act,
        observation=robot_obs,
        complementary_data=transition.get(TransitionKey.COMPLEMENTARY_DATA),
    )
    transition = ee_to_follower_joints(transition)
    follower_joints_act = transition[TransitionKey.ACTION]

    # Send action to robot
    _ = follower.send_action(follower_joints_act)

    # Visualize
    # log_rerun_data(observation=leader_ee_act, action=follower_joints_act)

    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
