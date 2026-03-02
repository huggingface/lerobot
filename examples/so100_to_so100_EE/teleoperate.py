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

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.pipelines import (
    make_so10x_fk_observation_pipeline,
    make_so10x_ik_action_pipeline,
)
from lerobot.scripts.lerobot_teleoperate import teleop_loop
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.teleoperators.so_leader.pipelines import make_so10x_leader_fk_pipeline
from lerobot.utils.pipeline_utils import check_action_space_compatibility
from lerobot.utils.visualization_utils import init_rerun

FPS = 30

# NOTE: Use the URDF from the SO-ARM100 repo:
# https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
URDF_PATH = "./SO101/so101_new_calib.urdf"


def main():
    # Initialize the robot and teleoperator config
    follower_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5A460814411", id="my_awesome_follower_arm", use_degrees=True
    )
    leader_config = SO100LeaderConfig(port="/dev/tty.usbmodem5A460819811", id="my_awesome_leader_arm")

    # Initialize the robot and teleoperator
    follower = SO100Follower(follower_config)
    leader = SO100Leader(leader_config)

    # Attach EE-space pipelines to the objects
    motor_names = list(follower.bus.motors.keys())
    follower.set_output_pipeline(make_so10x_fk_observation_pipeline(URDF_PATH, motor_names))
    follower.set_input_pipeline(make_so10x_ik_action_pipeline(URDF_PATH, motor_names))
    leader.set_output_pipeline(make_so10x_leader_fk_pipeline(URDF_PATH, list(leader.bus.motors.keys())))

    # Verify action space alignment (warns if leader EE ≠ follower action_features)
    check_action_space_compatibility(leader, follower)

    # Connect to the robot and teleoperator
    follower.connect()
    leader.connect()

    # Init rerun viewer
    init_rerun(session_name="so100_so100_EE_teleop")

    print("Starting teleop loop...")
    try:
        # Pipelines applied automatically inside teleop.get_action() and robot.send_action()
        teleop_loop(teleop=leader, robot=follower, fps=FPS, display_data=True)
    finally:
        follower.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    main()
