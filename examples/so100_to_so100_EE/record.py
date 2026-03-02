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


from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.pipelines import (
    make_so10x_fk_observation_pipeline,
    make_so10x_ik_action_pipeline,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.teleoperators.so_leader.pipelines import make_so10x_leader_fk_pipeline
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.pipeline_utils import (
    build_dataset_features,
    check_action_space_compatibility,
    check_observation_space_compatibility,
)
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "My task description"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"

# NOTE: Use the URDF from the SO-ARM100 repo:
# https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
URDF_PATH = "./SO101/so101_new_calib.urdf"


def main():
    # Create the robot and teleoperator configurations
    camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
    follower_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5A460814411",
        id="my_awesome_follower_arm",
        cameras=camera_config,
        use_degrees=True,
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

    # Dataset features are derived automatically from robot/teleop pipelines
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=build_dataset_features(follower, leader, use_videos=True),
        robot_type=follower.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Connect the robot and teleoperator
    leader.connect()
    follower.connect()

    # Verify action/observation space alignment (warns on mismatch)
    check_action_space_compatibility(leader, follower)
    check_observation_space_compatibility(follower, leader)

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="recording_ee")

    try:
        if not leader.is_connected or not follower.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        print("Starting record loop...")
        episode_idx = 0
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

            # Pipelines applied automatically inside robot.get_observation(),
            # teleop.get_action(), and robot.send_action()
            record_loop(
                robot=follower,
                events=events,
                fps=FPS,
                teleop=leader,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (
                episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=follower,
                    events=events,
                    fps=FPS,
                    teleop=leader,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                )

            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
            episode_idx += 1

    finally:
        # Clean up
        log_say("Stop recording")
        leader.disconnect()
        follower.disconnect()
        listener.stop()

        dataset.finalize()
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
