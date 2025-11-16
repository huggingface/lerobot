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

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "Put the green tissues in the box."
HF_MODEL_ID = "models/act"
HF_DATASET_ID = "pinkocelot/il_gym2"

# 检查并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Device Configuration ===")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("⚠ WARNING: GPU not available, inference will be slow on CPU")

# Create the robot configuration & robot
robot_config = LeKiwiClientConfig(remote_ip="192.168.31.203", id="my_awesome_kiwi")
robot = LeKiwiClient(robot_config)

# Create policy with explicit device
policy = ACTPolicy.from_pretrained(HF_MODEL_ID)
policy.model = policy.model.to(device)
policy.config.device = device

# 验证模型在正确设备上
actual_device = next(policy.model.parameters()).device
print(f"✓ Model loaded on: {actual_device}")

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, ACTION)
obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Build Policy Processors
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy,
    pretrained_path=HF_MODEL_ID,
    dataset_stats=dataset.meta.stats,
    preprocessor_overrides={"device_processor": {"device": str(device)}},  # 使用正确的设备
)

# Connect the robot
robot.connect()

# TODO(Steven): Update this example to use pipelines
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# Initialize the keyboard listener and rerun visualization
listener, events = init_keyboard_listener()
init_rerun(session_name="lekiwi_evaluate")

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

print("Starting evaluate loop...")
recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Running inference, recording eval episode {recorded_episodes} of {NUM_EPISODES}")

    # Main record loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (
        (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
    ):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

    if events["rerecord_episode"]:
        log_say("Re-record episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    # Save episode
    dataset.save_episode()
    recorded_episodes += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
listener.stop()

dataset.finalize()
dataset.push_to_hub()
