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

"""Run a trained policy on the Hans Robot S30.

Loads a policy checkpoint from the Hugging Face Hub (or a local path) and
executes it in a closed-loop on the real robot.

Edit the constants below to match your setup before running::

    python examples/hans_s30/evaluate.py

Key settings to edit:
- ``ROBOT_IP``    – IPv4 address of the Hans controller.
- ``MODEL_ID``    – HF Hub model ID or local path to the policy checkpoint.
- ``TASK``        – Natural-language task description (required for VLA models).
- ``MAX_EPISODES``– Number of evaluation episodes to run.
- ``MAX_STEPS``   – Maximum steps per episode before resetting.
"""

import time

import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.hans_s30 import HansS30, HansS30RobotConfig
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep

# ── User settings ────────────────────────────────────────────────────────────
ROBOT_IP = "192.168.115.11"
MODEL_ID = "<hf_username>/<policy_repo_id>"
TASK = "Pick up the red block and place it in the bin"
MAX_EPISODES = 5
MAX_STEPS = 200
FPS = 30
# ─────────────────────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy
    from lerobot.policies import get_policy_class

    policy_cls = get_policy_class(MODEL_ID)
    policy = policy_cls.from_pretrained(MODEL_ID)
    policy.eval()
    policy.to(device)

    preprocess, postprocess = make_pre_post_processors(policy.config, MODEL_ID)

    # Robot setup
    camera_config = {
        "wrist_cam": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
        "base_cam": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS),
    }

    robot_config = HansS30RobotConfig(
        ip=ROBOT_IP,
        port=10003,
        id="my_hans_s30",
        cameras=camera_config,
    )
    robot = HansS30(robot_config)
    robot.connect()

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    try:
        for ep in range(MAX_EPISODES):
            print(f"\n── Episode {ep + 1} / {MAX_EPISODES} ──")
            for _step in range(MAX_STEPS):
                t0 = time.perf_counter()

                obs = robot.get_observation()
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=device,
                    task=TASK,
                    robot_type=robot.name,
                )
                obs_preprocessed = preprocess(obs_frame)

                with torch.inference_mode():
                    action_tensor = policy.select_action(obs_preprocessed)

                action_postprocessed = postprocess(action_tensor)
                action = make_robot_action(action_postprocessed, dataset_features)
                robot.send_action(action)

                precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

            print("Episode finished.")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
