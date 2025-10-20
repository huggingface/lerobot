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

"""
This script demonstrates how to evaluate pretrained vision-language-action (VLA) policies
such as SmolVLA on Libero benchmark tasks using the LeRobot framework.

It showcases the full evaluation pipeline — from environment creation to policy inference,
visualization, and result logging — and is intended as a reference for benchmarking or
integrating new robotic policies.

Features included in this script:
- loading Libero environments (e.g., libero_spatial, libero_object) via `make_env`.
- initializing pretrained policies (e.g., SmolVLA) from Hugging Face using `make_policy`.
- applying preprocessing and postprocessing transformations for model compatibility.
- running evaluation rollouts and recording rendered frames from the simulator.
- computing success metrics and saving rollout videos as MP4 for qualitative analysis.

The script ends by saving a rollout video (`rollout.mp4`) and printing per-environment
success indicators for quick visual and numerical evaluation.
"""

import numpy as np
import torch
import imageio.v2 as imageio
from lerobot.envs.factory import make_env, make_env_config
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.factory import make_policy_config
from lerobot.envs.utils import (
    add_envs_task,
    preprocess_observation,
)
import os 
os.environ["MUJOCO_GL"] = "egl"

SMOLVLA_LIBERO_PATH = "HuggingFaceVLA/smolvla_libero"
LIBERO_CONFIG = make_env_config("libero", task="libero_spatial")
breakpoint()
POLICY_CONFIG = make_policy_config("smolvla", pretrained_path=SMOLVLA_LIBERO_PATH)
policy = make_policy(
        cfg=POLICY_CONFIG,
        env_cfg=LIBERO_CONFIG,
)
breakpoint()
libero_env = make_env(LIBERO_CONFIG)
breakpoint()
print(type(libero_env)) # <class 'dict'>
print(libero_env.keys()) # dict_keys(['libero_spatial', 'libero_object'])

# initilize your policy, here we use smolvla
breakpoint()
policy.eval()
preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=POLICY_CONFIG,
        pretrained_path=SMOLVLA_LIBERO_PATH,
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
policy.reset()
# for the sake of this exemple we only use one env from each task
libero_spatial_env = libero_env['libero_spatial'][0]
# libero_object_env = libero_env['libero_object'][0]

# let's first run an evaluation throgut the first task
observation, info = libero_spatial_env.reset() # you can pass seeds
max_steps = 220
step = 0
all_images = []
done = np.array([False] * libero_spatial_env.num_envs)
while not np.all(done) and step < max_steps:
    observation = preprocess_observation(observation)
    observation = add_envs_task(libero_spatial_env, observation)
    observation = preprocessor(observation)
    with torch.inference_mode():
            action = policy.select_action(observation)
    action = postprocessor(action)
    # Convert to CPU / numpy.
    action_numpy = action.to("cpu").numpy() 
    # Apply the next action.
    # let's render the video
    image = libero_spatial_env.call("render")[0]
    all_images.append(image)
    observation, reward, terminated, truncated, info = libero_spatial_env.step(action_numpy)
    if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
    else:
        successes = [False] * libero_spatial_env.num_envs
    
    done = terminated | truncated | done
    if step + 1 == max_steps:
        done = np.ones_like(done, dtype=bool)
    step += 1

print("The success: ", successes)

