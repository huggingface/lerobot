#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import warnings
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.utils import get_channel_first_image_shape

def preprocess_observation(
    observations: dict[str, np.ndarray], cfg: dict[str, Any] = None
) -> dict[str, Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observations: Dictionary of observation batches from a Gym vector environment.
        cfg: Policy config containing expected feature keys.
    Returns:
        Dictionary of observation batches with keys renamed to match policy expectations.
    """
    return_observations = {}

    # expected keys from policy
    policy_img_keys = list(cfg.image_features.keys()) if cfg else ["observation.image"]
    state_key = cfg.robot_state_feature_key if cfg else "observation.state"

    # handle images
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            env_img_keys = list(observations["pixels"].keys())
            imgs = observations["pixels"]
        else:
            env_img_keys = ["pixels"]
            imgs = {"pixels": observations["pixels"]}

        # build rename map env_key -> policy_key
        rename_map = dict(zip(env_img_keys, policy_img_keys))

        for imgkey, img in imgs.items():
            target_key = rename_map.get(imgkey, imgkey)

            img = torch.from_numpy(img)

            # sanity checks
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, got {img.shape=}"
            assert img.dtype == torch.uint8, f"expect torch.uint8, got {img.dtype=}"

            # channel last → channel first, normalize
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.float() / 255.0

            return_observations[target_key] = img

    # handle state
    if "environment_state" in observations:
        return_observations["observation.environment_state"] = torch.from_numpy(
            observations["environment_state"]
        ).float()

    return_observations[state_key] = torch.from_numpy(observations["agent_pos"]).float()

    if "task" in observations:
        return_observations["task"] = observations["task"]

    return return_observations

def env_to_policy_features(env_cfg: EnvConfig) -> dict[str, PolicyFeature]:
    # TODO(aliberts, rcadene): remove this hardcoding of keys and just use the nested keys as is
    # (need to also refactor preprocess_observation and externalize normalization from policies)
    policy_features = {}
    for key, ft in env_cfg.features.items():
        if ft.type is FeatureType.VISUAL:
            if len(ft.shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={ft.shape})")

            shape = get_channel_first_image_shape(ft.shape)
            feature = PolicyFeature(type=ft.type, shape=shape)
        else:
            feature = ft

        policy_key = env_cfg.features_map[key]
        policy_features[policy_key] = feature
    return policy_features


def are_all_envs_same_type(env: gym.vector.VectorEnv) -> bool:
    first_type = type(env.envs[0])  # Get type of first env
    return all(type(e) is first_type for e in env.envs)  # Fast type check


def check_env_attributes_and_types(env: gym.vector.VectorEnv) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("once", UserWarning)  # Apply filter only in this function

        if not (hasattr(env.envs[0], "task_description") and hasattr(env.envs[0], "task")):
            warnings.warn(
                "The environment does not have 'task_description' and 'task'. Some policies require these features.",
                UserWarning,
                stacklevel=2,
            )
        if not are_all_envs_same_type(env):
            warnings.warn(
                "The environments have different types. Make sure you infer the right task from each environment. Empty task will be passed instead.",
                UserWarning,
                stacklevel=2,
            )


def add_envs_task(env: gym.vector.VectorEnv, observation: dict[str, Any]) -> dict[str, Any]:
    """Adds task feature to the observation dict with respect to the first environment attribute."""
    if hasattr(env.envs[0], "task_description"):
        observation["task"] = env.call("task_description")
    elif hasattr(env.envs[0], "task"):
        observation["task"] = env.call("task")
    else:  #  For envs without language instructions, e.g. aloha transfer cube and etc.
        num_envs = observation[list(observation.keys())[0]].shape[0]
        observation["task"] = ["" for _ in range(num_envs)]
    return observation
