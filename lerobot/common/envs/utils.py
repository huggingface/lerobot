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
import einops
import numpy as np
import torch
from torch import Tensor

from lerobot.common.envs.configs import EnvConfig
from lerobot.common.utils.utils import get_channel_first_image_shape
from lerobot.configs.types import FeatureType, PolicyFeature


def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    # TODO(aliberts, rcadene): refactor this to use features from the environment (no hardcoding)
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    # TODO: You have to merge all tensors from agent key and extra key
    # You don't keep sensor param key in the observation
    # And you keep sensor data rgb
    for key, img in observations.items():
        if "images" not in key:
            continue

        # TODO(aliberts, rcadene): use transforms.ToTensor()?
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)

        if img.ndim == 3:
            img = img.unsqueeze(0)

        # sanity check that images are channel last
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

        # sanity check that images are uint8
        assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

        # convert to channel first of type float32 in range [0,1]
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)
        img /= 255

        return_observations[key] = img
        # obs state agent qpos and qvel
        # image

    if "environment_state" in observations:
        return_observations["observation.environment_state"] = torch.from_numpy(
            observations["environment_state"]
        ).float()

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    # requirement for "agent_pos"
    # return_observations["observation.state"] = torch.from_numpy(observations["agent_pos"]).float()
    return_observations["observation.state"] = observations["observation.state"].float()
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

        policy_key = env_cfg.features_map.get(key, key)
        policy_features[policy_key] = feature

    return policy_features


def preprocess_maniskill_observation(
    observations: dict[str, np.ndarray],
) -> dict[str, Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    # TODO: You have to merge all tensors from agent key and extra key
    # You don't keep sensor param key in the observation
    # And you keep sensor data rgb
    q_pos = observations["agent"]["qpos"]
    q_vel = observations["agent"]["qvel"]
    tcp_pos = observations["extra"]["tcp_pose"]
    img = observations["sensor_data"]["base_camera"]["rgb"]

    _, h, w, c = img.shape
    assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

    # sanity check that images are uint8
    assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

    # convert to channel first of type float32 in range [0,1]
    img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
    img = img.type(torch.float32)
    img /= 255

    state = torch.cat([q_pos, q_vel, tcp_pos], dim=-1)

    return_observations["observation.image"] = img
    return_observations["observation.state"] = state
    return return_observations
