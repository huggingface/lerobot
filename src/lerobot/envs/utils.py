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
from collections.abc import Mapping, Sequence
from functools import singledispatch
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.utils.utils import get_channel_first_image_shape


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
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            imgs = {f"{OBS_IMAGES}.{key}": img for key, img in observations["pixels"].items()}
        else:
            imgs = {OBS_IMAGE: observations["pixels"]}

        for imgkey, img in imgs.items():
            # TODO(aliberts, rcadene): use transforms.ToTensor()?
            img_tensor = torch.from_numpy(img)

            # When preprocessing observations in a non-vectorized environment, we need to add a batch dimension.
            # This is the case for human-in-the-loop RL where there is only one environment.
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)
            # sanity check that images are channel last
            _, h, w, c = img_tensor.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img_tensor.shape=}"

            # sanity check that images are uint8
            assert img_tensor.dtype == torch.uint8, f"expect torch.uint8, but instead {img_tensor.dtype=}"

            # convert to channel first of type float32 in range [0,1]
            img_tensor = einops.rearrange(img_tensor, "b h w c -> b c h w").contiguous()
            img_tensor = img_tensor.type(torch.float32)
            img_tensor /= 255

            return_observations[imgkey] = img_tensor

    if "environment_state" in observations:
        env_state = torch.from_numpy(observations["environment_state"]).float()
        if env_state.dim() == 1:
            env_state = env_state.unsqueeze(0)

        return_observations[OBS_ENV_STATE] = env_state

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    agent_pos = torch.from_numpy(observations["agent_pos"]).float()
    if agent_pos.dim() == 1:
        agent_pos = agent_pos.unsqueeze(0)
    return_observations[OBS_STATE] = agent_pos

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
        task_result = env.call("task_description")

        if isinstance(task_result, tuple):
            task_result = list(task_result)

        if not isinstance(task_result, list):
            raise TypeError(f"Expected task_description to return a list, got {type(task_result)}")
        if not all(isinstance(item, str) for item in task_result):
            raise TypeError("All items in task_description result must be strings")

        observation["task"] = task_result
    elif hasattr(env.envs[0], "task"):
        task_result = env.call("task")

        if isinstance(task_result, tuple):
            task_result = list(task_result)

        if not isinstance(task_result, list):
            raise TypeError(f"Expected task to return a list, got {type(task_result)}")
        if not all(isinstance(item, str) for item in task_result):
            raise TypeError("All items in task result must be strings")

        observation["task"] = task_result
    else:  #  For envs without language instructions, e.g. aloha transfer cube and etc.
        num_envs = observation[list(observation.keys())[0]].shape[0]
        observation["task"] = ["" for _ in range(num_envs)]
    return observation


def _close_single_env(env: Any) -> None:
    try:
        env.close()
    except Exception as exc:
        print(f"Exception while closing env {env}: {exc}")


@singledispatch
def close_envs(obj: Any) -> None:
    """Default: raise if the type is not recognized."""
    raise NotImplementedError(f"close_envs not implemented for type {type(obj).__name__}")


@close_envs.register
def _(env: Mapping) -> None:
    for v in env.values():
        if isinstance(v, Mapping):
            close_envs(v)
        elif hasattr(v, "close"):
            _close_single_env(v)


@close_envs.register
def _(envs: Sequence) -> None:
    if isinstance(envs, (str | bytes)):
        return
    for v in envs:
        if isinstance(v, Mapping) or isinstance(v, Sequence) and not isinstance(v, (str | bytes)):
            close_envs(v)
        elif hasattr(v, "close"):
            _close_single_env(v)


@close_envs.register
def _(env: gym.Env) -> None:
    _close_single_env(env)
