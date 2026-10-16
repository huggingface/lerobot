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

from lerobot.configs import FeatureType, PolicyFeature

# Compatibility exports of the torch-free Hub loader, shared with the simulator server.
from lerobot.sims.native.envhub import (
    _call_make_env as _call_make_env,
    _download_hub_file as _download_hub_file,
    _import_hub_module as _import_hub_module,
    _load_module_from_path as _load_module_from_path,
    _normalize_hub_result as _normalize_hub_result,
    _parse_hub_url as _parse_hub_url,
)
from lerobot.sims.native.utils import parse_camera_names as parse_camera_names
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR
from lerobot.utils.utils import get_channel_first_image_shape

from .configs import EnvConfig


def _convert_nested_dict(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _convert_nested_dict(v)
        elif isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v)
        else:
            result[k] = v
    return result


def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    # TODO(jadechoghari, imstevenpmwork): refactor this to use features from the environment (no hardcoding)
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

    if "agent_pos" in observations:
        agent_pos = torch.from_numpy(observations["agent_pos"]).float()
        if agent_pos.dim() == 1:
            agent_pos = agent_pos.unsqueeze(0)
        return_observations[OBS_STATE] = agent_pos

    if "robot_state" in observations:
        return_observations[f"{OBS_STR}.robot_state"] = _convert_nested_dict(observations["robot_state"])

    # Handle IsaacLab Arena format: observations have 'policy' and 'camera_obs' keys
    if "policy" in observations:
        return_observations[f"{OBS_STR}.policy"] = observations["policy"]

    if "camera_obs" in observations:
        return_observations[f"{OBS_STR}.camera_obs"] = observations["camera_obs"]

    # Pass through any remaining ndarray/tensor keys not already handled above,
    # so env plugins can expose extra observation keys via get_env_processors().
    _handled = {"pixels", "environment_state", "agent_pos", "robot_state", "policy", "camera_obs"}
    for key, value in observations.items():
        if key in _handled:
            continue
        target = f"{OBS_STR}.{key}"
        if target in return_observations:
            continue
        if isinstance(value, np.ndarray):
            val = torch.from_numpy(value).float()
            if val.dim() == 1:
                val = val.unsqueeze(0)
            return_observations[target] = val
        elif isinstance(value, Tensor):
            val = value.float()
            if val.dim() == 1:
                val = val.unsqueeze(0)
            return_observations[target] = val

    return return_observations


def env_to_policy_features(env_cfg: EnvConfig) -> dict[str, PolicyFeature]:
    # TODO(jadechoghari, imstevenpmwork): remove this hardcoding of keys and just use the nested keys as is
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


def _sub_env_has_attr(env: gym.vector.VectorEnv, attr: str) -> bool:
    try:
        env.get_attr(attr)
        return True
    except (AttributeError, Exception):
        return False


def check_env_attributes_and_types(env: gym.vector.VectorEnv) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("once", UserWarning)

        if not (_sub_env_has_attr(env, "task_description") and _sub_env_has_attr(env, "task")):
            warnings.warn(
                "The environment does not have 'task_description' and 'task'. Some policies require these features.",
                UserWarning,
                stacklevel=2,
            )


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
