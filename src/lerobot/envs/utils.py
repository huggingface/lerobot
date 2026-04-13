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
import importlib.util
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import singledispatch
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torch import Tensor

from lerobot.configs import FeatureType, PolicyFeature
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


class _LazyAsyncVectorEnv:
    """Defers AsyncVectorEnv creation until first use.

    Creating all tasks' AsyncVectorEnvs upfront spawns N_tasks × n_envs worker
    processes, all of which allocate EGL/GPU resources immediately. Since tasks
    are evaluated sequentially, only one task's workers need to be alive at a
    time. This wrapper stores the factory functions and creates the real
    AsyncVectorEnv on first reset()/step()/call(), keeping peak process count = n_envs.
    """

    def __init__(
        self,
        env_fns: list[Callable],
        observation_space=None,
        action_space=None,
    ):
        self._env_fns = env_fns
        self._env: gym.vector.AsyncVectorEnv | None = None
        self.num_envs = len(env_fns)
        if observation_space is not None and action_space is not None:
            self.observation_space = observation_space
            self.action_space = action_space
        else:
            tmp = env_fns[0]()
            self.observation_space = tmp.observation_space
            self.action_space = tmp.action_space
            tmp.close()
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    def _ensure(self) -> None:
        if self._env is None:
            self._env = gym.vector.AsyncVectorEnv(self._env_fns, context="forkserver", shared_memory=True)

    def reset(self, **kwargs):
        self._ensure()
        return self._env.reset(**kwargs)

    def step(self, actions):
        self._ensure()
        return self._env.step(actions)

    def call(self, name, *args, **kwargs):
        self._ensure()
        return self._env.call(name, *args, **kwargs)

    def get_attr(self, name):
        self._ensure()
        return self._env.get_attr(name)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


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


# helper to safely load a python file as a module
def _load_module_from_path(path: str, module_name: str | None = None):
    module_name = module_name or f"hub_env_{os.path.basename(path).replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Could not load module spec for {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


# helper to parse hub string (supports "user/repo", "user/repo@rev", optional path)
# examples:
#   "user/repo" -> will look for env.py at repo root
#   "user/repo@main:envs/my_env.py" -> explicit revision and path
def _parse_hub_url(hub_uri: str):
    # very small parser: [repo_id][@revision][:path]
    # repo_id is required (user/repo or org/repo)
    revision = None
    file_path = "env.py"
    if "@" in hub_uri:
        repo_and_rev, *rest = hub_uri.split(":", 1)
        repo_id, rev = repo_and_rev.split("@", 1)
        revision = rev
        if rest:
            file_path = rest[0]
    else:
        repo_id, *rest = hub_uri.split(":", 1)
        if rest:
            file_path = rest[0]
    return repo_id, revision, file_path


def _download_hub_file(
    cfg_str: str,
    trust_remote_code: bool,
    hub_cache_dir: str | None,
) -> tuple[str, str, str, str]:
    """
    Parse `cfg_str` (hub URL), enforce `trust_remote_code`, and return
    (repo_id, file_path, local_file, revision).
    """
    if not trust_remote_code:
        raise RuntimeError(
            f"Refusing to execute remote code from the Hub for '{cfg_str}'. "
            "Executing hub env modules runs arbitrary Python code from third-party repositories. "
            "If you trust this repo and understand the risks, call `make_env(..., trust_remote_code=True)` "
            "and prefer pinning to a specific revision: 'user/repo@<commit-hash>:env.py'."
        )

    repo_id, revision, file_path = _parse_hub_url(cfg_str)

    try:
        local_file = hf_hub_download(
            repo_id=repo_id, filename=file_path, revision=revision, cache_dir=hub_cache_dir
        )
    except Exception as e:
        # fallback to snapshot download
        snapshot_dir = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=hub_cache_dir)
        local_file = os.path.join(snapshot_dir, file_path)
        if not os.path.exists(local_file):
            raise FileNotFoundError(
                f"Could not find {file_path} in repository {repo_id}@{revision or 'main'}"
            ) from e

    return repo_id, file_path, local_file, revision


def _import_hub_module(local_file: str, repo_id: str) -> Any:
    """
    Import the downloaded file as a module and surface helpful import error messages.
    """
    module_name = f"hub_env_{repo_id.replace('/', '_')}"
    try:
        module = _load_module_from_path(local_file, module_name=module_name)
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise ModuleNotFoundError(
            f"Hub env '{repo_id}:{os.path.basename(local_file)}' failed to import because the dependency "
            f"'{missing}' is not installed locally.\n\n"
        ) from e
    except ImportError as e:
        raise ImportError(
            f"Failed to load hub env module '{repo_id}:{os.path.basename(local_file)}'. Import error: {e}\n\n"
        ) from e
    return module


def _call_make_env(module: Any, n_envs: int, use_async_envs: bool, cfg: EnvConfig | None) -> Any:
    """
    Ensure module exposes make_env and call it.
    """
    if not hasattr(module, "make_env"):
        raise AttributeError(
            f"The hub module {getattr(module, '__name__', 'hub_module')} must expose `make_env(n_envs=int, use_async_envs=bool)`."
        )
    entry_fn = module.make_env
    # Only pass cfg if it's not None (i.e., when an EnvConfig was provided, not a string hub ID)
    if cfg is not None:
        return entry_fn(n_envs=n_envs, use_async_envs=use_async_envs, cfg=cfg)
    else:
        return entry_fn(n_envs=n_envs, use_async_envs=use_async_envs)


def _normalize_hub_result(result: Any) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """
    Normalize possible return types from hub `make_env` into the mapping:
      { suite_name: { task_id: vector_env } }
    Accepts:
      - dict (assumed already correct)
      - gym.vector.VectorEnv
      - gym.Env (will be wrapped into SyncVectorEnv)
    """
    if isinstance(result, dict):
        return result

    # VectorEnv: use its spec.id if available
    if isinstance(result, gym.vector.VectorEnv):
        suite_name = getattr(result, "spec", None) and getattr(result.spec, "id", None) or "hub_env"
        return {suite_name: {0: result}}

    # Single Env: wrap into SyncVectorEnv
    if isinstance(result, gym.Env):
        vec = gym.vector.SyncVectorEnv([lambda: result])
        suite_name = getattr(result, "spec", None) and getattr(result.spec, "id", None) or "hub_env"
        return {suite_name: {0: vec}}

    raise ValueError(
        "Hub `make_env` must return either a mapping {suite: {task_id: vec_env}}, "
        "a gym.vector.VectorEnv, or a single gym.Env."
    )
