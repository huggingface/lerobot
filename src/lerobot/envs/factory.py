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
from __future__ import annotations

from typing import Any

import gymnasium as gym

from lerobot.envs.configs import EnvConfig, HubEnvConfig
from lerobot.envs.utils import _call_make_env, _download_hub_file, _import_hub_module, _normalize_hub_result


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    try:
        cls = EnvConfig.get_choice_class(env_type)
    except KeyError as err:
        raise ValueError(
            f"Environment type '{env_type}' is not registered. "
            f"Available: {list(EnvConfig.get_known_choices().keys())}"
        ) from err
    return cls(**kwargs)


def make_env_pre_post_processors(
    env_cfg: EnvConfig,
    policy_cfg: Any,
) -> tuple[Any, Any]:
    """
    Create preprocessor and postprocessor pipelines for environment observations.

    Returns a tuple of (preprocessor, postprocessor). By default, delegates to
    ``env_cfg.get_env_processors()``.  The XVLAConfig policy-specific override
    stays here because it depends on the *policy* config, not the env config.
    """
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig

    if isinstance(policy_cfg, XVLAConfig):
        from lerobot.policies.xvla.processor_xvla import make_xvla_libero_pre_post_processors

        return make_xvla_libero_pre_post_processors()

    return env_cfg.get_env_processors()


def make_env(
    cfg: EnvConfig | str,
    n_envs: int = 1,
    use_async_envs: bool = False,
    hub_cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Makes a gym vector environment according to the config or Hub reference.

    Args:
        cfg (EnvConfig | str): Either an `EnvConfig` object describing the environment to build locally,
            or a Hugging Face Hub repository identifier (e.g. `"username/repo"`). In the latter case,
            the repo must include a Python file (usually `env.py`).
        n_envs (int, optional): The number of parallelized env to return. Defaults to 1.
        use_async_envs (bool, optional): Whether to return an AsyncVectorEnv or a SyncVectorEnv. Defaults to
            False.
        hub_cache_dir (str | None): Optional cache path for downloaded hub files.
        trust_remote_code (bool): **Explicit consent** to execute remote code from the Hub.
            Default False — must be set to True to import/exec hub `env.py`.
    Raises:
        ValueError: if n_envs < 1
        ModuleNotFoundError: If the requested env package is not installed

    Returns:
        dict[str, dict[int, gym.vector.VectorEnv]]:
            A mapping from suite name to indexed vectorized environments.
            - For multi-task benchmarks (e.g., LIBERO): one entry per suite, and one vec env per task_id.
            - For single-task environments: a single suite entry (cfg.type) with task_id=0.

    """
    # if user passed a hub id string (e.g., "username/repo", "username/repo@main:env.py")
    # simplified: only support hub-provided `make_env`
    # TODO: (jadechoghari): deprecate string API and remove this check
    if isinstance(cfg, str):
        hub_path: str | None = cfg
    elif isinstance(cfg, HubEnvConfig):
        hub_path = cfg.hub_path
    else:
        hub_path = None

    # If hub_path is set, download and call hub-provided `make_env`
    if hub_path:
        # _download_hub_file will raise the same RuntimeError if trust_remote_code is False
        repo_id, file_path, local_file, revision = _download_hub_file(
            hub_path, trust_remote_code, hub_cache_dir
        )

        # import and surface clear import errors
        module = _import_hub_module(local_file, repo_id)

        # call the hub-provided make_env
        env_cfg = None if isinstance(cfg, str) else cfg
        raw_result = _call_make_env(module, n_envs=n_envs, use_async_envs=use_async_envs, cfg=env_cfg)

        # normalize the return into {suite: {task_id: vec_env}}
        return _normalize_hub_result(raw_result)

    # At this point, cfg must be an EnvConfig (not a string) since hub_path would have been set otherwise
    if isinstance(cfg, str):
        raise TypeError("cfg should be an EnvConfig at this point")

    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")

    return cfg.create_envs(n_envs=n_envs, use_async_envs=use_async_envs)
