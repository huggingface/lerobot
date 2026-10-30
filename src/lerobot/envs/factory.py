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

from lerobot.sim.integrations.envhub import (
    _call_make_env,
    _download_hub_file,
    _import_hub_module,
    _normalize_hub_result,
)

from .configs import EnvConfig, HubEnvConfig


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


def make_hub_envs(
    hub_path: str,
    n_envs: int,
    use_async_envs: bool,
    trust_remote_code: bool,
    hub_cache_dir: str | None = None,
    cfg: EnvConfig | None = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Call a Hub repo's ``make_env`` and return its envs, as ``{suite: {task_id: env}}``."""
    repo_id, _, local_file, _ = _download_hub_file(hub_path, trust_remote_code, hub_cache_dir)
    module = _import_hub_module(local_file, repo_id)
    # A plain EnvHub config passes its make_env_kwargs. Provider configs (Isaac Lab Arena, Unitree G1)
    # subclass it and pass themselves as ``cfg``, since their options are config fields.
    options = cfg.make_env_kwargs if isinstance(cfg, HubEnvConfig) else {}
    provider_cfg = None if type(cfg) is HubEnvConfig else cfg
    raw_result = _call_make_env(
        module, n_envs=n_envs, use_async_envs=use_async_envs, cfg=provider_cfg, **options
    )
    return _normalize_hub_result(raw_result)


def make_env(
    cfg: EnvConfig | str,
    n_envs: int = 1,
    use_async_envs: bool = False,
    hub_cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Build the vector envs of a config, or of a Hub repo's ``make_env``.

    Args:
        cfg (EnvConfig | str): Either an `EnvConfig` describing the env, or a Hugging Face Hub repo ID
            (e.g. `"username/repo"`) whose `env.py` builds the envs.
        n_envs (int, optional): The number of parallelized env to return. Defaults to 1.
        use_async_envs (bool, optional): Whether to run each sub-env in its own worker process. Defaults
            to False.
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
            - For single-task envs: a single suite entry with task_id=0.

    """
    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")
    # TODO: (jadechoghari): deprecate string API and remove this check
    if isinstance(cfg, str):
        return make_hub_envs(cfg, n_envs, use_async_envs, trust_remote_code, hub_cache_dir)
    if isinstance(cfg, HubEnvConfig) and trust_remote_code:
        cfg.trust_remote_code = True
    return cfg.create_envs(n_envs=n_envs, use_async_envs=use_async_envs)
