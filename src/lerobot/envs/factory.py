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

from lerobot.sim.integrations import envhub

from .configs import EnvConfig, HubEnvConfig, LiberoEnvConfig


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
    ``env_cfg.get_env_processors()``.  The XVLAConfig and Cosmos3Config overrides for LIBERO environments
    stay here because they depend on the *policy* config, not the env config.
    """
    from lerobot.policies.cosmos3.configuration_cosmos3 import Cosmos3Config
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig

    if isinstance(policy_cfg, XVLAConfig) and isinstance(env_cfg, LiberoEnvConfig):
        from lerobot.policies.xvla.processor_xvla import make_xvla_libero_pre_post_processors

        return make_xvla_libero_pre_post_processors()

    if isinstance(policy_cfg, Cosmos3Config) and isinstance(env_cfg, LiberoEnvConfig):
        from lerobot.policies.cosmos3.processor_cosmos3 import make_cosmos3_libero_pre_post_processors

        return make_cosmos3_libero_pre_post_processors()

    return env_cfg.get_env_processors()


def make_env(
    cfg: EnvConfig | str,
    n_envs: int = 1,
    use_async_envs: bool = False,
    hub_cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Build the vector envs of a config, or of a Hub repo's `make_env`.

    Args:
        cfg (`EnvConfig | str`):
            The env config, or a Hugging Face Hub repo ID (e.g. `"username/repo"`) whose `env.py` builds the
            envs.
        n_envs (`int`, *optional*, defaults to `1`):
            Number of envs in each vector env.
        use_async_envs (`bool`, *optional*, defaults to `False`):
            Run each env in its own worker process.
        hub_cache_dir (`str`, *optional*):
            Folder to cache the downloaded `env.py` in. Only used with a repo ID.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Consent to run a Hub repo's code, needed for a repo ID or an EnvHub config.

    Returns:
        `dict[str, dict[int, gym.vector.VectorEnv]]`: The vector envs, as `{suite: {task_id: env}}`. A
        single-task env has one suite, with task ID 0.

    Raises:
        ValueError: If `n_envs` is below 1.
        RuntimeError: If a Hub repo's code would run without `trust_remote_code`.
        ModuleNotFoundError: If the simulator, or a package the repo's `env.py` needs, is not installed.
    """
    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")
    # TODO: (jadechoghari): deprecate string API and remove this check
    if isinstance(cfg, str):
        return envhub.make_envs(cfg, n_envs, use_async_envs, trust_remote_code, hub_cache_dir)
    if isinstance(cfg, HubEnvConfig) and trust_remote_code:
        cfg.trust_remote_code = True
    return cfg.create_envs(n_envs=n_envs, use_async_envs=use_async_envs)
