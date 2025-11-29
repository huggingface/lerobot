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
import importlib
from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import registry as gym_registry

from lerobot.envs.configs import AlohaEnv, EnvConfig, LiberoEnv, PushtEnv
from lerobot.envs.utils import _call_make_env, _download_hub_file, _import_hub_module, _normalize_hub_result
from lerobot.processor import ProcessorStep
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    if env_type == "aloha":
        return AlohaEnv(**kwargs)
    elif env_type == "pusht":
        return PushtEnv(**kwargs)
    elif env_type == "libero":
        return LiberoEnv(**kwargs)
    else:
        raise ValueError(f"Policy type '{env_type}' is not available.")


def make_env_pre_post_processors(
    env_cfg: EnvConfig,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
]:
    """
    Create preprocessor and postprocessor pipelines for environment observations.

    This function creates processor pipelines that transform raw environment
    observations and actions. By default, it returns identity processors that do nothing.
    For specific environments like LIBERO, it adds environment-specific processing steps.

    Args:
        env_cfg: The configuration of the environment.

    Returns:
        A tuple containing:
            - preprocessor: Pipeline that processes environment observations
            - postprocessor: Pipeline that processes environment outputs (currently identity)
    """
    # Preprocessor and Postprocessor steps are Identity for most environments
    preprocessor_steps: list[ProcessorStep] = []
    postprocessor_steps: list[ProcessorStep] = []

    # For LIBERO environments, add the LiberoProcessorStep to preprocessor
    if isinstance(env_cfg, LiberoEnv) or "libero" in env_cfg.type:
        preprocessor_steps.append(LiberoProcessorStep())

    preprocessor = PolicyProcessorPipeline(steps=preprocessor_steps)
    postprocessor = PolicyProcessorPipeline(steps=postprocessor_steps)

    return preprocessor, postprocessor


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
    if isinstance(cfg, str):
        # _download_hub_file will raise the same RuntimeError if trust_remote_code is False
        repo_id, file_path, local_file, revision = _download_hub_file(cfg, trust_remote_code, hub_cache_dir)

        # import and surface clear import errors
        module = _import_hub_module(local_file, repo_id)

        # call the hub-provided make_env
        raw_result = _call_make_env(module, n_envs=n_envs, use_async_envs=use_async_envs)

        # normalize the return into {suite: {task_id: vec_env}}
        return _normalize_hub_result(raw_result)

    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")

    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv

    if "libero" in cfg.type:
        from lerobot.envs.libero import create_libero_envs

        if cfg.task is None:
            raise ValueError("LiberoEnv requires a task to be specified")

        return create_libero_envs(
            task=cfg.task,
            n_envs=n_envs,
            camera_name=cfg.camera_name,
            init_states=cfg.init_states,
            gym_kwargs=cfg.gym_kwargs,
            env_cls=env_cls,
        )
    elif "metaworld" in cfg.type:
        from lerobot.envs.metaworld import create_metaworld_envs

        if cfg.task is None:
            raise ValueError("MetaWorld requires a task to be specified")

        return create_metaworld_envs(
            task=cfg.task,
            n_envs=n_envs,
            gym_kwargs=cfg.gym_kwargs,
            env_cls=env_cls,
        )

    if cfg.gym_id not in gym_registry:
        print(f"gym id '{cfg.gym_id}' not found, attempting to import '{cfg.package_name}'...")
        try:
            importlib.import_module(cfg.package_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Package '{cfg.package_name}' required for env '{cfg.type}' not found. "
                f"Please install it or check PYTHONPATH."
            ) from e

        if cfg.gym_id not in gym_registry:
            raise gym.error.NameNotFound(
                f"Environment '{cfg.gym_id}' not registered even after importing '{cfg.package_name}'."
            )

    def _make_one():
        return gym.make(cfg.gym_id, disable_env_checker=cfg.disable_env_checker, **(cfg.gym_kwargs or {}))

    vec = env_cls([_make_one for _ in range(n_envs)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    # normalize to {suite: {task_id: vec_env}} for consistency
    suite_name = cfg.type  # e.g., "pusht", "aloha"
    return {suite_name: {0: vec}}
