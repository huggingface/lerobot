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

import logging
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import gymnasium as gym

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.sim import EnvDescriptor, LocalVectorEnv
from lerobot.sim.integrations import envhub
from lerobot.utils.constants import ACTION, OBS_ENV_STATE

from .configs import EnvConfig, HubEnvConfig, LiberoEnvConfig

if TYPE_CHECKING:
    from lerobot.env_server import RemoteVectorEnv

logger = logging.getLogger(__name__)

# Timeout of each request to a served env.
SERVED_TIMEOUT_S = 120.0


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
    cfg: EnvConfig,
    n_envs: int = 1,
    use_async_envs: bool = False,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Build the vector envs of a config.

    Args:
        cfg (`EnvConfig`):
            The env config. An EnvHub repository is `HubEnvConfig(hub_path="user/repo", trust_remote_code=True)`.
        n_envs (`int`, *optional*, defaults to `1`):
            Number of envs in each vector env.
        use_async_envs (`bool`, *optional*, defaults to `False`):
            Run each env in its own worker process.

    Returns:
        `dict[str, dict[int, gym.vector.VectorEnv]]`: The vector envs, as `{suite: {task_id: env}}`. A
        single-task env has one suite, with task ID 0.

    Raises:
        ValueError: If `n_envs` is below 1.
        RuntimeError: If an EnvHub repo's code would run without its config's `trust_remote_code`.
        ModuleNotFoundError: If the simulator, or a package the repo's `env.py` needs, is not installed.
    """
    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")
    if isinstance(cfg, str):
        raise TypeError(
            "make_env() no longer takes a Hub repo ID. The equivalent is "
            f'make_env(HubEnvConfig(hub_path="{cfg}", trust_remote_code=True, use_native_batch=True), n_envs=...). '
            "Without use_native_batch, LeRobot builds the envs itself, as lerobot-eval and the env server do."
        )
    if type(cfg).create_envs is not EnvConfig.create_envs:
        raise TypeError(
            f"{type(cfg).__name__} overrides EnvConfig.create_envs(), which make_env no longer calls: make_env "
            "builds every env from its config's sim_settings(). Return the simulator and its options from "
            "sim_settings() instead, as described in https://huggingface.co/docs/lerobot/adding_benchmarks"
        )
    if isinstance(cfg, HubEnvConfig) and cfg.use_native_batch:
        return _make_native_batch(cfg, n_envs, use_async_envs)
    descriptor, make = _env_factory(cfg, n_envs, use_async_envs)
    groups = [task.strip() for task in cfg.task.split(",")] if cfg.task else list(descriptor.tasks)
    return {
        group: {task_id: make(group, task_id) for task_id in _suite(descriptor, group)} for group in groups
    }


def make_task_env(
    cfg: EnvConfig,
    task_group: str | None = None,
    task_id: int | None = None,
    n_envs: int = 1,
    use_async_envs: bool = False,
) -> LocalVectorEnv | RemoteVectorEnv:
    """Build the local or served vector env of one task; no `task_id` means the suite's first task."""
    descriptor, make = _env_factory(cfg, n_envs, use_async_envs, task_group, task_id)
    group = task_group or next(iter(descriptor.tasks))
    task_ids = _suite(descriptor, group)
    if task_id is None:
        task_id = task_ids[0]
    if task_id not in task_ids:
        raise ValueError(f"Unknown task ID {task_id} for {group!r}; this suite has {sorted(task_ids)}")
    return make(group, task_id)


def _make_native_batch(
    cfg: HubEnvConfig, n_envs: int, use_async_envs: bool
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Return an EnvHub repo's own vector envs (e.g. GPU batches) unchanged."""
    if cfg.server is not None:
        raise ValueError("use_native_batch runs locally; it cannot use an env server")
    if not cfg.hub_path:
        raise ValueError("--env.hub_path is required for an EnvHub environment")
    # Provider configs (Isaac Lab Arena, Unitree G1) subclass HubEnvConfig and also pass themselves as `cfg`,
    # since their options are config fields.
    provider = None if type(cfg) is HubEnvConfig else cfg
    return envhub.make_envs(
        cfg.hub_path, n_envs, use_async_envs, cfg.trust_remote_code, cfg=provider, **cfg.make_env_kwargs
    )


def _env_factory(
    cfg: EnvConfig,
    n_envs: int,
    use_async_envs: bool,
    task_group: str | None = None,
    task_id: int | None = None,
) -> tuple[EnvDescriptor, Callable[[str, int], LocalVectorEnv | RemoteVectorEnv]]:
    """Return the env's descriptor, and a function building the vector env of one of its tasks."""
    settings = cfg.sim_settings()
    server = cfg.server
    if server is not None:
        from lerobot.env_server import EnvClient, RemoteVectorEnv

        if cfg.max_parallel_tasks != 1:
            raise ValueError("A served environment runs one task at a time")
        connect = partial(
            EnvClient,
            server.endpoint,
            server.deployment or cfg.type,
            SERVED_TIMEOUT_S,
            handshake_timeout_s=server.handshake_timeout_s,
            zenoh_config_path=server.zenoh_config_path,
            max_payload_bytes=server.max_payload_bytes,
        )
        probe = connect()
        try:
            descriptor = probe.describe()
        finally:
            probe.close()
        check_served(descriptor.config, settings, server.endpoint)

        def make(group: str, task_id: int) -> LocalVectorEnv | RemoteVectorEnv:
            return RemoteVectorEnv(
                connect(encoding=server.encoding, jpeg_quality=server.jpeg_quality),
                n_envs,
                group,
                task_id,
                use_async_envs and n_envs > 1,
                descriptor,
            )

        return descriptor, make

    sim = cfg.sim_config()
    parallel = use_async_envs and n_envs > 1
    # Reuse the env built to read the descriptor for its own task instead of building it twice.
    first = [LocalVectorEnv(sim, n_envs, task_group, task_id, parallel)]
    descriptor = first[0].descriptor

    def make_local(group: str, task_id: int) -> LocalVectorEnv | RemoteVectorEnv:
        if first:
            env = first.pop()
            if (group, task_id) == (env.task_group, env.task_id):
                return env
            env.close()
        # With the descriptor given, the other tasks build their sims on first use.
        return LocalVectorEnv(sim, n_envs, group, task_id, parallel, descriptor)

    return descriptor, make_local


def _suite(descriptor: EnvDescriptor, group: str) -> list[int]:
    """Return the task IDs of one suite of the env, or raise naming the suites it has."""
    if group not in descriptor.tasks:
        raise ValueError(f"Unknown task {group!r}; this environment has {sorted(descriptor.tasks)}")
    return descriptor.tasks[group]


def descriptor_features(descriptor: EnvDescriptor) -> dict[str, PolicyFeature]:
    """Return an env's observation and action features, typed as for a dataset; images stay channel last."""
    features = {}
    for spec in descriptor.features:
        if spec.kind == "rgb":
            kind = FeatureType.VISUAL
        else:
            kind = FeatureType.ENV if spec.name == OBS_ENV_STATE else FeatureType.STATE
        features[spec.name] = PolicyFeature(type=kind, shape=spec.shape)
    features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=descriptor.action_feature.shape)
    return features


def check_served(served: dict[str, Any], expected: dict[str, Any], endpoint: str) -> None:
    """Raise if the server runs a different env than ``expected``; warn if only its kwargs differ."""

    def plain(value: Any) -> Any:
        # Make YAML, wire and dataclass values comparable: tuples become lists, recursively.
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    # A server opens any of the tasks it configures (all of them for an empty list), so the requested tasks
    # are checked against the descriptor's task list instead.
    errors = [
        f"{key}: server {served.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if key not in ("kwargs", "task") and value is not None and plain(value) != plain(served.get(key))
    ]
    served_kwargs = served.get("kwargs") or {}
    # hub_path and trust_remote_code may differ: the env server loads EnvHub repos from its own image.
    differences = [
        f"kwargs.{key}: server {served_kwargs.get(key)!r}, expected {value!r}"
        for key, value in (expected.get("kwargs") or {}).items()
        if key not in ("hub_path", "trust_remote_code") and plain(value) != plain(served_kwargs.get(key))
    ]
    if errors:
        raise ValueError(
            f"The env server at {endpoint} runs another environment than this command describes "
            f"({'; '.join(errors)}). Pass the same --env options as the server's configuration."
        )
    for difference in differences:
        logger.warning("The env server at %s differs in %s", endpoint, difference)
