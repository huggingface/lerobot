"""Tests for the benchmark dispatch refactor (create_envs / get_env_processors on EnvConfig)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import gymnasium as gym
import pytest
from gymnasium.envs.registration import register, registry as gym_registry

from lerobot.configs.types import PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors

logger = logging.getLogger(__name__)


def test_registry_all_types():
    """make_env_config should resolve every registered EnvConfig subclass via the registry."""
    known = list(EnvConfig.get_known_choices().keys())
    assert len(known) >= 6
    for t in known:
        cfg = make_env_config(t)
        if not isinstance(cfg, EnvConfig):
            continue
        assert cfg.type == t


def test_unknown_type():
    with pytest.raises(ValueError, match="not registered"):
        make_env_config("nonexistent")


def test_identity_processors():
    """Base class get_env_processors() returns identity pipelines."""
    cfg = make_env_config("aloha")
    pre, post = cfg.get_env_processors()
    assert len(pre.steps) == 0 and len(post.steps) == 0


def test_delegation():
    """make_env() should call cfg.create_envs(), not use if/elif dispatch."""
    sentinel = {"delegated": {0: "marker"}}
    fake = type(
        "Fake",
        (),
        {
            "hub_path": None,
            "create_envs": lambda self, n_envs, use_async_envs=False: sentinel,
        },
    )()
    result = make_env(fake, n_envs=1)
    assert result is sentinel


def test_processors_delegation():
    """make_env_pre_post_processors delegates to cfg.get_env_processors()."""
    cfg = make_env_config("aloha")
    pre, post = make_env_pre_post_processors(cfg, policy_cfg=None)
    assert len(pre.steps) == 0


def test_base_create_envs():
    """Base class create_envs() should build a single-task VectorEnv via gym.make()."""
    gym_id = "_dispatch_test/CartPole-v99"
    if gym_id not in gym_registry:
        register(id=gym_id, entry_point="gymnasium.envs.classic_control:CartPoleEnv")

    @EnvConfig.register_subclass("_dispatch_base_test")
    @dataclass
    class _Env(EnvConfig):
        task: str = "CartPole-v99"
        fps: int = 10
        features: dict[str, PolicyFeature] = field(default_factory=dict)

        @property
        def package_name(self):
            return "_dispatch_test"

        @property
        def gym_id(self):
            return gym_id

        @property
        def gym_kwargs(self):
            return {}

    try:
        envs = _Env().create_envs(n_envs=2)
        assert "_dispatch_base_test" in envs
        env = envs["_dispatch_base_test"][0]
        assert isinstance(env, gym.vector.SyncVectorEnv)
        assert env.num_envs == 2
        env.close()
    finally:
        if gym_id in gym_registry:
            del gym_registry[gym_id]


def test_custom_create_envs_override():
    """A custom EnvConfig subclass can override create_envs()."""
    mock_vec = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])

    @EnvConfig.register_subclass("_dispatch_custom_test")
    @dataclass
    class _Env(EnvConfig):
        task: str = "x"
        features: dict[str, PolicyFeature] = field(default_factory=dict)

        @property
        def gym_kwargs(self):
            return {}

        def create_envs(self, n_envs, use_async_envs=False):
            return {"custom_suite": {0: mock_vec}}

    try:
        result = make_env(_Env(), n_envs=1)
        assert "custom_suite" in result
    finally:
        mock_vec.close()


def test_custom_get_env_processors_override():
    """A custom EnvConfig subclass can override get_env_processors()."""
    from lerobot.processor.pipeline import DataProcessorPipeline

    @EnvConfig.register_subclass("_dispatch_proc_test")
    @dataclass
    class _Env(EnvConfig):
        task: str = "x"
        features: dict[str, PolicyFeature] = field(default_factory=dict)

        @property
        def gym_kwargs(self):
            return {}

        def get_env_processors(self):
            return DataProcessorPipeline(steps=[]), DataProcessorPipeline(steps=[])

    pre, post = _Env().get_env_processors()
    assert isinstance(pre, DataProcessorPipeline)
