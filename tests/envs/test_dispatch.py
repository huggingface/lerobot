"""Tests for the benchmark dispatch refactor (create_envs / get_env_processors on EnvConfig)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import gymnasium as gym
import pytest
import torch
from gymnasium.envs.registration import register, registry as gym_registry

from lerobot.configs.types import PolicyFeature
from lerobot.envs.configs import EnvConfig, LiberoEnv
from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
from lerobot.processor import LiberoActionProcessorStep, LiberoProcessorStep
from lerobot.utils.constants import OBS_PREFIX, OBS_STATE

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


def test_processors_delegation_supports_legacy_override_signature():
    """External EnvConfig subclasses with the old get_env_processors() signature keep working."""
    from lerobot.processor.pipeline import DataProcessorPipeline

    @EnvConfig.register_subclass("_dispatch_legacy_proc_test")
    @dataclass
    class _Env(EnvConfig):
        task: str = "x"
        features: dict[str, PolicyFeature] = field(default_factory=dict)

        @property
        def gym_kwargs(self):
            return {}

        def get_env_processors(self):
            return DataProcessorPipeline(steps=[]), DataProcessorPipeline(steps=[])

    pre, post = make_env_pre_post_processors(_Env(), policy_cfg=object())
    assert isinstance(pre, DataProcessorPipeline)
    assert isinstance(post, DataProcessorPipeline)


def test_libero_evo1_processors_use_padded_state_and_env_action_dim():
    """EVO1 uses padded LIBERO state features while env actions stay executable."""

    class _Evo1Config:
        type = "evo1"
        max_state_dim = 24

    cfg = LiberoEnv()
    pre, post = make_env_pre_post_processors(cfg, policy_cfg=_Evo1Config())
    assert isinstance(pre.steps[0], LiberoProcessorStep)
    assert pre.steps[0].max_state_dim == 24
    assert isinstance(post.steps[0], LiberoActionProcessorStep)
    assert post.steps[0].action_dim == cfg.features["action"].shape[0] == 7
    assert post.steps[0].binarize_gripper is True

    class _OtherConfig:
        type = "other"

    pre_other, post_other = make_env_pre_post_processors(cfg, policy_cfg=_OtherConfig())
    assert pre_other.steps[0].max_state_dim is None
    assert post_other.steps[0].binarize_gripper is False

    cfg.binarize_gripper = False
    _, post_disabled = make_env_pre_post_processors(cfg, policy_cfg=_Evo1Config())
    assert post_disabled.steps[0].binarize_gripper is False


def test_libero_processor_pads_state_to_max_dim():
    step = LiberoProcessorStep(max_state_dim=24)
    observation = {
        OBS_PREFIX
        + "robot_state": {
            "eef": {
                "pos": torch.tensor([[1.0, 2.0, 3.0]]),
                "quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            },
            "gripper": {"qpos": torch.tensor([[4.0, 5.0]])},
        }
    }

    state = step.observation(observation)[OBS_STATE]
    assert state.shape == (1, 24)
    assert torch.allclose(state[:, :8], torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0]]))
    assert torch.count_nonzero(state[:, 8:]).item() == 0


def test_libero_action_processor_slices_padded_action():
    step = LiberoActionProcessorStep(action_dim=7)
    action = torch.arange(2 * 3 * 24, dtype=torch.float32).reshape(2, 3, 24)

    sliced = step.action(action)
    assert sliced.shape == (2, 3, 7)
    assert torch.equal(sliced, action[..., :7])

    with pytest.raises(ValueError, match="smaller than action_dim=7"):
        step.action(torch.zeros(2, 6))


def test_libero_action_processor_can_binarize_gripper():
    step = LiberoActionProcessorStep(action_dim=7, binarize_gripper=True)
    action = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 7.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.6, 7.0],
        ],
        dtype=torch.float32,
    )

    processed = step.action(action)

    assert processed.shape == (2, 7)
    assert torch.equal(processed[:, :6], action[:, :6])
    assert torch.equal(processed[:, 6], torch.tensor([1.0, -1.0]))
    assert torch.equal(action[:, 6], torch.tensor([0.5, 0.6]))


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
        assert isinstance(env, gym.vector.VectorEnv)
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

        def get_env_processors(self, policy_cfg=None):
            return DataProcessorPipeline(steps=[]), DataProcessorPipeline(steps=[])

    pre, post = _Env().get_env_processors()
    assert isinstance(pre, DataProcessorPipeline)
