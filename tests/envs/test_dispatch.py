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
from lerobot.processor import LiberoProcessorStep
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


def test_libero_fps_controls_simulator_frequency():
    cfg = LiberoEnv(fps=17)

    assert cfg.gym_kwargs["control_freq"] == 17


def test_libero_rejects_nonpositive_fps():
    with pytest.raises(ValueError, match="fps must be positive"):
        LiberoEnv(fps=0)


def test_libero_create_envs_without_simulator_raises_clear_error(monkeypatch):
    """When the LIBERO simulator package is absent — the stock situation on non-Linux
    installs, where ``hf-libero``'s ``sys_platform == 'linux'`` marker makes
    ``pip install 'lerobot[libero]'`` silently omit it — ``create_envs()`` must fail
    fast with an actionable message instead of a cryptic ``ModuleNotFoundError`` raised
    deep inside the ``lerobot.envs.libero`` import. Regression test for #4388."""
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "libero":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)

    with pytest.raises(ModuleNotFoundError, match="Linux") as exc_info:
        LiberoEnv().create_envs(n_envs=1)

    message = str(exc_info.value)
    assert "hf-libero" in message
    assert "lerobot[libero]" in message


def _hide_libero(monkeypatch):
    """Make `find_spec("libero")` report the simulator as absent.

    Reproduces a non-Linux install, where `lerobot[libero]`'s
    `sys_platform == 'linux'` marker on `hf-libero` silently omits it. Patched
    rather than assumed, so these tests still mean something on Linux CI.
    """
    import importlib.util as importlib_util

    real_find_spec = importlib_util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "libero":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)


def test_libero_validate_platform_without_simulator(monkeypatch):
    """`validate_platform()` is the startup-time twin of the `create_envs()` guard.

    It must raise the same actionable error from the same single source, so a
    LIBERO run on a non-Linux install fails before any setup work rather than at
    the first rollout (#4388).
    """
    _hide_libero(monkeypatch)

    with pytest.raises(ModuleNotFoundError, match="Linux") as exc_info:
        LiberoEnv().validate_platform()

    message = str(exc_info.value)
    assert "hf-libero" in message
    assert "lerobot[libero]" in message


def test_validate_platform_defaults_to_noop():
    """Only envs with a platform constraint override the hook.

    Every other registered env must accept it silently, so
    `TrainPipelineConfig.validate()` can call it unconditionally.
    """
    for env_type in EnvConfig.get_known_choices():
        cfg = make_env_config(env_type)
        # `register_subclass` also accepts classes that do not inherit from
        # EnvConfig (plugins may register one, and a test fixture leaves one in
        # the registry), so screen them out the same way test_registry_all_types
        # above does.
        if not isinstance(cfg, EnvConfig) or isinstance(cfg, LiberoEnv):
            continue
        cfg.validate_platform()  # must not raise


def test_validate_platform_not_called_on_construction(monkeypatch):
    """Constructing a config must stay side-effect free.

    `make_env_config()` builds every registered env just to enumerate the
    choices, so moving this check into `__post_init__` would break callers that
    never intended to run one. See the note on the base method.
    """
    _hide_libero(monkeypatch)

    make_env_config("libero")  # must not raise even with the simulator absent
    LiberoEnv()


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


def test_libero_processors_are_policy_agnostic():
    cfg = LiberoEnv()
    pre, post = make_env_pre_post_processors(cfg, policy_cfg=object())

    assert isinstance(pre.steps[0], LiberoProcessorStep)
    assert len(post.steps) == 0


def test_libero_processor_flattens_state_to_raw_8_dim():
    step = LiberoProcessorStep()
    observation = {
        OBS_PREFIX + "robot_state": {
            "eef": {
                "pos": torch.tensor([[1.0, 2.0, 3.0]]),
                "quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            },
            "gripper": {"qpos": torch.tensor([[4.0, 5.0]])},
        }
    }

    state = step.observation(observation)[OBS_STATE]
    assert state.shape == (1, 8)
    assert torch.allclose(state, torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0]]))


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

        def get_env_processors(self):
            return DataProcessorPipeline(steps=[]), DataProcessorPipeline(steps=[])

    pre, post = _Env().get_env_processors()
    assert isinstance(pre, DataProcessorPipeline)
