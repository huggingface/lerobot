# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace

import gymnasium as gym
import pytest

from lerobot.envs import factory
from lerobot.envs.configs import HubEnvConfig
from lerobot.envs.utils import _call_make_env


@pytest.mark.parametrize("use_config", [False, True])
@pytest.mark.parametrize("use_async_envs", [False, True])
def test_make_env_accepts_two_argument_hub_factory(tmp_path, monkeypatch, use_config, use_async_envs):
    """Both public loading forms support the documented two-argument factory."""
    env_file = tmp_path / "env.py"
    env_file.write_text(
        "import gymnasium as gym\n"
        "def make_env(n_envs=1, use_async_envs=False):\n"
        "    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv\n"
        "    return env_cls([lambda: gym.make('CartPole-v1') for _ in range(n_envs)])\n"
    )
    hub_path = "test/cartpole"
    monkeypatch.setattr(
        factory,
        "_download_hub_file",
        lambda *args: (hub_path, "env.py", str(env_file), "test-revision"),
    )
    cfg = HubEnvConfig(hub_path=hub_path) if use_config else hub_path

    envs = factory.make_env(cfg, n_envs=2, use_async_envs=use_async_envs, trust_remote_code=True)
    env = next(iter(next(iter(envs.values())).values()))
    try:
        assert env.num_envs == 2
        env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
        assert isinstance(env, env_cls)
        observation, _ = env.reset(seed=123)
        assert observation.shape == (2, 4)
        _, rewards, _, _, _ = env.step(env.action_space.sample())
        assert rewards.shape == (2,)
    finally:
        env.close()


@pytest.mark.parametrize("config_parameter", ["positional_or_keyword", "keyword_only", "kwargs"])
def test_call_make_env_forwards_supported_config(config_parameter):
    cfg = HubEnvConfig(hub_path="test/configurable")

    if config_parameter == "kwargs":

        def make_env(n_envs, use_async_envs, **kwargs):
            return n_envs, use_async_envs, kwargs["cfg"]

    elif config_parameter == "keyword_only":

        def make_env(n_envs, use_async_envs, *, cfg):
            return n_envs, use_async_envs, cfg

    else:

        def make_env(n_envs, use_async_envs, cfg):
            return n_envs, use_async_envs, cfg

    result = _call_make_env(SimpleNamespace(make_env=make_env), n_envs=3, use_async_envs=True, cfg=cfg)
    assert result[:2] == (3, True)
    assert result[2] is cfg


def test_call_make_env_without_config_preserves_factory_default():
    default_config = object()

    def make_env(n_envs, use_async_envs, cfg=default_config):
        return cfg

    result = _call_make_env(SimpleNamespace(make_env=make_env), n_envs=1, use_async_envs=False, cfg=None)
    assert result is default_config


def test_call_make_env_does_not_pass_config_to_variadic_positional_parameter():
    def make_env(*cfg, n_envs, use_async_envs):
        return cfg, n_envs, use_async_envs

    result = _call_make_env(
        SimpleNamespace(make_env=make_env),
        n_envs=2,
        use_async_envs=False,
        cfg=HubEnvConfig(hub_path="test/variadic"),
    )
    assert result == ((), 2, False)


def test_call_make_env_preserves_positional_only_config_default():
    default_config = object()

    def make_env(cfg=default_config, /, *, n_envs, use_async_envs):
        return cfg

    result = _call_make_env(
        SimpleNamespace(make_env=make_env),
        n_envs=1,
        use_async_envs=False,
        cfg=HubEnvConfig(hub_path="test/positional"),
    )
    assert result is default_config


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_call_make_env_preserves_forwarding_without_inspectable_signature(error_type):
    class Factory:
        @property
        def __signature__(self):
            raise error_type("signature unavailable")

        def __call__(self, n_envs, use_async_envs, cfg):
            return cfg

    cfg = HubEnvConfig(hub_path="test/opaque")
    result = _call_make_env(SimpleNamespace(make_env=Factory()), n_envs=1, use_async_envs=False, cfg=cfg)
    assert result is cfg


@pytest.mark.parametrize("accepts_config", [False, True])
def test_call_make_env_preserves_factory_type_error(accepts_config):
    error = TypeError("invalid simulator configuration")
    calls = []

    if accepts_config:

        def make_env(n_envs, use_async_envs, cfg):
            calls.append(cfg)
            raise error

    else:

        def make_env(n_envs, use_async_envs):
            calls.append(None)
            raise error

    with pytest.raises(TypeError) as caught:
        _call_make_env(
            SimpleNamespace(make_env=make_env),
            n_envs=1,
            use_async_envs=False,
            cfg=HubEnvConfig(hub_path="test/broken"),
        )

    assert caught.value is error
    assert len(calls) == 1
