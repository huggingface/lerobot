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

"""Dependency and behavior boundaries for local simulation."""

import ast
import builtins
from pathlib import Path

import numpy as np

from tests.env_server.toy_env import ToyEnvConfig


def test_shared_simulation_code_names_no_simulator() -> None:
    root = Path(__file__).parents[2] / "src" / "lerobot"
    simulators = {path.stem for path in (root / "sims" / "integrations").glob("*.py")} - {"__init__", "utils"}
    shared = [
        *(root / "sims").glob("*.py"),
        *(root / "env_server").glob("*.py"),
        *(root / "robots" / "sim").glob("*.py"),
    ]
    named = [
        f"{path.relative_to(root)}: {node.value}"
        for path in shared
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Constant) and node.value in simulators
    ]
    assert named == []


def test_simulation_package_does_not_import_service_or_transport() -> None:
    sims = Path(__file__).parents[2] / "src" / "lerobot" / "sims"
    forbidden: list[str] = []
    for path in sims.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            forbidden.extend(
                f"{path.relative_to(sims)}: {module}"
                for module in modules
                if module.startswith(("lerobot.env_server", "lerobot.transport"))
            )
    assert forbidden == []


def test_local_environment_does_not_import_deployment_code(monkeypatch) -> None:
    original_import = builtins.__import__

    def import_without_deployment(name, *args, **kwargs):
        if name.startswith(("lerobot.env_server", "lerobot.transport")):
            raise AssertionError(f"Local environment imported deployment module {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_deployment)
    env = ToyEnvConfig().create_envs(n_envs=1)["reach"][0]
    try:
        obs, info = env.reset(seed=3)
        assert obs["observation.state"].shape == (1, 2)
        assert info["step"].tolist() == [0]

        terminal = env.step(np.array([[2.0, 0.0]], dtype=np.float32))
        assert terminal[1].tolist() == [1.0]
        assert terminal[2].tolist() == [True]
        terminal_obs = terminal[0]["observation.state"].copy()

        frozen = env.step(np.array([[-1.0, 0.0]], dtype=np.float32))
        np.testing.assert_array_equal(frozen[0]["observation.state"], terminal_obs)
        assert frozen[4]["step"].tolist() == [1]

        _, reset_info = env.reset(seed=3)
        assert reset_info["step"].tolist() == [0]
    finally:
        env.close()


def test_vector_env_delegates_native_observation_conventions(monkeypatch) -> None:
    import sys
    from types import ModuleType

    from gymnasium import spaces

    from lerobot.sims.types import SimConfig
    from lerobot.sims.vector_env import VectorEnv

    class NativeEnv:
        observation_space = spaces.Dict({"native_state": spaces.Box(-1, 1, (1,), np.float32)})
        action_space = spaces.Box(-1, 1, (1,), np.float32)
        task = "pick"
        task_description = "Pick the object"
        _max_episode_steps = 2

        def reset(self, seed=None):
            return {"native_state": np.array([0.25], dtype=np.float32)}, {}

        def step(self, action):
            return {"native_state": np.asarray(action, dtype=np.float32)}, 0.0, False, False, {}

        def close(self):
            pass

    adapted = []

    def canonical_observation(obs):
        adapted.append(obs)
        return {"observation.state": np.asarray(obs["native_state"], dtype=np.float32)}

    def create_envs(task, n_envs, env_cls, **kwargs):
        return {task: {0: env_cls([NativeEnv] * n_envs)}}

    # A new simulator is a module of these two functions; the shared environment never names it.
    integration = ModuleType("lerobot.sims.integrations.new_engine")
    integration.create_envs = create_envs  # type: ignore[attr-defined]
    integration.canonical_observation = canonical_observation  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.sims.integrations.new_engine", integration)
    env = VectorEnv(SimConfig(type="new_engine", task="pick", semantics="new-engine-v1"))
    try:
        assert env.single_observation_space["observation.state"].shape == (1,)
        obs, _ = env.reset(seed=[7])
        np.testing.assert_array_equal(obs["observation.state"], [[0.25]])
        assert len(adapted) == 2
    finally:
        env.close()


def test_envhub_rejects_custom_vector_behavior_without_unwrapping_it() -> None:
    import gymnasium as gym
    import pytest

    from lerobot.sims.integrations.envhub import _env_factories

    class CustomBatch(gym.vector.SyncVectorEnv):
        def step(self, actions):
            obs, reward, terminated, truncated, info = super().step(actions)
            return obs, reward + 50, terminated, truncated, info

    env = CustomBatch([lambda: gym.make("CartPole-v1")])
    with pytest.raises(ValueError, match="use_native_batch"):
        _env_factories(env, 1)
    assert env.closed


def test_envhub_native_batch_preserves_provider_and_options(monkeypatch) -> None:
    import gymnasium as gym
    import pytest

    from lerobot.envs import factory
    from lerobot.envs.configs import HubEnvConfig

    calls = []

    class Provider:
        @staticmethod
        def make_env(n_envs, use_async_envs, bonus):
            calls.append((n_envs, use_async_envs, bonus))
            return gym.wrappers.vector.TransformReward(
                gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")] * n_envs),
                lambda reward: reward + bonus,
            )

    monkeypatch.setattr(factory, "_download_hub_file", lambda *args: ("user/repo", "env.py", "unused", None))
    monkeypatch.setattr(factory, "_import_hub_module", lambda *args: Provider)
    config = HubEnvConfig(
        hub_path="user/repo", trust_remote_code=True, use_native_batch=True, make_env_kwargs={"bonus": 50}
    )
    env = next(iter(config.create_envs(2).values()))[0]
    try:
        env.reset(seed=2)
        assert env.step(np.zeros(2, dtype=np.int64))[1].tolist() == [51, 51]
        assert calls == [(2, False, 50)]
        assert config.descriptor is None
    finally:
        env.close()
    config.endpoint = "tcp/127.0.0.1:7448"
    with pytest.raises(ValueError, match="endpoint"):
        config.create_envs(1)
    config.endpoint = None
    with pytest.raises(ValueError, match="SimRobot"):
        config.create_env()
