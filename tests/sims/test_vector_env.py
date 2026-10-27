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

"""Local vector environments built from a simulator integration."""

import os
import sys
import types

import draccus
import numpy as np
import pytest

from tests.sims.toy_env import ToyEnv, ToyEnvConfig


@pytest.mark.parametrize("where", ["local", "served"])
def test_env_config_creates_vector_envs_locally_or_served(where, request):
    from lerobot.env_server.client import RemoteVectorEnv
    from lerobot.sims.vector_env import VectorEnv

    config = ToyEnvConfig(endpoint=request.getfixturevalue("server")[1] if where == "served" else None)
    env = config.create_envs(n_envs=2)["reach"][0]
    try:
        assert isinstance(env, RemoteVectorEnv if where == "served" else VectorEnv)
        # Canonical observation names, as the server describes them.
        assert set(config.features) == {"observation.state", "observation.image", "action"}
        obs, _ = env.reset(seed=[1, 2])
        assert obs["observation.state"].shape == (2, 2)
        env.step(np.zeros((2, 2), np.float32))
    finally:
        env.close()


def test_environments_in_separate_processes_match_sequential_environments():
    from lerobot.sims.vector_env import VectorEnv

    results = []
    for use_async_envs in (False, True):
        env = VectorEnv(ToyEnvConfig().sim_config(), num_envs=3, use_async_envs=use_async_envs)
        try:
            env.reset(seed=[4, 5, 6])
            steps = [env.latest]
            for actions in ([[2, 0], [0.1, 0], [0, 0]], [[0.3, 0.1]] * 3):
                env.step(np.array(actions, np.float32))
                steps.append(env.latest)
            results.append(steps)
        finally:
            env.close()
    for sequential, parallel in zip(*results, strict=True):
        np.testing.assert_array_equal(parallel.obs["observation.state"], sequential.obs["observation.state"])
        np.testing.assert_array_equal(parallel.terminated, sequential.terminated)
        np.testing.assert_array_equal(parallel.elapsed_steps, sequential.elapsed_steps)
        assert parallel.task == sequential.task


def test_env_config_builds_each_simulator_once():
    # Some simulators (Isaac Sim) start only once per process: the environments built to read the
    # description are the ones used.
    before = ToyEnv.builds
    env = ToyEnvConfig().create_envs(n_envs=2)["reach"][0]
    try:
        env.reset(seed=[1, 2])
        env.step(np.zeros((2, 2), np.float32))
        assert ToyEnv.builds - before == 2
    finally:
        env.close()


def test_other_tasks_build_their_simulators_on_first_use():
    # lerobot-eval creates every task's environment up front, then prefetches the next one with _ensure.
    from lerobot.sims.vector_env import VectorEnv

    config = ToyEnvConfig()
    first = config.create_env(n_envs=2)
    env = VectorEnv(config.sim_config(), 2, descriptor=first.descriptor)
    try:
        assert not env._envs
        env.reset(seed=[1, 2])
        assert len(env._envs) == 2
    finally:
        first.close()
        env.close()


def test_batched_env_works_under_gymnasium_vector_wrappers():
    from gymnasium.wrappers.vector import RecordEpisodeStatistics

    env = RecordEpisodeStatistics(ToyEnvConfig().create_envs(n_envs=2)["reach"][0])
    try:
        env.reset(seed=[1, 2])
        env.step(np.zeros((2, 2), np.float32))
    finally:
        env.close()


def test_env_config_saves_only_its_settings():
    # Saved with a training config after its env is built; the env's description is runtime state.
    config = ToyEnvConfig()
    config.create_env().close()
    assert "descriptor" not in draccus.encode(config)


def test_episode_limit_is_found_through_wrappers(monkeypatch):
    import sys
    from types import ModuleType

    import gymnasium as gym

    from lerobot.sims.integrations.utils import canonical_observation
    from lerobot.sims.types import SimConfig
    from lerobot.sims.vector_env import VectorEnv

    class Reach(gym.Env):
        observation_space = gym.spaces.Dict({"agent_pos": gym.spaces.Box(-1, 1, (2,), np.float32)})
        action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

        def reset(self, seed=None, options=None):
            return {"agent_pos": np.zeros(2, np.float32)}, {}

        def step(self, action):
            return {"agent_pos": np.zeros(2, np.float32)}, 0.0, False, False, {}

    def create_envs(task, n_envs, env_cls, **kwargs):
        # TimeLimit sits under another wrapper, so it is not the outermost environment.
        return {task: {0: env_cls([lambda: gym.wrappers.ClipAction(gym.wrappers.TimeLimit(Reach(), 1000))])}}

    integration = ModuleType("lerobot.sims.integrations.wrapped")
    integration.create_envs = create_envs  # type: ignore[attr-defined]
    integration.canonical_observation = canonical_observation  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.sims.integrations.wrapped", integration)
    env = VectorEnv(SimConfig(type="wrapped", task="reach"))
    try:
        assert env.call("_max_episode_steps") == (1000,)
        assert env.descriptor.max_episode_steps == 1000
    finally:
        env.close()


class _CrashingEnv(ToyEnv):
    def reset(self, seed=None):
        os._exit(3)


def test_a_dead_environment_process_is_reported():
    # MuJoCo rendering can kill worker processes (macOS): say so instead of raising a bare EOFError.
    from lerobot.sims.vector_env import _ProcessEnv

    env = _ProcessEnv(_CrashingEnv)
    try:
        env.ready()
        with pytest.raises(RuntimeError, match="exited with code 3.*use_async_envs=false"):
            env.reset()
    finally:
        env.close()


def test_a_missing_simulator_says_how_to_get_it(monkeypatch):
    from lerobot.sims.types import SimConfig
    from lerobot.sims.vector_env import VectorEnv

    # LIBERO imports its simulator with the integration, and has an extra to install.
    monkeypatch.setitem(sys.modules, "libero", None)
    monkeypatch.delitem(sys.modules, "lerobot.sims.integrations.libero", raising=False)
    with pytest.raises(ModuleNotFoundError, match=r"pip install 'lerobot\[libero\]', or serve it"):
        SimConfig.resolve({"type": "libero", "task": "libero_10"})

    # Others import their simulator only when building an environment, possibly in a worker process.
    def build():
        raise ModuleNotFoundError("No module named 'some_simulator'", name="some_simulator")

    integration = types.ModuleType("lerobot.sims.integrations.some_sim")
    integration.create_envs = lambda task, n_envs, env_cls: {task: {0: env_cls([build] * n_envs)}}
    integration.canonical_observation = lambda obs: obs
    monkeypatch.setitem(sys.modules, integration.__name__, integration)
    expected = r"some_sim simulator is not installed \('some_simulator' is missing\): install it, or serve it"
    with pytest.raises(ModuleNotFoundError, match=expected):
        VectorEnv(SimConfig(type="some_sim", task="reach"))
    with pytest.raises(RuntimeError, match=expected):
        VectorEnv(SimConfig(type="some_sim", task="reach"), num_envs=2, use_async_envs=True)
