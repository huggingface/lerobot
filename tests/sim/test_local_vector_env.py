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
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from lerobot.env_server.client import EnvClient, RemoteVectorEnv
from lerobot.envs import EnvClientConfig
from lerobot.sim.config import SimConfig
from lerobot.sim.integrations.utils import canonical_observation
from lerobot.sim.local_vector_env import LocalVectorEnv, _ProcessEnv
from tests.fixtures.toy_env import ToyEnv, ToyEnvConfig


def _register_integration(monkeypatch, name, env_fns, adapter=canonical_observation):
    """Register `lerobot.sim.integrations.<name>` for one test, as a simulator integration module."""
    integration = types.ModuleType(f"lerobot.sim.integrations.{name}")
    integration.env_fns = env_fns  # type: ignore[attr-defined]
    integration.canonical_observation = adapter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, integration.__name__, integration)


@pytest.mark.parametrize("where", ["local", "served"])
def test_env_config_creates_vector_envs_locally_or_served(where, request):
    config = ToyEnvConfig(
        server=EnvClientConfig(request.getfixturevalue("sim_server")[1]) if where == "served" else None
    )
    env = config.create_envs(n_envs=2)["reach"][0]
    try:
        assert isinstance(env, RemoteVectorEnv if where == "served" else LocalVectorEnv)
        # Canonical observation names, as the server describes them.
        assert set(config.features) == {"observation.state", "observation.image", "action"}
        obs, _ = env.reset(seed=[1, 2])
        assert obs["observation.state"].shape == (2, 2)
        env.step(np.zeros((2, 2), np.float32))
        assert env.call("task_description") == ("Move to the target",) * 2
        assert env.call("_max_episode_steps") == (5, 5)
    finally:
        env.close()


def test_a_finished_env_keeps_its_terminal_observation_with_zero_reward():
    env = ToyEnvConfig().create_envs(n_envs=1)["reach"][0]
    try:
        _, info = env.reset(seed=3)
        assert info["elapsed_steps"].tolist() == [0]
        terminal_obs, reward, terminated, _, _ = env.step(np.array([[2.0, 0.0]], np.float32))
        assert reward.tolist() == [1.0] and terminated.tolist() == [True]
        # Stepping again neither moves it nor rewards it.
        obs, reward, _, _, info = env.step(np.array([[-1.0, 0.0]], np.float32))
        np.testing.assert_array_equal(obs["observation.state"], terminal_obs["observation.state"])
        assert reward.tolist() == [0.0] and info["elapsed_steps"].tolist() == [1]
        _, info = env.reset(seed=3)
        assert info["elapsed_steps"].tolist() == [0]
    finally:
        env.close()


def test_every_run_mode_takes_the_same_steps():
    # Env 0 succeeds on the first step, env 1 on the third, and env 2 is truncated after five.
    actions = [[[2, 0], [0.1, 0], [0, 0]], [[0.3, 0.1], [0.3, 0.1], [0, 0]], [[1, 0], [1, 0], [0, 0]]]
    actions += [[[0, 0]] * 3] * 2
    per_env = ToyEnvConfig().sim_config()
    # A GPU simulator steps its whole batch, finished envs included; they must still look finished.
    batched = replace(per_env, type="toy_vector")
    runs = []
    for config, use_async_envs in ((per_env, False), (per_env, True), (batched, False), (batched, True)):
        env = LocalVectorEnv(config, num_envs=3, use_async_envs=use_async_envs)
        try:
            env.reset(seed=[4, 5, 6])
            steps = [env.latest]
            for batch in actions:
                env.step(np.array(batch, np.float32))
                steps.append(env.latest)
            runs.append((env.descriptor, env.call("_max_episode_steps"), steps))
        finally:
            env.close()
    descriptor, limits, expected_steps = runs[0]
    assert limits == (5, 5, 5)
    assert expected_steps[-1].terminated.tolist() == [True, True, False]
    assert expected_steps[-1].truncated.tolist() == [False, False, True]
    for other_descriptor, other_limits, steps in runs[1:]:
        assert other_descriptor.same_schema(descriptor) and other_limits == limits
        for expected, step in zip(expected_steps, steps, strict=True):
            for key in expected.obs:
                np.testing.assert_array_equal(step.obs[key], expected.obs[key])
            for field in ("reward", "terminated", "truncated", "is_success", "elapsed_steps"):
                np.testing.assert_array_equal(getattr(step, field), getattr(expected, field))
            assert step.task == expected.task


def test_a_batched_simulator_that_autoresets_is_rejected():
    config = replace(ToyEnvConfig().sim_config(), type="toy_vector", kwargs={"autoreset_mode": "NEXT_STEP"})
    with pytest.raises(ValueError, match="must not autoreset"):
        LocalVectorEnv(config, num_envs=2)


@pytest.mark.parametrize("where", ["local", "served"])
def test_misuse_is_rejected_locally_and_served(where, request):
    server = EnvClientConfig(request.getfixturevalue("sim_server")[1]) if where == "served" else None
    env = ToyEnvConfig(server=server).create_env()
    try:
        with pytest.raises(RuntimeError, match="Reset before stepping"):
            env.step(np.zeros((1, 2), np.float32))
        with pytest.raises(RuntimeError, match="Reset before reading"):
            env.call("task_description")
        with pytest.raises(TypeError, match="do not take arguments"):
            env.call("task", 0)
        with pytest.raises(ValueError, match="does not define reset options"):
            env.reset(options={"task_id": 1})
    finally:
        env.close()


def test_env_config_builds_each_simulator_once():
    # The environments built to read the description are the ones used, not built again.
    before = ToyEnv.builds
    env = ToyEnvConfig().create_envs(n_envs=2)["reach"][0]
    try:
        env.reset(seed=[1, 2])
        env.step(np.zeros((2, 2), np.float32))
        assert ToyEnv.builds - before == 2
    finally:
        env.close()


def test_a_named_task_builds_its_simulators_once():
    # A sim robot runs one named task: the environment built to read the description must be that task's.
    before = ToyEnv.builds
    env = ToyEnvConfig(task_ids=(0, 3)).create_env(task_id=3)
    try:
        env.reset(seed=1)
        assert env.task_id == 3 and ToyEnv.builds - before == 1
    finally:
        env.close()


def test_other_tasks_build_their_simulators_on_first_use():
    # lerobot-eval creates every task's environment up front; each builds its simulator on its first reset.
    config = ToyEnvConfig()
    first = config.create_env(n_envs=2)
    before = ToyEnv.builds
    env = LocalVectorEnv(config.sim_config(), 2, descriptor=first.descriptor)
    try:
        assert ToyEnv.builds == before
        env.reset(seed=[1, 2])
        assert ToyEnv.builds - before == 2
    finally:
        first.close()
        env.close()


def test_a_filtered_suite_starts_with_its_first_task():
    # LIBERO's --env.task_ids may leave out task 0.
    envs = ToyEnvConfig(task_ids=(3, 5)).create_envs(n_envs=1)["reach"]
    try:
        assert list(envs) == [3, 5] and envs[3].task_id == 3
    finally:
        for env in envs.values():
            env.close()
    env = ToyEnvConfig(task_ids=(3, 5)).create_env()
    try:
        assert env.task_id == 3
    finally:
        env.close()


def test_local_and_served_environments_agree_on_tasks():
    local = LocalVectorEnv(ToyEnvConfig(task_ids=(3, 5)).sim_config())
    try:
        # Nothing is opened before the first reset, so no server is needed.
        served = RemoteVectorEnv(EnvClient("tcp/127.0.0.1:7448"), descriptor=local.descriptor)
        assert local.task_id == served.task_id == 3
        # Both reject a task the filtered suite doesn't have.
        with pytest.raises(ValueError, match="Unknown task"):
            LocalVectorEnv(local.config, task_id=0, descriptor=local.descriptor)
        with pytest.raises(ValueError, match="Unknown task"):
            RemoteVectorEnv(EnvClient("tcp/127.0.0.1:7448"), task_id=0, descriptor=local.descriptor)
    finally:
        local.close()


def test_batched_env_works_under_gymnasium_vector_wrappers():
    env = RecordEpisodeStatistics(ToyEnvConfig().create_envs(n_envs=2)["reach"][0])
    try:
        env.reset(seed=[1, 2])
        env.step(np.zeros((2, 2), np.float32))
    finally:
        env.close()


class _Reach(gym.Env):
    observation_space = spaces.Dict({"agent_pos": spaces.Box(-1, 1, (2,), np.float32)})
    action_space = spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, seed=None, options=None):
        return {"agent_pos": np.zeros(2, np.float32)}, {}

    def step(self, action):
        return {"agent_pos": np.zeros(2, np.float32)}, 0.0, False, False, {}


def test_episode_limit_is_found_through_wrappers(monkeypatch):
    def env_fns(task, n_envs, **kwargs):
        # TimeLimit sits under another wrapper, so it is not the outermost environment.
        return {task: {0: [lambda: gym.wrappers.ClipAction(gym.wrappers.TimeLimit(_Reach(), 1000))]}}

    _register_integration(monkeypatch, "wrapped", env_fns)
    env = LocalVectorEnv(SimConfig(type="wrapped", task="reach"))
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
    env = _ProcessEnv(_CrashingEnv)
    try:
        env.ready()
        with pytest.raises(RuntimeError, match="exited with code 3.*use_async_envs=false"):
            env.reset()
    finally:
        env.close()


def test_a_missing_simulator_says_how_to_get_it(monkeypatch):
    # LIBERO imports its simulator with the integration, and has an extra to install. Hide all of it: an
    # earlier test may have imported its submodules.
    for name in [name for name in sys.modules if name.startswith("libero.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "libero", None)
    monkeypatch.delitem(sys.modules, "lerobot.sim.integrations.libero", raising=False)
    with pytest.raises(ModuleNotFoundError, match=r"pip install 'lerobot\[libero\]', or serve it"):
        SimConfig.resolve({"type": "libero", "task": "libero_10"})

    # Others import their simulator only when building an environment, possibly in a worker process.
    def build():
        raise ModuleNotFoundError("No module named 'some_simulator'", name="some_simulator")

    _register_integration(
        monkeypatch, "some_sim", lambda task, n_envs: {task: {0: [build] * n_envs}}, lambda obs: obs
    )
    expected = r"some_sim simulator is not installed \('some_simulator' is missing\): install it, or serve it"
    with pytest.raises(ModuleNotFoundError, match=expected):
        LocalVectorEnv(SimConfig(type="some_sim", task="reach"))
    with pytest.raises(RuntimeError, match=expected):
        LocalVectorEnv(SimConfig(type="some_sim", task="reach"), num_envs=2, use_async_envs=True)


class _NativeEnv:
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


def test_a_new_simulator_needs_only_its_factories_and_observation_adapter(monkeypatch):
    adapted = []

    def adapter(obs):
        adapted.append(obs)
        return {"observation.state": np.asarray(obs["native_state"], dtype=np.float32)}

    # The shared environment never names the simulator: it only calls the integration's two functions.
    _register_integration(
        monkeypatch, "new_engine", lambda task, n_envs: {task: {0: [_NativeEnv] * n_envs}}, adapter
    )
    env = LocalVectorEnv(SimConfig(type="new_engine", task="pick", semantics="new-engine-v1"))
    try:
        assert env.single_observation_space["observation.state"].shape == (1,)
        obs, _ = env.reset(seed=[7])
        np.testing.assert_array_equal(obs["observation.state"], [[0.25]])
        # Once to describe the observations, once for the reset.
        assert len(adapted) == 2
    finally:
        env.close()
