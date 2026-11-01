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

import draccus
import numpy as np
import pytest

from lerobot.envs.configs import EnvServerConfig
from tests.fixtures.toy_env import ToyEnv, ToyEnvConfig


@pytest.mark.parametrize("where", ["local", "served"])
def test_env_config_creates_vector_envs_locally_or_served(where, request):
    from lerobot.env_server.client import RemoteVectorEnv
    from lerobot.sim.vector_env import LocalVectorEnv

    config = ToyEnvConfig(
        server=EnvServerConfig(request.getfixturevalue("sim_server")[1]) if where == "served" else None
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


def test_environments_in_separate_processes_match_sequential_environments():
    from lerobot.sim.vector_env import LocalVectorEnv

    results = []
    for use_async_envs in (False, True):
        env = LocalVectorEnv(ToyEnvConfig().sim_config(), num_envs=3, use_async_envs=use_async_envs)
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


@pytest.mark.parametrize("use_async_envs", [False, True])
def test_a_batched_simulator_runs_like_one_env_per_batch_entry(use_async_envs):
    # A GPU simulator steps its whole batch, finished envs included; they must still look finished.
    from lerobot.sim.vector_env import LocalVectorEnv

    toy = ToyEnvConfig().sim_config()
    # Env 0 succeeds on the first step, env 1 on the third, and env 2 is truncated after five.
    actions = [[[2, 0], [0.1, 0], [0, 0]], [[0.3, 0.1], [0.3, 0.1], [0, 0]], [[1, 0], [1, 0], [0, 0]]]
    actions += [[[0, 0]] * 3] * 2
    runs = []
    for config, in_process in ((toy, True), (replace(toy, type="toy_vector"), not use_async_envs)):
        env = LocalVectorEnv(config, num_envs=3, use_async_envs=not in_process)
        try:
            env.reset(seed=[4, 5, 6])
            steps = [env.latest]
            for batch in actions:
                env.step(np.array(batch, np.float32))
                steps.append(env.latest)
            runs.append((env.descriptor, env.call("_max_episode_steps"), steps))
        finally:
            env.close()
    (per_env, per_env_limits, per_env_steps), (batched, batched_limits, batched_steps) = runs
    assert batched.same_schema(per_env) and batched_limits == per_env_limits == (5, 5, 5)
    for expected, step in zip(per_env_steps, batched_steps, strict=True):
        for key in expected.obs:
            np.testing.assert_array_equal(step.obs[key], expected.obs[key])
        for field in ("reward", "terminated", "truncated", "is_success", "elapsed_steps"):
            np.testing.assert_array_equal(getattr(step, field), getattr(expected, field))
        assert step.task == expected.task
    assert batched_steps[-1].terminated.tolist() == [True, True, False]
    assert batched_steps[-1].truncated.tolist() == [False, False, True]


def test_a_batched_simulator_that_autoresets_is_rejected():
    from lerobot.sim.vector_env import LocalVectorEnv

    config = replace(ToyEnvConfig().sim_config(), type="toy_vector", kwargs={"autoreset_mode": "NEXT_STEP"})
    with pytest.raises(ValueError, match="must not autoreset"):
        LocalVectorEnv(config, num_envs=2)


@pytest.mark.parametrize("where", ["local", "served"])
def test_reset_options_are_rejected_locally_and_served(where, request):
    server = EnvServerConfig(request.getfixturevalue("sim_server")[1]) if where == "served" else None
    env = ToyEnvConfig(server=server).create_env()
    try:
        with pytest.raises(ValueError, match="does not define reset options"):
            env.reset(options={"task_id": 1})
    finally:
        env.close()


@pytest.mark.parametrize("where", ["local", "served"])
def test_stepping_or_reading_before_reset_fails_locally_and_served(where, request):
    server = EnvServerConfig(request.getfixturevalue("sim_server")[1]) if where == "served" else None
    env = ToyEnvConfig(server=server).create_env()
    try:
        with pytest.raises(RuntimeError, match="Reset before stepping"):
            env.step(np.zeros((1, 2), np.float32))
        with pytest.raises(RuntimeError, match="Reset before reading"):
            env.call("task_description")
    finally:
        env.close()


@pytest.mark.parametrize("where", ["local", "served"])
def test_attributes_take_no_arguments_locally_and_served(where, request):
    server = EnvServerConfig(request.getfixturevalue("sim_server")[1]) if where == "served" else None
    env = ToyEnvConfig(server=server).create_env()
    try:
        with pytest.raises(TypeError, match="do not take arguments"):
            env.call("task", 0)
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


def test_a_filtered_suite_starts_with_its_first_task():
    # LIBERO's --env.task_ids may leave out task 0.
    envs = ToyEnvConfig(task_ids=(3, 5)).create_envs(n_envs=1)["reach"]
    try:
        assert list(envs) == [3, 5] and envs[3].task_id == 3
    finally:
        for env in envs.values():
            env.close()


def test_local_and_served_environments_default_to_the_same_task():
    from lerobot.env_server.client import EnvClient, RemoteVectorEnv
    from lerobot.sim.vector_env import LocalVectorEnv

    local = LocalVectorEnv(ToyEnvConfig(task_ids=(3, 5)).sim_config())
    try:
        # Nothing is opened before the first reset, so no server is needed.
        served = RemoteVectorEnv(EnvClient("tcp/127.0.0.1:7448"), descriptor=local.descriptor)
        assert local.task_id == served.task_id == 3
    finally:
        local.close()


def test_other_tasks_build_their_simulators_on_first_use():
    # lerobot-eval creates every task's environment up front; each builds its simulator on its first reset.
    from lerobot.sim.vector_env import LocalVectorEnv

    config = ToyEnvConfig()
    first = config.create_env(n_envs=2)
    env = LocalVectorEnv(config.sim_config(), 2, descriptor=first.descriptor)
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


def test_env_server_options_are_one_optional_group():
    assert ToyEnvConfig().server is None and draccus.encode(ToyEnvConfig())["server"] is None
    config = draccus.parse(
        ToyEnvConfig, args=["--server.endpoint=tcp/127.0.0.1:7448", "--server.encoding=jpeg"]
    )
    assert (config.server.endpoint, config.server.encoding) == ("tcp/127.0.0.1:7448", "jpeg")
    with pytest.raises(draccus.utils.DecodingError, match="endpoint"):
        draccus.parse(ToyEnvConfig, args=["--server.encoding=jpeg"])


def test_env_config_saves_only_its_settings():
    # Saved with a training config after its env is built; the env's description is runtime state.
    config = ToyEnvConfig()
    config.create_env().close()
    assert "descriptor" not in draccus.encode(config)


def test_episode_limit_is_found_through_wrappers(monkeypatch):
    import sys
    from types import ModuleType

    import gymnasium as gym

    from lerobot.sim.config import SimConfig
    from lerobot.sim.integrations.utils import canonical_observation
    from lerobot.sim.vector_env import LocalVectorEnv

    class Reach(gym.Env):
        observation_space = gym.spaces.Dict({"agent_pos": gym.spaces.Box(-1, 1, (2,), np.float32)})
        action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

        def reset(self, seed=None, options=None):
            return {"agent_pos": np.zeros(2, np.float32)}, {}

        def step(self, action):
            return {"agent_pos": np.zeros(2, np.float32)}, 0.0, False, False, {}

    def env_fns(task, n_envs, **kwargs):
        # TimeLimit sits under another wrapper, so it is not the outermost environment.
        return {task: {0: [lambda: gym.wrappers.ClipAction(gym.wrappers.TimeLimit(Reach(), 1000))]}}

    integration = ModuleType("lerobot.sim.integrations.wrapped")
    integration.env_fns = env_fns  # type: ignore[attr-defined]
    integration.canonical_observation = canonical_observation  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.sim.integrations.wrapped", integration)
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
    from lerobot.sim.vector_env import _ProcessEnv

    env = _ProcessEnv(_CrashingEnv)
    try:
        env.ready()
        with pytest.raises(RuntimeError, match="exited with code 3.*use_async_envs=false"):
            env.reset()
    finally:
        env.close()


def test_a_missing_simulator_says_how_to_get_it(monkeypatch):
    from lerobot.sim.config import SimConfig
    from lerobot.sim.vector_env import LocalVectorEnv

    # LIBERO imports its simulator with the integration, and has an extra to install.
    monkeypatch.setitem(sys.modules, "libero", None)
    monkeypatch.delitem(sys.modules, "lerobot.sim.integrations.libero", raising=False)
    with pytest.raises(ModuleNotFoundError, match=r"pip install 'lerobot\[libero\]', or serve it"):
        SimConfig.resolve({"type": "libero", "task": "libero_10"})

    # Others import their simulator only when building an environment, possibly in a worker process.
    def build():
        raise ModuleNotFoundError("No module named 'some_simulator'", name="some_simulator")

    integration = types.ModuleType("lerobot.sim.integrations.some_sim")
    integration.env_fns = lambda task, n_envs: {task: {0: [build] * n_envs}}
    integration.canonical_observation = lambda obs: obs
    monkeypatch.setitem(sys.modules, integration.__name__, integration)
    expected = r"some_sim simulator is not installed \('some_simulator' is missing\): install it, or serve it"
    with pytest.raises(ModuleNotFoundError, match=expected):
        LocalVectorEnv(SimConfig(type="some_sim", task="reach"))
    with pytest.raises(RuntimeError, match=expected):
        LocalVectorEnv(SimConfig(type="some_sim", task="reach"), num_envs=2, use_async_envs=True)


def test_vector_env_delegates_native_observation_conventions(monkeypatch) -> None:
    import sys
    from types import ModuleType

    from gymnasium import spaces

    from lerobot.sim.config import SimConfig
    from lerobot.sim.vector_env import LocalVectorEnv

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

    def env_fns(task, n_envs, **kwargs):
        return {task: {0: [NativeEnv] * n_envs}}

    # A new simulator is a module of these two functions; the shared environment never names it.
    integration = ModuleType("lerobot.sim.integrations.new_engine")
    integration.env_fns = env_fns  # type: ignore[attr-defined]
    integration.canonical_observation = canonical_observation  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lerobot.sim.integrations.new_engine", integration)
    env = LocalVectorEnv(SimConfig(type="new_engine", task="pick", semantics="new-engine-v1"))
    try:
        assert env.single_observation_space["observation.state"].shape == (1,)
        obs, _ = env.reset(seed=[7])
        np.testing.assert_array_equal(obs["observation.state"], [[0.25]])
        assert len(adapted) == 2
    finally:
        env.close()
