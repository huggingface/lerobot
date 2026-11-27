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

"""Environments built from an EnvHub repository's `env.py`."""

import builtins
import contextlib
import logging
import threading
import warnings
from collections.abc import Iterator
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from lerobot.envs import EnvClientConfig, HubEnvConfig
from lerobot.envs.factory import make_env, make_task_env
from lerobot.sim import LocalVectorEnv, SimConfig, SimSettings
from lerobot.sim.integrations import envhub

_ENVHUB_ENV = """
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Reach(gym.Env):
    observation_space = spaces.Dict(
        {
            "pixels": spaces.Dict({"front": spaces.Box(0, 255, (8, 8, 3), np.uint8)}),
            "agent_pos": spaces.Box(-10, 10, (2,), np.float32),
        }
    )
    action_space = spaces.Box(-1, 1, (2,), np.float32)
    task_description = "reach the goal"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.np_random.uniform(-1, 1, 2).astype(np.float32)
        return self._obs(), {}

    def step(self, action):
        self.pos += action
        done = bool(np.linalg.norm(self.pos) > 3)
        return self._obs(), float(done), done, False, {"is_success": done}

    def _obs(self):
        return {"pixels": {"front": np.zeros((8, 8, 3), np.uint8)}, "agent_pos": self.pos.copy()}


def make_env(n_envs=1, use_async_envs=False):
    return {"reach_suite": {0: gym.vector.SyncVectorEnv([Reach] * n_envs)}}
"""


@pytest.fixture
def hub_repo(tmp_path, monkeypatch):
    """Return a function that sets the `env.py` the Hub repository `user/reach` downloads."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    # Only the network download is stubbed; the consent check and URL parsing run for real.
    monkeypatch.setattr(envhub, "snapshot_download", lambda **kwargs: str(snapshot))
    return (snapshot / "env.py").write_text


def _envhub(hub_path: str = "user/reach", trust_remote_code: bool = True) -> SimSettings:
    return SimSettings(type="envhub", kwargs={"hub_path": hub_path, "trust_remote_code": trust_remote_code})


def test_envhub_runs_a_hub_repository(hub_repo):
    hub_repo(_ENVHUB_ENV)
    env = LocalVectorEnv(_envhub(), num_envs=2)
    try:
        assert env.tasks == {"reach_suite": [0]}
        assert {f.name: (f.shape, f.kind) for f in env.descriptor.features} == {
            "observation.images.front": ((8, 8, 3), "rgb"),
            "observation.state": ((2,), "tensor"),
        }
        assert env.descriptor.action_feature.shape == (2,)
        first = env.reset(seed=[3, 4])[0]["observation.state"]
        np.testing.assert_array_equal(env.reset(seed=[3, 4])[0]["observation.state"], first)
        info = env.step(np.full((2, 2), 0.5, np.float32))[4]
        assert info["task"] == ("reach the goal", "reach the goal")
    finally:
        env.close()
    # Running remote code needs explicit consent.
    with pytest.raises(RuntimeError, match="Refusing"):
        LocalVectorEnv(_envhub(trust_remote_code=False), num_envs=1)


def test_envhub_make_env_takes_the_options_of_the_command(hub_repo):
    # The repository's make_env sizes its camera from an option of its own.
    hub_repo(
        _ENVHUB_ENV
        + """
class SizedReach(Reach):
    def __init__(self, size):
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict({"front": spaces.Box(0, 255, (size, size, 3), np.uint8)}),
                "agent_pos": spaces.Box(-10, 10, (2,), np.float32),
            }
        )


def make_env(n_envs=1, use_async_envs=False, size=8):
    return {"reach_suite": {0: gym.vector.SyncVectorEnv([lambda: SizedReach(size)] * n_envs)}}
"""
    )
    config = HubEnvConfig(hub_path="user/reach", trust_remote_code=True, make_env_kwargs={"size": 4})
    env = LocalVectorEnv(SimSettings(**config.sim_settings()))
    try:
        shapes = {f.name: f.shape for f in env.descriptor.features}
        assert shapes["observation.images.front"] == (4, 4, 3)
    finally:
        env.close()


def test_envhub_repository_files_sit_next_to_env_py(hub_repo, tmp_path):
    # The whole repository is downloaded, so env.py finds its other files as in a local directory.
    (tmp_path / "snapshot" / "task.txt").write_text("Reach the target.")
    hub_repo(
        'from pathlib import Path\nTASK = (Path(__file__).parent / "task.txt").read_text()\n' + _ENVHUB_ENV
    )
    env = LocalVectorEnv(_envhub())
    try:
        assert env.tasks == {"reach_suite": [0]}
    finally:
        env.close()


def test_envhub_names_a_file_missing_from_the_repository(tmp_path, monkeypatch):
    downloads = []

    def download(**kwargs):
        downloads.append((kwargs["repo_id"], kwargs["repo_type"], kwargs["revision"]))
        return str(tmp_path)

    monkeypatch.setattr(envhub, "snapshot_download", download)
    with pytest.raises(FileNotFoundError, match="custom.py in repository user/reach@v1"):
        LocalVectorEnv(_envhub(hub_path="user/reach@v1:custom.py"))
    assert downloads == [("user/reach", "space", "v1")]


def test_envhub_runs_a_local_directory(tmp_path, monkeypatch):
    (tmp_path / "env.py").write_text(_ENVHUB_ENV)

    def no_download(**kwargs):
        raise AssertionError("A local directory is not downloaded")

    monkeypatch.setattr(envhub, "snapshot_download", no_download)
    env = LocalVectorEnv(_envhub(hub_path=str(tmp_path)))
    try:
        assert env.tasks == {"reach_suite": [0]}
    finally:
        env.close()
    # A local directory needs the same consent as a Hub repository.
    with pytest.raises(RuntimeError, match="Refusing to run the EnvHub environment"):
        LocalVectorEnv(_envhub(hub_path=str(tmp_path), trust_remote_code=False))


def test_envhub_module_runs_once_per_server_process(hub_repo, monkeypatch):
    # A simulator app that can start only once per process (e.g. Isaac Sim) lives in module state.
    hub_repo("import builtins\nbuiltins.envhub_imports += 1\n" + _ENVHUB_ENV)
    monkeypatch.setattr(builtins, "envhub_imports", 0, raising=False)
    # The probe and two sessions each build their environments.
    for _ in range(3):
        env = LocalVectorEnv(_envhub())
        env.reset(seed=[0])
        env.close()
    assert builtins.envhub_imports == 1


def test_envhub_needs_dict_observations(hub_repo):
    # A flat array observation, which local LeRobot execution does not accept either.
    hub_repo(
        "import gymnasium as gym\n"
        "def make_env(n_envs=1, use_async_envs=False):\n"
        "    return gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1')] * n_envs)\n"
    )
    with pytest.raises(ValueError, match="dict observations"):
        LocalVectorEnv(_envhub())


def test_envhub_settings_come_from_the_repository(hub_repo):
    declared = (
        "def sim_settings(task, **options):\n"
        "    return {'fps': 15, 'control_mode': 'delta', 'semantics': 'reach-delta-v1', 'state_names': ['x', 'y'],"
        " 'action_names': ['dx', 'dy'], 'robot_type': 'so_follower'}\n"
    )
    sim = _envhub()
    # Repository values fill the settings left out; an explicit value wins over them.
    hub_repo(_ENVHUB_ENV + declared)
    resolved = SimConfig.resolve(replace(sim, fps=50), "reach_suite")
    assert (resolved.fps, resolved.control_mode, resolved.semantics) == (50, "delta", "reach-delta-v1")
    assert (resolved.state_names, resolved.action_names) == (("x", "y"), ("dx", "dy"))
    assert resolved.robot_type == "so_follower"
    # Repositories without sim_settings keep working with the defaults.
    hub_repo(_ENVHUB_ENV)
    resolved = SimConfig.resolve(sim, "reach_suite")
    assert (resolved.fps, resolved.control_mode, resolved.semantics) == (30, "position", "envhub-v1")
    # A typo or an old key is an error naming the closest valid key, not an ignored setting.
    hub_repo(
        _ENVHUB_ENV
        + "def sim_settings(task):\n    return {'fsp': 30, 'control': 'delta', 'gripper_indices': [6]}\n"
    )
    with pytest.raises(ValueError) as error:
        SimConfig.resolve(sim, "reach_suite")
    message = str(error.value)
    assert "sim_settings has unknown keys" in message
    assert "'control' (did you mean 'control_mode'?)" in message
    assert "'fsp' (did you mean 'fps'?)" in message
    assert "'gripper_indices'; expected" in message
    hub_repo(_ENVHUB_ENV + "def sim_settings(task):\n    return [('fps', 15)]\n")
    with pytest.raises(TypeError, match="must return a dict"):
        SimConfig.resolve(sim, "reach_suite")


def test_envhub_still_reads_lerobot_sim_with_a_warning(hub_repo):
    # Repositories written for LeRobot 0.7 set a LEROBOT_SIM dict.
    sim = _envhub()
    hub_repo(_ENVHUB_ENV + "LEROBOT_SIM = {'fps': 15}\n")
    with pytest.warns(FutureWarning, match=r"define sim_settings\(task, \*\*make_env_kwargs\)"):
        assert SimConfig.resolve(sim, "reach_suite").fps == 15
    # A repository keeping LEROBOT_SIM for older LeRobot versions next to sim_settings gets no warning.
    hub_repo(_ENVHUB_ENV + "LEROBOT_SIM = {'fps': 15}\n\ndef sim_settings(task):\n    return {'fps': 20}\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert SimConfig.resolve(sim, "reach_suite").fps == 20


def test_envhub_rejects_custom_vector_behavior_without_unwrapping_it():
    class CustomBatch(gym.vector.SyncVectorEnv):
        def step(self, actions):
            obs, reward, terminated, truncated, info = super().step(actions)
            return obs, reward + 50, terminated, truncated, info

    env = CustomBatch([lambda: gym.make("CartPole-v1")])
    with pytest.raises(ValueError, match="use_native_batch"):
        envhub._env_factories(env, 1)
    assert env.closed


_NATIVE_BATCH_ENV = """
import builtins

import gymnasium as gym

builtins.envhub_imports += 1


def make_env(n_envs, use_async_envs, bonus):
    env = gym.wrappers.vector.TransformReward(
        gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")] * n_envs), lambda reward: reward + bonus
    )
    env.made_with = (n_envs, use_async_envs, bonus)
    return env
"""


def test_envhub_native_batch_preserves_provider_and_options(hub_repo, monkeypatch):
    hub_repo(_NATIVE_BATCH_ENV)
    monkeypatch.setattr(builtins, "envhub_imports", 0, raising=False)
    config = HubEnvConfig(
        hub_path="user/repo", trust_remote_code=True, use_native_batch=True, make_env_kwargs={"bonus": 50}
    )
    for _ in range(2):
        env = next(iter(make_env(config, 2).values()))[0]
        try:
            env.reset(seed=2)
            assert env.step(np.zeros(2, dtype=np.int64))[1].tolist() == [51, 51]
            assert env.made_with == (2, False, 50)
            assert not hasattr(env, "descriptor")
        finally:
            env.close()
    # Like the shared path, the native batch imports the repo's env.py once per process.
    assert builtins.envhub_imports == 1
    config.server = EnvClientConfig("tcp/127.0.0.1:7448")
    with pytest.raises(ValueError, match="env server"):
        make_env(config, 1)
    config.server = None
    with pytest.raises(ValueError, match="SimRobot"):
        make_task_env(config)


_BATCHED_ENV = """
import builtins
import threading
from functools import partial

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

# Module state that can't be pickled, like a running simulator's.
_SIMULATOR = threading.Lock()


class ReachBatch(gym.vector.VectorEnv):
    # One simulator stepping every env at once, as a GPU simulator does.
    metadata = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}

    def __init__(self, n_envs, goal):
        with _SIMULATOR:
            getattr(builtins, "envhub_builds", []).append(goal)
        self.num_envs, self.goal = n_envs, goal
        self.task_description = f"reach {goal}"
        self._max_episode_steps = 5
        self.single_observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict({"front": spaces.Box(0, 255, (8, 8, 3), np.uint8)}),
                "agent_pos": spaces.Box(-10, 10, (2,), np.float32),
            }
        )
        self.single_action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, n_envs)
        self.action_space = batch_space(self.single_action_space, n_envs)

    def reset(self, seed=None, options=None):
        self.pos = np.array([np.random.default_rng(s).uniform(-1, 1, 2) for s in seed], np.float32)
        return self._obs(), {}

    def step(self, actions):
        self.pos += actions
        done = np.abs(self.pos - self.goal).max(axis=1) < 0.5
        return self._obs(), done.astype(np.float32), done, np.zeros(self.num_envs, bool), {"is_success": done}

    def _obs(self):
        return {"pixels": {"front": np.zeros((self.num_envs, 8, 8, 3), np.uint8)}, "agent_pos": self.pos.copy()}


def vector_env_fns(task, n_envs, spacing=1.0):
    return {"reach_batch": {i: partial(ReachBatch, n_envs, i * spacing) for i in range(3)}}
"""


def test_envhub_runs_a_batched_repository_one_task_at_a_time(hub_repo, monkeypatch):
    hub_repo(_BATCHED_ENV)
    monkeypatch.setattr(builtins, "envhub_builds", [], raising=False)
    config = HubEnvConfig(hub_path="user/reach", trust_remote_code=True, make_env_kwargs={"spacing": 2.0})
    envs = make_env(config, 2)["reach_batch"]
    try:
        assert sorted(envs) == [0, 1, 2]
        # One simulator for the batch of 2, and only the first task's: the others build when they run.
        assert builtins.envhub_builds == [0.0]
        env = envs[2]
        first = env.reset(seed=[3, 4])[0]["observation.state"]
        assert first.shape == (2, 2)
        assert builtins.envhub_builds == [0.0, 4.0]
        np.testing.assert_array_equal(env.reset(seed=[3, 4])[0]["observation.state"], first)
        info = env.step(np.zeros((2, 2), np.float32))[4]
        assert info["task"] == ("reach 4.0", "reach 4.0")
        assert env.call("_max_episode_steps") == (5, 5)
    finally:
        for env in envs.values():
            env.close()


def test_envhub_batched_repository_receives_the_requested_task(hub_repo, monkeypatch):
    # A repo whose suites are expensive to list builds only the one asked for, as served with --sim.task.
    hub_repo(
        _BATCHED_ENV
        + '\n\ndef vector_env_fns(task, n_envs, spacing=1.0):\n    return {task or "reach_batch": {0: partial(ReachBatch, n_envs, 0.0)}}\n'
    )
    monkeypatch.setattr(builtins, "envhub_builds", [], raising=False)
    envs = make_env(HubEnvConfig(hub_path="user/reach", trust_remote_code=True, task="push"), 2)
    try:
        assert list(envs) == ["push"]
    finally:
        envs["push"][0].close()


def test_envhub_builds_only_the_requested_suite(hub_repo, monkeypatch):
    # Each suite's simulator is expensive (e.g. a GPU world): the env is described from the one requested.
    hub_repo(
        _BATCHED_ENV
        + '\n\ndef vector_env_fns(task, n_envs, spacing=1.0):\n    return {"near": {0: partial(ReachBatch, n_envs, 0.0)}, "far": {0: partial(ReachBatch, n_envs, 5.0)}}\n'
    )
    monkeypatch.setattr(builtins, "envhub_builds", [], raising=False)
    envs = make_env(HubEnvConfig(hub_path="user/reach", trust_remote_code=True, task="far"), 2)
    try:
        assert list(envs) == ["far"]
        assert builtins.envhub_builds == [5.0]
    finally:
        envs["far"][0].close()
    # Several suites, comma-separated: described from the first, each other one built when it runs.
    builtins.envhub_builds.clear()
    envs = make_env(HubEnvConfig(hub_path="user/reach", trust_remote_code=True, task="far, near"), 2)
    try:
        assert list(envs) == ["far", "near"]
        assert builtins.envhub_builds == [5.0]
        envs["near"][0].reset(seed=[3, 4])
        assert builtins.envhub_builds == [5.0, 0.0]
    finally:
        for suite in envs.values():
            suite[0].close()
    with pytest.raises(ValueError, match=r"Unknown task 'other'; this environment has \['far', 'near'\]"):
        make_env(HubEnvConfig(hub_path="user/reach", trust_remote_code=True, task="other"), 2)


_TWO_SUITES = (
    _BATCHED_ENV
    + """

class WideReach(ReachBatch):
    # A suite whose robot reports one more state component.
    def __init__(self, n_envs, goal):
        super().__init__(n_envs, goal)
        self.single_observation_space["agent_pos"] = spaces.Box(-10, 10, (3,), np.float32)

    def _obs(self):
        obs = super()._obs()
        obs["agent_pos"] = np.concatenate([obs["agent_pos"], np.zeros((self.num_envs, 1), np.float32)], axis=1)
        return obs


def vector_env_fns(task, n_envs, wide=False):
    return {"near": {0: partial(ReachBatch, n_envs, 0.0)}, "far": {0: partial(WideReach if wide else ReachBatch, n_envs, 5.0)}}
"""
)


def test_envhub_settings_differ_per_suite(hub_repo, monkeypatch):
    # One repository whose suites run at different rates; sim_settings only ever gets one suite's name.
    hub_repo(
        _TWO_SUITES
        + "\n\ndef sim_settings(task, wide=False):\n    return {'fps': {'near': 50, 'far': 15}[task]}\n"
    )
    monkeypatch.setattr(builtins, "envhub_builds", [], raising=False)
    # Both suites requested, or every suite of the repository.
    for task in ("far,near", ""):
        envs = make_env(HubEnvConfig(hub_path="user/reach", trust_remote_code=True, task=task), 2)
        try:
            assert envs["far"][0].metadata["render_fps"] == 15 and envs["far"][0].descriptor.fps == 15
            assert envs["near"][0].metadata["render_fps"] == 50 and envs["near"][0].descriptor.fps == 50
        finally:
            for suite in envs.values():
                suite[0].close()
    # Given the settings, as a server configured with every suite has them, an env resolves its own suite.
    served = SimSettings(type="envhub", kwargs={"hub_path": "user/reach", "trust_remote_code": True})
    env = LocalVectorEnv(served, 2, "far")
    try:
        assert env.descriptor.fps == 15
    finally:
        env.close()


@contextlib.contextmanager
def _serve_every_suite(**make_env_kwargs) -> Iterator[str]:
    """Serve the repository `user/reach` with every suite, as `lerobot-env-server --envhub=user/reach` does."""
    pytest.importorskip("zenoh")
    pytest.importorskip("msgpack")
    from lerobot.env_server import EnvServer, EnvServerConfig
    from lerobot.transport.zenoh import ZenohConfig
    from tests.fixtures.sim_server import free_port

    endpoint = f"tcp/127.0.0.1:{free_port()}"
    kwargs = {"hub_path": "user/reach", "trust_remote_code": True, "make_env_kwargs": make_env_kwargs}
    server = EnvServer(
        EnvServerConfig(
            sim=SimSettings(type="envhub", task="", kwargs=kwargs),
            deployment="envhub",
            zenoh=ZenohConfig(listen_endpoints=[endpoint]),
        )
    )
    thread = threading.Thread(target=server.serve)
    thread.start()
    try:
        assert server.ready.wait(5)
        yield endpoint
    finally:
        server.stop()
        thread.join(5)
    assert not thread.is_alive()


def test_a_server_with_every_suite_gives_each_session_its_suite_settings(hub_repo, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    hub_repo(
        _TWO_SUITES
        + "\n\ndef sim_settings(task, wide=False):\n"
        + "    return {'fps': {'near': 50, 'far': 15}[task], 'robot_type': {'near': 'panda', 'far': 'sawyer'}[task]}\n"
    )
    monkeypatch.setattr(builtins, "envhub_builds", [], raising=False)
    with _serve_every_suite() as endpoint:
        # Without declared features, the first suite is built to describe the env.
        assert builtins.envhub_builds == [0.0]
        assert "suite 'near' has no declared features" in caplog.text
        for suite, fps in (("far", 15), ("near", 50)):
            env = make_env(HubEnvConfig(task=suite, server=EnvClientConfig(endpoint)), 2)[suite][0]
            try:
                # Described with its own settings before it is built, as a sim robot reads them.
                assert env.descriptor.fps == env.metadata["render_fps"] == fps
                assert env.descriptor.robot_type == {"far": "sawyer", "near": "panda"}[suite]
                env.reset(seed=[3, 4])
                assert env.metadata["render_fps"] == fps
            finally:
                env.close()


def test_a_server_serves_suites_that_differ_to_different_runs(hub_repo, monkeypatch):
    # A wide arm suite and a narrow one, each declaring its own observations.
    hub_repo(
        _TWO_SUITES
        + "\n\ndef sim_settings(task, wide=False):\n"
        + "    state = 3 if task == 'far' and wide else 2\n"
        + "    return {'features': {'observation.images.front': (8, 8, 3), 'observation.state': (state,), 'action': (2,)}}\n"
    )
    monkeypatch.setattr(builtins, "envhub_builds", [], raising=False)
    with _serve_every_suite(wide=True) as endpoint:
        # Ready without building any simulator.
        assert builtins.envhub_builds == []
        # One run over both suites: one policy can't run both, so it fails before either is built, as locally.
        with pytest.raises(
            ValueError, match=r"Suite 'far' .* observation.state shape: expected \(2,\), got \(3,\)"
        ):
            make_env(HubEnvConfig(task="near,far", server=EnvClientConfig(endpoint)), 2)
        assert builtins.envhub_builds == []
        # One run per suite: each gets its own observations, and only its own simulator is built.
        for suite, state, builds in (("far", 3, [5.0]), ("near", 2, [5.0, 0.0])):
            env = make_env(HubEnvConfig(task=suite, server=EnvClientConfig(endpoint)), 2)[suite][0]
            try:
                assert env.reset(seed=[3, 4])[0]["observation.state"].shape == (2, state)
                assert builtins.envhub_builds == builds
            finally:
                env.close()
        # A client that described the server first still opens another suite, against that suite's description.
        from lerobot.env_server import EnvClient

        client = EnvClient(endpoint)
        try:
            client.describe()
            assert client.open(2, "far", seeds=[3, 4]).obs["observation.state"].shape == (2, 3)
        finally:
            client.close()


def test_a_server_holds_a_suite_without_features_to_its_description(hub_repo, monkeypatch):
    # Described with the first suite's shapes, a wider suite is refused when its session builds it.
    hub_repo(_TWO_SUITES)
    monkeypatch.setattr(builtins, "envhub_builds", [], raising=False)
    with _serve_every_suite(wide=True) as endpoint:
        env = make_env(HubEnvConfig(task="far", server=EnvClientConfig(endpoint)), 2)["far"][0]
        try:
            assert {f.name: f.shape for f in env.descriptor.features}["observation.state"] == (2,)
            with pytest.raises(Exception, match=r"Suite 'far' .* first suite: observation.state shape"):
                env.reset(seed=[3, 4])
        finally:
            env.close()


def test_envhub_suites_of_one_run_must_match(hub_repo, monkeypatch):
    hub_repo(_TWO_SUITES)
    monkeypatch.setattr(builtins, "envhub_builds", [], raising=False)
    config = HubEnvConfig(
        hub_path="user/reach", trust_remote_code=True, task="near,far", make_env_kwargs={"wide": True}
    )
    envs = make_env(config, 2)
    try:
        # One policy runs every suite: one with other observations fails when it is built, not at its first step.
        with pytest.raises(
            ValueError,
            match=r"Suite 'far' .* first suite: observation.state shape: expected \(2,\), got \(3,\)",
        ):
            envs["far"][0].reset(seed=[3, 4])
    finally:
        for suite in envs.values():
            suite[0].close()


def test_envhub_suites_of_one_run_must_command_the_same_actions(hub_repo):
    # Same shapes and names, but one suite's actions are joint targets and the other's deltas.
    hub_repo(
        _TWO_SUITES
        + "\n\ndef sim_settings(task, wide=False):\n    return {'control_mode': 'delta' if task == 'far' else 'position'}\n"
    )
    with pytest.raises(ValueError, match=r"Suite 'far' .* control_mode: expected 'position', got 'delta'"):
        make_env(HubEnvConfig(hub_path="user/reach", trust_remote_code=True, task="near,far"), 2)


def test_a_run_that_fails_to_build_closes_the_envs_built_before(hub_repo, monkeypatch):
    # The first suite declares nothing, so it is built to describe the run; the second declares other shapes.
    hub_repo(
        _TWO_SUITES
        + "\n\nReachBatch.close_extras = lambda self, **kwargs: builtins.envhub_closes.append(self.goal)\n"
        + "\n\ndef sim_settings(task, wide=False):\n"
        + "    return {'features': {'observation.state': (3,), 'action': (2,)}} if task == 'far' else {}\n"
    )
    monkeypatch.setattr(builtins, "envhub_builds", [], raising=False)
    monkeypatch.setattr(builtins, "envhub_closes", [], raising=False)
    with pytest.raises(ValueError, match="Suite 'far'"):
        make_env(HubEnvConfig(hub_path="user/reach", trust_remote_code=True, task="near,far"), 2)
    assert builtins.envhub_builds == builtins.envhub_closes == [0.0]


def test_envhub_declared_features_describe_suites_without_building_them(hub_repo, monkeypatch):
    # In another order than the simulator returns them: observations are matched by name.
    features = "{'action': (2,), 'observation.state': (2,), 'observation.images.front': (8, 8, 3)}"
    hub_repo(
        _TWO_SUITES + f"\n\ndef sim_settings(task, wide=False):\n    return {{'features': {features}}}\n"
    )
    monkeypatch.setattr(builtins, "envhub_builds", [], raising=False)
    envs = make_env(HubEnvConfig(hub_path="user/reach", trust_remote_code=True, task="near,far"), 2)
    try:
        assert builtins.envhub_builds == []
        assert {f.name: f.shape for f in envs["near"][0].descriptor.features} == {
            "observation.images.front": (8, 8, 3),
            "observation.state": (2,),
        }
        declared = envs["near"][0].descriptor
        # Built on the first reset and checked against what it declares.
        envs["near"][0].reset(seed=[3, 4])
        assert builtins.envhub_builds == [0.0]
        # Still in the declared order: a 0.7 client compares the described and opened features in order.
        assert envs["near"][0].descriptor.features == declared.features
        assert envs["near"][0].call("_max_episode_steps") == (5, 5)
    finally:
        for suite in envs.values():
            suite[0].close()
    # A simulator building other observations than its suite declares fails on its first reset.
    config = HubEnvConfig(
        hub_path="user/reach", trust_remote_code=True, task="far", make_env_kwargs={"wide": True}
    )
    env = make_env(config, 2)["far"][0]
    try:
        with pytest.raises(
            ValueError, match=r"Suite 'far' .* its declared features: observation.state shape"
        ):
            env.reset(seed=[3, 4])
    finally:
        env.close()
    # Suites declaring different observations fail before any simulator is built.
    hub_repo(
        _TWO_SUITES
        + "\n\ndef sim_settings(task, wide=False):\n"
        + "    return {'features': {'observation.images.front': (8, 8, 3), 'observation.state': (2 if task == 'near' else 3,), 'action': (2,)}}\n"
    )
    builtins.envhub_builds.clear()
    with pytest.raises(ValueError, match="Suite 'far' .* first suite"):
        make_env(HubEnvConfig(hub_path="user/reach", trust_remote_code=True, task="near,far"), 2)
    assert builtins.envhub_builds == []


def test_a_served_envhub_client_needs_no_hub_path():
    # The server loads the repository from its own copy; only an env run in this process needs its path.
    served = HubEnvConfig(server=EnvClientConfig(endpoint="tcp/127.0.0.1:7448"))
    assert served.sim_settings()["kwargs"]["hub_path"] is None
    with pytest.raises(ValueError, match="hub_path is required for an EnvHub environment that is not served"):
        HubEnvConfig().sim_settings()


def test_envhub_batched_repository_must_not_autoreset(hub_repo, monkeypatch):
    hub_repo(_BATCHED_ENV.replace("AutoresetMode.DISABLED", "AutoresetMode.NEXT_STEP"))
    monkeypatch.setattr(builtins, "envhub_builds", [], raising=False)
    with pytest.raises(ValueError, match="must not autoreset"):
        LocalVectorEnv(_envhub(), num_envs=2)


def test_envhub_batched_repository_runs_in_a_worker_process(tmp_path, monkeypatch):
    # A worker can't import env.py, so it loads the repo itself instead of receiving its classes.
    (tmp_path / "env.py").write_text(_BATCHED_ENV)
    monkeypatch.setattr(builtins, "envhub_builds", [], raising=False)
    env = LocalVectorEnv(_envhub(hub_path=str(tmp_path)), num_envs=2, task_id=1, use_async_envs=True)
    try:
        # Built in the worker, not in this process.
        assert builtins.envhub_builds == []
        assert env.reset(seed=[3, 4])[0]["observation.state"].shape == (2, 2)
        assert env.step(np.zeros((2, 2), np.float32))[4]["task"] == ("reach 1.0", "reach 1.0")
    finally:
        env.close()
