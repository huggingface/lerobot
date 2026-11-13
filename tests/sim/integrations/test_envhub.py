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

import gymnasium as gym
import httpx
import numpy as np
import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from lerobot.envs import EnvClientConfig, HubEnvConfig
from lerobot.envs.factory import make_env, make_task_env
from lerobot.sim import LocalVectorEnv, SimConfig
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
    monkeypatch.setattr(envhub, "snapshot_download", lambda *args, **kwargs: str(snapshot))
    return (snapshot / "env.py").write_text


def _envhub(hub_path: str = "user/reach", trust_remote_code: bool = True) -> SimConfig:
    return SimConfig(
        type="envhub",
        task="",
        semantics="envhub-v1",
        kwargs={"hub_path": hub_path, "trust_remote_code": trust_remote_code},
    )


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


@pytest.mark.parametrize(
    "hub_path,available_type,expected_types",
    [
        ("user/reach", "model", ["model"]),
        ("user/reach", "space", ["model", "space"]),
        ("spaces/user/reach", "space", ["space"]),
        ("models/user/reach", "model", ["model"]),
    ],
)
def test_envhub_selects_the_repository_type(tmp_path, monkeypatch, hub_path, available_type, expected_types):
    (tmp_path / "custom.py").write_text(_ENVHUB_ENV)
    calls = []

    def download(repo_id, **kwargs):
        assert repo_id == "user/reach"
        assert kwargs["revision"] == "v1"
        assert kwargs["cache_dir"] == str(tmp_path / "cache")
        calls.append(kwargs["repo_type"])
        if kwargs["repo_type"] != available_type:
            raise RepositoryNotFoundError(
                "No such repository",
                response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co")),
            )
        return str(tmp_path)

    monkeypatch.setattr(envhub, "snapshot_download", download)
    envs = envhub.make_envs(
        hub_path=f"{hub_path}@v1:custom.py",
        n_envs=1,
        use_async_envs=False,
        trust_remote_code=True,
        hub_cache_dir=str(tmp_path / "cache"),
    )
    env = envs["reach_suite"][0]
    try:
        assert env.reset(seed=0)[0]["agent_pos"].shape == (1, 2)
    finally:
        env.close()
    assert calls == expected_types


@pytest.mark.parametrize(
    "error_type", [GatedRepoError, RevisionNotFoundError, HfHubHTTPError, ConnectionError]
)
def test_envhub_does_not_change_repository_on_access_revision_or_network_errors(monkeypatch, error_type):
    calls = []
    error = (
        error_type(
            "Cannot download",
            response=httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co")),
        )
        if issubclass(error_type, HfHubHTTPError)
        else error_type("Cannot download")
    )

    def download(*args, **kwargs):
        calls.append(kwargs["repo_type"])
        raise error

    monkeypatch.setattr(envhub, "snapshot_download", download)
    with pytest.raises(error_type) as caught:
        LocalVectorEnv(_envhub())
    assert caught.value is error
    assert calls == ["model"]


def test_explicit_space_does_not_fall_back_to_a_model(monkeypatch):
    calls = []

    def download(*args, **kwargs):
        calls.append(kwargs["repo_type"])
        raise RepositoryNotFoundError(
            "No such Space",
            response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co")),
        )

    monkeypatch.setattr(envhub, "snapshot_download", download)
    with pytest.raises(RepositoryNotFoundError):
        LocalVectorEnv(_envhub(hub_path="spaces/user/reach"))
    assert calls == ["space"]


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
    env = LocalVectorEnv(config.sim_config())
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

    def download(repo_id, **kwargs):
        downloads.append((repo_id, kwargs["revision"]))
        return str(tmp_path)

    monkeypatch.setattr(envhub, "snapshot_download", download)
    with pytest.raises(FileNotFoundError, match="custom.py in repository user/reach@v1"):
        LocalVectorEnv(_envhub(hub_path="user/reach@v1:custom.py"))
    assert downloads == [("user/reach", "v1")]


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
        'LEROBOT_SIM = {"fps": 15, "control_mode": "delta", "semantics": "reach-delta-v1",'
        ' "state_names": ["x", "y"], "action_names": ["dx", "dy"], "robot_type": "so_follower"}\n'
    )
    sim = {"type": "envhub", "task": "", "kwargs": {"hub_path": "user/reach", "trust_remote_code": True}}
    # Repository values fill the settings left out; an explicit value wins over them.
    hub_repo(_ENVHUB_ENV + declared)
    resolved = SimConfig.resolve({**sim, "fps": 50})
    assert (resolved.fps, resolved.control_mode, resolved.semantics) == (50, "delta", "reach-delta-v1")
    assert (resolved.state_names, resolved.action_names) == (("x", "y"), ("dx", "dy"))
    assert resolved.robot_type == "so_follower"
    # Repositories without LEROBOT_SIM keep working with the defaults.
    hub_repo(_ENVHUB_ENV)
    resolved = SimConfig.resolve(sim)
    assert (resolved.fps, resolved.control_mode, resolved.semantics) == (30, "position", "envhub-v1")
    # A typo or an old key in LEROBOT_SIM is an error naming the closest valid key, not an ignored setting.
    hub_repo(_ENVHUB_ENV + 'LEROBOT_SIM = {"fsp": 30, "control": "delta", "gripper_indices": [6]}\n')
    with pytest.raises(ValueError) as error:
        SimConfig.resolve(sim)
    message = str(error.value)
    assert "'control' (did you mean 'control_mode'?)" in message
    assert "'fsp' (did you mean 'fps'?)" in message
    assert "'gripper_indices'; expected" in message


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
