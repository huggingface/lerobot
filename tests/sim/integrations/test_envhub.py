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

"""Environments served from an EnvHub repository."""

import numpy as np
import pytest

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


def test_envhub_serves_a_hub_repository(tmp_path, monkeypatch):
    from lerobot.sim.config import SimConfig
    from lerobot.sim.integrations import envhub
    from lerobot.sim.vector_env import VectorEnv

    env_file = tmp_path / "env.py"
    env_file.write_text(_ENVHUB_ENV)
    # Only the network download is stubbed; the consent check and URL parsing run for real.
    monkeypatch.setattr(envhub, "hf_hub_download", lambda **kwargs: str(env_file))
    config = SimConfig(
        type="envhub",
        task="",
        semantics="envhub-v1",
        kwargs={"hub_path": "user/reach", "trust_remote_code": True},
    )
    env = VectorEnv(config, num_envs=2)
    try:
        assert env.tasks == {"reach_suite": ["0"]}
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
    config.kwargs["trust_remote_code"] = False
    with pytest.raises(RuntimeError, match="Refusing"):
        VectorEnv(config, num_envs=1)


def test_envhub_make_env_takes_the_options_of_the_command(tmp_path, monkeypatch):
    from lerobot.envs.configs import HubEnvConfig
    from lerobot.sim.integrations import envhub
    from lerobot.sim.vector_env import VectorEnv

    # The repository's make_env sizes its camera from an option of its own.
    env_file = tmp_path / "env.py"
    env_file.write_text(
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
    monkeypatch.setattr(envhub, "hf_hub_download", lambda **kwargs: str(env_file))
    config = HubEnvConfig(hub_path="user/reach", trust_remote_code=True, make_env_kwargs={"size": 4})
    env = VectorEnv(config.sim_config())
    try:
        shapes = {f.name: f.shape for f in env.descriptor.features}
        assert shapes["observation.images.front"] == (4, 4, 3)
    finally:
        env.close()


def test_envhub_serves_a_local_directory(tmp_path, monkeypatch):
    from lerobot.sim.config import SimConfig
    from lerobot.sim.integrations import envhub
    from lerobot.sim.vector_env import VectorEnv

    (tmp_path / "env.py").write_text(_ENVHUB_ENV)

    def no_download(**kwargs):
        raise AssertionError("A local directory is not downloaded")

    monkeypatch.setattr(envhub, "hf_hub_download", no_download)
    env = VectorEnv(
        SimConfig(
            type="envhub",
            task="",
            semantics="envhub-v1",
            kwargs={"hub_path": str(tmp_path), "trust_remote_code": True},
        )
    )
    try:
        assert env.tasks == {"reach_suite": ["0"]}
    finally:
        env.close()


def test_envhub_module_runs_once_per_server_process(tmp_path, monkeypatch):
    import builtins

    from lerobot.sim.config import SimConfig
    from lerobot.sim.integrations import envhub
    from lerobot.sim.vector_env import VectorEnv

    # A simulator app that can start only once per process (e.g. Isaac Sim) lives in module state.
    env_file = tmp_path / "env.py"
    env_file.write_text("import builtins\nbuiltins.envhub_imports += 1\n" + _ENVHUB_ENV)
    monkeypatch.setattr(builtins, "envhub_imports", 0, raising=False)
    monkeypatch.setattr(envhub, "hf_hub_download", lambda **kwargs: str(env_file))
    config = SimConfig(
        type="envhub",
        task="",
        semantics="envhub-v1",
        kwargs={"hub_path": "user/reach", "trust_remote_code": True},
    )
    # The probe and two sessions each build their environments.
    for _ in range(3):
        env = VectorEnv(config)
        env.reset(seed=[0])
        env.close()
    assert builtins.envhub_imports == 1


def test_envhub_needs_dict_observations(tmp_path, monkeypatch):
    from lerobot.sim.config import SimConfig
    from lerobot.sim.integrations import envhub
    from lerobot.sim.vector_env import VectorEnv

    env_file = tmp_path / "env.py"
    # A flat array observation, which local LeRobot execution does not accept either.
    env_file.write_text(
        "import gymnasium as gym\n"
        "def make_env(n_envs=1, use_async_envs=False):\n"
        "    return gym.vector.SyncVectorEnv([lambda: gym.make('CartPole-v1')] * n_envs)\n"
    )
    monkeypatch.setattr(envhub, "hf_hub_download", lambda **kwargs: str(env_file))
    config = SimConfig(
        type="envhub",
        task="",
        semantics="envhub-v1",
        kwargs={"hub_path": "user/reach", "trust_remote_code": True},
    )
    with pytest.raises(ValueError, match="dict observations"):
        VectorEnv(config)


def test_envhub_settings_come_from_the_repository(tmp_path, monkeypatch):
    from lerobot.sim.config import SimConfig
    from lerobot.sim.integrations import envhub

    declared = (
        'LEROBOT_SIM = {"fps": 15, "control_mode": "delta", "semantics": "reach-delta-v1",'
        ' "state_names": ["x", "y"], "action_names": ["dx", "dy"], "robot_type": "so_follower"}\n'
    )
    env_file = tmp_path / "env.py"
    monkeypatch.setattr(envhub, "hf_hub_download", lambda **kwargs: str(env_file))
    sim = {"type": "envhub", "task": "", "kwargs": {"hub_path": "user/reach", "trust_remote_code": True}}
    # Repository values fill the settings left out; an explicit value wins over them.
    env_file.write_text(_ENVHUB_ENV + declared)
    resolved = SimConfig.resolve({**sim, "fps": 50})
    assert (resolved.fps, resolved.control_mode, resolved.semantics) == (50, "delta", "reach-delta-v1")
    assert (resolved.state_names, resolved.action_names) == (("x", "y"), ("dx", "dy"))
    assert resolved.robot_type == "so_follower"
    # Repositories without LEROBOT_SIM keep working with the defaults.
    env_file.write_text(_ENVHUB_ENV)
    resolved = SimConfig.resolve(sim)
    assert (resolved.fps, resolved.control_mode, resolved.semantics) == (30, "position", "envhub-v1")
    # A typo in LEROBOT_SIM is an error, not a silently ignored setting.
    env_file.write_text(_ENVHUB_ENV + 'LEROBOT_SIM = {"fsp": 30}\n')
    with pytest.raises(ValueError, match="unknown keys"):
        SimConfig.resolve(sim)


def test_envhub_rejects_custom_vector_behavior_without_unwrapping_it() -> None:
    import gymnasium as gym
    import pytest

    from lerobot.sim.integrations.envhub import _env_factories

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
        assert not hasattr(env, "descriptor")
    finally:
        env.close()
    config.endpoint = "tcp/127.0.0.1:7448"
    with pytest.raises(ValueError, match="endpoint"):
        config.create_envs(1)
    config.endpoint = None
    with pytest.raises(ValueError, match="SimRobot"):
        config.create_env()
