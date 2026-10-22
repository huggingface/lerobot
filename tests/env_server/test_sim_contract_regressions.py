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

"""Provider contracts that scalar smoke tests alone do not exercise."""

from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from lerobot.env_server.contracts import EnvDescriptor
from lerobot.envs.configs import AlohaEnv, HubEnvConfig, IsaaclabArenaEnv, UnitreeG1MujocoEnv
from lerobot.envs.factory import make_env
from lerobot.robots.sim import SimRobot, SimRobotConfig
from lerobot.sims import backend as backend_module
from lerobot.sims.adapters import check_explicit_reset
from lerobot.sims.backend import Backend, BackendConfig, ToyEnv
from lerobot.sims.native import envhub
from tests.env_server.toy_env import ToyEnvConfig


class NativeBatch(gym.vector.VectorEnv):
    """A native batch owns its worlds, rather than exposing scalar factories."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.closed = False

    def close(self, **kwargs):
        self.closed = True


def test_typed_envhub_preserves_native_batch_and_factory_arguments(monkeypatch):
    batch = NativeBatch(3)
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return {"native": {0: batch}}

    module = SimpleNamespace(make_env=factory, LEROBOT_SIM={})
    monkeypatch.setattr(
        "lerobot.envs.factory._download_hub_file", lambda *args: ("audit/native", "env.py", "/unused", None)
    )
    monkeypatch.setattr("lerobot.envs.factory._import_hub_module", lambda *args: module)
    monkeypatch.setattr(envhub, "_load_hub_module", lambda *args: ("audit/native", module))
    config = HubEnvConfig(hub_path="audit/native", trust_remote_code=True)
    try:
        assert make_env(config, n_envs=3, use_async_envs=True)["native"][0] is batch
        assert calls == [{"n_envs": 3, "use_async_envs": True, "cfg": config}]
        assert not batch.closed
    finally:
        batch.close()


@pytest.mark.parametrize(
    "config",
    [HubEnvConfig(hub_path="audit/native"), IsaaclabArenaEnv(), UnitreeG1MujocoEnv()],
)
def test_typed_envhub_subclasses_keep_factory_dispatch_and_cache(monkeypatch, config):
    batch = NativeBatch(2)

    def factory(hub_path, n_envs, use_async_envs, trust_remote_code, hub_cache_dir, cfg):
        assert hub_path == config.hub_path
        assert n_envs == 2 and use_async_envs and trust_remote_code
        assert hub_cache_dir == "/audit/hub-cache"
        assert cfg is config
        return {"native": {0: batch}}

    monkeypatch.setattr("lerobot.envs.factory.make_hub_envs", factory)
    try:
        result = make_env(
            config, n_envs=2, use_async_envs=True, trust_remote_code=True, hub_cache_dir="/audit/hub-cache"
        )
        assert result["native"][0] is batch
        assert not batch.closed
    finally:
        batch.close()


def test_scalar_envhub_rejection_closes_all_created_batches(monkeypatch):
    batches = [NativeBatch(1), NativeBatch(1)]
    module = SimpleNamespace(make_env=lambda **kwargs: {"native": dict(enumerate(batches))})
    monkeypatch.setattr(envhub, "_load_hub_module", lambda *args: ("audit/native", module))
    with pytest.raises(ValueError, match="scalar|per-world factories"):
        envhub.create_envhub_envs("", 1, lambda fns: fns, "audit/native", True)
    assert all(batch.closed for batch in batches)


class LongEpisode(gym.Env):
    observation_space = gym.spaces.Dict({"agent_pos": gym.spaces.Box(-1, 1, (2,), np.float32)})
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    _max_episode_steps = 900

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return {"agent_pos": np.zeros(2, np.float32)}, {}

    def step(self, action):
        self.steps += 1
        return {"agent_pos": np.zeros(2, np.float32)}, 0.0, False, self.steps >= 900, {}


def wrapped_long_episode():
    return gym.wrappers.OrderEnforcing(LongEpisode())


@pytest.mark.parametrize("async_worlds", [False, True])
def test_backend_resolves_wrapped_episode_limit(monkeypatch, async_worlds):
    monkeypatch.setattr(backend_module, "ToyEnv", wrapped_long_episode)
    backend = Backend(BackendConfig(), async_worlds=async_worlds)
    try:
        assert backend.descriptor().max_episode_steps == 900
        backend.reset()
        for _ in range(501):
            result = backend.step(np.zeros((1, 2), np.float32))
        assert not result.truncated.any()
    finally:
        backend.close()


def test_local_robot_resolves_wrapped_episode_limit(monkeypatch, tmp_path):
    config = ToyEnvConfig()
    monkeypatch.setattr(config, "create_env", lambda **kwargs: wrapped_long_episode())
    robot = SimRobot(SimRobotConfig(env=config, calibration_dir=tmp_path))
    try:
        robot.connect()
        assert robot.descriptor.max_episode_steps == 900
        for _ in range(501):
            robot.send_action(dict.fromkeys(robot.action_features, 0.0))
        assert not robot.episode_status.truncated
    finally:
        robot.disconnect()


def test_unknown_episode_limit_does_not_invent_a_cutoff(monkeypatch, tmp_path):
    class Unlimited(LongEpisode):
        _max_episode_steps = None

    monkeypatch.setattr(backend_module, "ToyEnv", Unlimited)
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(), calibration_dir=tmp_path))
    try:
        robot.connect()
        assert robot.descriptor.max_episode_steps is None
        for _ in range(501):
            robot.send_action(dict.fromkeys(robot.action_features, 0.0))
        assert not robot.episode_status.truncated
    finally:
        robot.disconnect()


class InstructionEnv(ToyEnv):
    def reset(self, seed=None):
        obs, info = super().reset(seed)
        return obs, {**info, "task_description": "Pick up the red cube"}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, {**info, "task_description": "Put the cube in the bowl"}


@pytest.mark.parametrize("remote", [False, True])
def test_task_info_matches_local_and_server(monkeypatch, request, tmp_path, remote):
    monkeypatch.setattr(backend_module, "ToyEnv", InstructionEnv)
    config = ToyEnvConfig()
    if remote:
        config = replace(config, endpoint=request.getfixturevalue("server")[1])
    robot = SimRobot(SimRobotConfig(env=config, calibration_dir=tmp_path))
    try:
        robot.connect()
        assert robot.task_description == "Pick up the red cube"
        robot.send_action(dict.fromkeys(robot.action_features, 2.0))
        assert robot.task_description == "Put the cube in the bowl"
        assert robot.episode_status.terminated
        robot.hold()
        assert robot.task_description == "Put the cube in the bowl"
        robot.reset_episode()
        assert robot.task_description == "Pick up the red cube"
    finally:
        robot.disconnect()


def test_task_info_survives_worker_process(monkeypatch):
    monkeypatch.setattr(backend_module, "ToyEnv", InstructionEnv)
    backend = Backend(BackendConfig(), async_worlds=True)
    try:
        assert backend.reset().task == ("Pick up the red cube",)
        assert backend.step(np.zeros((1, 2), np.float32)).task == ("Put the cube in the bowl",)
    finally:
        backend.close()


def test_native_batch_cannot_masquerade_as_scalar_gym_env():
    class NativeGymBatch(LongEpisode):
        is_vector_env = True

    with pytest.raises(ValueError, match="scalar|vector"):
        check_explicit_reset(NativeGymBatch())


def test_aloha_rejects_an_incorrect_control_interval():
    with pytest.raises(ValueError, match="fixed 50 Hz"):
        AlohaEnv(fps=25)


def test_aloha_rejects_unsafe_batched_rendering_on_macos(monkeypatch):
    monkeypatch.setattr("lerobot.envs.configs.sys.platform", "darwin")
    with pytest.raises(ValueError, match="use_async_envs=True"):
        make_env(AlohaEnv(), n_envs=1)


def test_aloha_keyboard_mapping_uses_absolute_joints_and_normalized_grippers():
    from tests.env_server.native_workflow import aloha_keyboard_processor

    obs = dict.fromkeys((f"joint_{i}.pos" for i in range(14)), 0.25)
    processor = aloha_keyboard_processor()
    assert processor(({}, obs)) == obs
    action = processor(({"3": 1, "d": 1, "j": 1, "c": 1, "m": 1}, obs))
    assert action["joint_2.pos"] == pytest.approx(0.26)
    assert action["joint_9.pos"] == pytest.approx(0.24)
    assert action["joint_6.pos"] == 0
    assert action["joint_13.pos"] == 1
    assert action["joint_0.pos"] == 0.25
    assert obs["joint_2.pos"] == 0.25


@pytest.mark.parametrize("async_worlds", [False, True])
def test_provider_versions_and_asset_revisions_survive_serialization(monkeypatch, async_worlds):
    class VersionedEnv(ToyEnv):
        metadata = {"sim_build": {"physics": "1.2.3"}, "asset_revisions": {"robot": "sha256:abc"}}

    monkeypatch.setattr(backend_module, "ToyEnv", VersionedEnv)
    backend = Backend(BackendConfig(), async_worlds=async_worlds)
    try:
        descriptor = backend.descriptor()
        assert descriptor.sim_build["physics"] == "1.2.3"
        assert descriptor.asset_revisions == {"robot": "sha256:abc"}
        assert EnvDescriptor.from_dict(descriptor.to_dict()) == descriptor
    finally:
        backend.close()
