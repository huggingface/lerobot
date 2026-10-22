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

"""Scalar Robot behaviour, independent of local or server execution."""

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.sim import SimRobot, SimRobotConfig
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.sims import backend as backend_module
from lerobot.sims.adapters import canonical_observation
from lerobot.sims.backend import ToyEnv
from tests.env_server.toy_env import ToyEnvConfig


def test_local_execution_uses_shared_backend_on_the_callers_thread(tmp_path, monkeypatch):
    events = []

    class World(ToyEnv):
        def __init__(self):
            events.append(("construct", threading.get_ident()))
            super().__init__()

        def reset(self, seed=None):
            events.append(("reset", threading.get_ident()))
            return super().reset(seed)

        def step(self, action):
            events.append(("step", threading.get_ident()))
            return super().step(action)

        def close(self):
            events.append(("close", threading.get_ident()))

    monkeypatch.setattr(backend_module, "ToyEnv", World)
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(robot_type="panda"), calibration_dir=tmp_path))
    assert robot.robot_type == "panda"
    assert [name for name, _ in events] == ["construct", "close"]
    assert robot.client._world is None
    assert not robot.is_connected
    try:
        assert robot.observation_features["image"] == (8, 8, 3)
        robot.connect()
        robot.send_action(dict.fromkeys(robot.action_features, 0.0))
        robot.get_observation()
    finally:
        robot.disconnect()
    assert [name for name, _ in events] == ["construct", "close", "construct", "reset", "step", "close"]
    assert {owner for _, owner in events} == {threading.get_ident()}
    before = list(events)
    _ = robot.observation_features
    _ = robot.action_features
    assert robot.robot_type == "panda" and events == before


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("success", [False, True])
def test_teleop_resets_episodes_and_processors(tmp_path, monkeypatch, request, remote, success):
    from lerobot.processor import make_default_processors
    from lerobot.scripts.lerobot_teleoperate import teleop_loop
    from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig

    endpoint = request.getfixturevalue("server")[1] if remote else None
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(endpoint=endpoint), calibration_dir=tmp_path))
    teleop = MockTeleop(MockTeleopConfig())
    processors = make_default_processors()
    resets = [MagicMock(wraps=processor.reset) for processor in processors]
    for processor, reset in zip(processors, resets, strict=True):
        monkeypatch.setattr(processor, "reset", reset)
    monkeypatch.setattr("lerobot.scripts.lerobot_teleoperate.CycleTimer.wait", lambda self: None)
    episode_steps = []
    episode_length = 1 if success else 5
    num_actions = 2 * episode_length + 1

    def action():
        episode_steps.append(robot.episode_status.step)
        if len(episode_steps) > num_actions:
            raise KeyboardInterrupt
        return dict.fromkeys(robot.action_features, 2.0 if success else 0.0)

    monkeypatch.setattr(teleop, "get_action", action)
    try:
        robot.connect()
        teleop.connect()
        with pytest.raises(KeyboardInterrupt):
            teleop_loop(teleop, robot, 20, *processors)
        assert episode_steps == [i % episode_length for i in range(num_actions + 1)]
        assert [reset.call_count for reset in resets] == [num_actions // episode_length] * 3
    finally:
        robot.disconnect()
        teleop.disconnect()


@pytest.mark.parametrize("control", ["delta", "position"])
def test_wrapped_hold_never_advances_physics(tmp_path, monkeypatch, control):
    config = ToyEnvConfig()
    settings = config.sim_settings()
    monkeypatch.setattr(config, "sim_settings", lambda: {**settings, "control": control})
    robot = SimRobot(SimRobotConfig(env=config, calibration_dir=tmp_path))
    wrapper = ThreadSafeRobot(robot)
    try:
        wrapper.configure_hold()
        robot.connect()
        wrapper.hold()
        assert robot.episode_status.step == 0
        wrapper.send_action(dict.fromkeys(robot.action_features, 0.1))
        wrapper.hold()
        assert robot.episode_status.step == 1
    finally:
        robot.disconnect()


def test_observations_and_terminal_result_are_owned_and_stable(tmp_path):
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(), seed=10, calibration_dir=tmp_path))
    try:
        robot.connect()
        original = robot.get_observation()["image"].copy()
        robot.get_observation()["image"][:] = 255
        np.testing.assert_array_equal(robot.get_observation()["image"], original)
        robot.send_action({"command_0": 2.0, "command_1": 0.0})
        status = robot.episode_status
        assert status.success and status.terminated and status.reward == 1
        robot.get_observation()
        robot.hold()
        assert robot.episode_status == status
        with pytest.raises(RuntimeError, match="episode ended"):
            robot.send_action(dict.fromkeys(robot.action_features, 0.0))
        assert robot.episode_status == status
        robot.reset_episode(seed=10)
        assert robot.episode_status.step == 0
        assert robot.episode_status.reward == 0
        assert not robot.episode_status.success
        assert not robot.episode_status.terminated
        assert not robot.episode_status.truncated
        np.testing.assert_array_equal(robot.get_observation()["image"], original)
    finally:
        robot.disconnect()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), [0, 1]])
def test_invalid_commands_do_not_step_the_world(tmp_path, bad_value):
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(), calibration_dir=tmp_path))
    try:
        robot.connect()
        with pytest.raises((ValueError, TypeError)):
            robot.send_action({"command_0": bad_value, "command_1": 0.0})
        assert robot.episode_status.step == 0
        robot.send_action(dict.fromkeys(robot.action_features, 0.0))
        assert robot.episode_status.step == 1
    finally:
        robot.disconnect()


def test_malformed_step_fails_until_reconnect(tmp_path, monkeypatch):
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(), calibration_dir=tmp_path))
    try:
        robot.connect()
        env = robot.client._world.backend.envs[0]
        original = env.step

        def malformed(action):
            obs, _, terminated, truncated, info = original(action)
            return obs, float("nan"), terminated, truncated, info

        monkeypatch.setattr(env, "step", malformed)
        with pytest.raises(ValueError, match="reward"):
            robot.send_action(dict.fromkeys(robot.action_features, 0.0))
        assert robot.is_connected
        with pytest.raises(RuntimeError, match="disconnect and reconnect"):
            robot.get_observation()
    finally:
        robot.disconnect()


def test_canonical_rgb_keeps_dtype_and_extra_numeric_observations():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    obs = canonical_observation(
        {"observation.images.front": image, "observation.state": np.ones(2), "force": np.ones(3)},
    )
    assert obs["observation.images.front"].dtype == np.uint8
    assert obs["observation.state"].dtype == np.float32
    assert obs["observation.force"].shape == (3,)


def test_feature_discovery_never_samples_or_resets(tmp_path, monkeypatch):
    config = ToyEnvConfig()
    env = ToyEnv()

    def unexpected(*args, **kwargs):
        raise AssertionError("Feature inspection must not sample, reset or step")

    monkeypatch.setattr(env.observation_space, "sample", unexpected)
    monkeypatch.setattr(env, "reset", unexpected)
    monkeypatch.setattr(env, "step", unexpected)
    monkeypatch.setattr(backend_module, "ToyEnv", lambda: env)
    robot = SimRobot(SimRobotConfig(env=config, calibration_dir=tmp_path))
    try:
        assert len(robot.action_features) == 2
        assert robot.observation_features["image"] == (8, 8, 3)
        assert not robot.is_connected
    finally:
        robot.disconnect()


def test_panda_keyboard_mapping_keeps_recording_keys_separate():
    from tests.env_server.native_workflow import PANDA_ACTION_NAMES, panda_keyboard_processor

    processor = panda_keyboard_processor()
    command = processor(({"w": None, "d": None, "e": None, "c": None}, {}))
    assert tuple(command) == PANDA_ACTION_NAMES
    assert command["eef.delta.x"] == command["eef.delta.y"] == command["eef.delta.z"] == 0.005
    assert command["gripper.delta"] == 0.1
    assert command["eef.delta.rx"] == command["eef.delta.ry"] == command["eef.delta.rz"] == 0
    assert not any(processor(({"left": None, "right": None}, {})).values())
    assert processor(({"o": None}, {}))["gripper.delta"] == -0.1


def test_initial_seed_is_used_once_then_reset_continues_rng(tmp_path, monkeypatch):
    config = ToyEnvConfig()
    env = ToyEnv()
    reset = env.reset
    seeds = []

    def seeded_reset(seed=None):
        seeds.append(seed)
        return reset(seed=seed)

    monkeypatch.setattr(env, "reset", seeded_reset)
    monkeypatch.setattr(backend_module, "ToyEnv", lambda: env)
    robot = SimRobot(SimRobotConfig(env=config, seed=42, calibration_dir=tmp_path))
    try:
        robot.connect()
        robot.reset_episode()
        assert seeds == [42, None]
    finally:
        robot.disconnect()


def test_extra_numeric_features_are_rejected_not_dropped(tmp_path, monkeypatch):
    from gymnasium import spaces

    config = ToyEnvConfig()
    env = ToyEnv()
    env.observation_space["force"] = spaces.Box(-1, 1, (3,), np.float32)
    monkeypatch.setattr(backend_module, "ToyEnv", lambda: env)
    closed = []
    monkeypatch.setattr(env, "close", lambda: closed.append(True))
    with pytest.raises(ValueError, match="unsupported.*observation.force"):
        SimRobot(SimRobotConfig(env=config, calibration_dir=tmp_path))
    assert closed == [True]


def test_gym_autoreset_wrapper_is_rejected_before_reset(tmp_path, monkeypatch):
    import gymnasium as gym

    config = ToyEnvConfig()

    class AutoResetEnv(gym.Env):
        observation_space = gym.spaces.Dict({"agent_pos": gym.spaces.Box(-1, 1, (2,), np.float32)})
        action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

        def reset(self, **kwargs):
            raise AssertionError("Reject the wrapper before reset")

    monkeypatch.setattr(backend_module, "ToyEnv", lambda: gym.wrappers.Autoreset(AutoResetEnv()))
    with pytest.raises(ValueError, match="Automatic reset"):
        SimRobot(SimRobotConfig(env=config, calibration_dir=tmp_path))


@pytest.mark.parametrize("key", ["final_observation", "final_obs", "terminal_observation"])
def test_hidden_autoreset_result_faults_instead_of_recording_next_episode(tmp_path, monkeypatch, key):
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(), calibration_dir=tmp_path))
    try:
        robot.connect()
        env = robot.client._world.backend.envs[0]
        step = env.step

        def autoreset(action):
            obs, reward, terminated, truncated, info = step(action)
            return obs, reward, True, truncated, {key: obs}

        monkeypatch.setattr(env, "step", autoreset)
        with pytest.raises(ValueError, match="Automatic-reset"):
            robot.send_action(dict.fromkeys(robot.action_features, 0.0))
        with pytest.raises(RuntimeError, match="disconnect and reconnect"):
            robot.get_observation()
    finally:
        robot.disconnect()


def test_local_and_remote_scalar_episodes_match(server, tmp_path):
    robots = [
        SimRobot(SimRobotConfig(env=ToyEnvConfig(endpoint=endpoint), calibration_dir=tmp_path))
        for endpoint in (None, server[1])
    ]
    try:
        for robot in robots:
            robot.connect()
            robot.reset_episode(seed=12)
        for _ in range(5):
            observations = [robot.get_observation() for robot in robots]
            assert observations[0].keys() == observations[1].keys()
            for name in observations[0]:
                np.testing.assert_array_equal(observations[0][name], observations[1][name])
            for robot in robots:
                robot.send_action({"command_0": 0.1, "command_1": 0.0})
            assert robots[0].episode_status == robots[1].episode_status
        assert robots[0].episode_status.truncated
    finally:
        for robot in robots:
            robot.disconnect()


@pytest.mark.parametrize("rerecord", [False, True])
def test_record_resets_once_per_episode_including_retry(tmp_path, monkeypatch, rerecord):
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.scripts import lerobot_record
    from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig

    config = ToyEnvConfig()
    robot = SimRobot(SimRobotConfig(env=config, calibration_dir=tmp_path / "calibration"))
    teleop = MockTeleop(MockTeleopConfig(n_motors=2))
    monkeypatch.setattr(teleop, "get_action", lambda: {"command_0": 0.1, "command_1": 0.0})
    resets = []

    class World(ToyEnv):
        def reset(self, seed=None):
            resets.append(seed)
            return super().reset(seed)

    monkeypatch.setattr(backend_module, "ToyEnv", World)
    monkeypatch.setattr(lerobot_record, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(lerobot_record, "make_teleoperator_from_config", lambda _: teleop)
    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
    monkeypatch.setattr(lerobot_record, "init_keyboard_listener", lambda: (None, events))
    original = lerobot_record.record_loop
    attempts = []

    def record_episode(**kwargs):
        original(**kwargs)
        attempts.append(1)
        if rerecord and len(attempts) == 1:
            events["rerecord_episode"] = True

    monkeypatch.setattr(lerobot_record, "record_loop", record_episode)
    dataset = lerobot_record.record(
        lerobot_record.RecordConfig(
            robot=robot.config,
            teleop=teleop.config,
            play_sounds=False,
            dataset=DatasetRecordConfig(
                repo_id="local/reset_test",
                root=tmp_path / "dataset",
                fps=20,
                num_episodes=2,
                episode_time_s=10,
                video=False,
                push_to_hub=False,
            ),
        )
    )
    assert len(resets) == 2 + int(rerecord)
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 10
    assert not robot.is_connected


def test_record_train_reload_and_run_act_on_local_sim(tmp_path):
    """Smoke test the normal dataset, policy, saved processors and inference dispatch together."""
    import torch
    from torch.utils.data import DataLoader

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets import LeRobotDataset
    from lerobot.env_server.admission import validate_checkpoint
    from lerobot.inference.sync import SyncInferenceEngine
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.utils.feature_utils import hw_to_dataset_features
    from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig

    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(), calibration_dir=tmp_path / "calibration"))
    teleop = MockTeleop(MockTeleopConfig(n_motors=2, random_values=False, static_values=[0.1, 0.0]))
    dataset = None
    try:
        features = {
            **hw_to_dataset_features(robot.observation_features, "observation", use_video=False),
            **hw_to_dataset_features(robot.action_features, "action"),
        }
        dataset = LeRobotDataset.create(
            "local/sim_roundtrip", fps=20, root=tmp_path / "dataset", features=features, use_videos=False
        )
        robot.connect()
        teleop.connect()
        for episode in range(2):
            if episode:
                robot.reset_episode(seed=episode)
            record_loop(
                robot,
                {"exit_early": False},
                20,
                lambda pair: {"command_0": pair[0]["motor_1.pos"], "command_1": pair[0]["motor_2.pos"]},
                lambda pair: pair[0],
                lambda obs: obs,
                dataset=dataset,
                teleop=teleop,
                control_time_s=10,
                single_task="Move to the target",
            )
            dataset.save_episode()
        dataset.finalize()
        dataset = LeRobotDataset("local/sim_roundtrip", root=tmp_path / "dataset")
        assert dataset.num_episodes == 2
        assert len(dataset) == 10
        config = ACTConfig(
            device="cpu",
            push_to_hub=False,
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
            chunk_size=1,
            n_action_steps=1,
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            use_vae=False,
            pretrained_backbone_weights=None,
        )
        validate_checkpoint(robot.descriptor, config, {})
        policy = ACTPolicy(config)
        preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        batch["action"] = batch["action"].unsqueeze(1)
        batch["action_is_pad"] = torch.zeros((2, 1), dtype=torch.bool)
        optimizer = torch.optim.Adam(policy.get_optim_params(), lr=1e-4)
        loss, _ = policy(preprocessor(batch))
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        checkpoint = tmp_path / "checkpoint"
        policy.save_pretrained(checkpoint)
        preprocessor.save_pretrained(checkpoint)
        postprocessor.save_pretrained(checkpoint)
        loaded = ACTPolicy.from_pretrained(checkpoint)
        preprocessor, postprocessor = make_pre_post_processors(loaded.config, pretrained_path=checkpoint)
        engine = SyncInferenceEngine(
            loaded,
            preprocessor,
            postprocessor,
            features,
            list(robot.action_features),
            "Move to the target",
            "cpu",
            robot.robot_type,
        )
        from lerobot.utils.feature_utils import build_dataset_frame

        robot.reset_episode(seed=42)
        engine.start()
        try:
            engine.reset()
            action = engine.get_action(build_dataset_frame(features, robot.get_observation(), "observation"))
            assert action.shape == (2,)
            robot.send_action(dict(zip(robot.action_features, action.tolist(), strict=True)))
            assert robot.episode_status.step == 1
        finally:
            engine.stop()
    finally:
        robot.disconnect()
        teleop.disconnect()
        if dataset is not None:
            dataset.finalize()
