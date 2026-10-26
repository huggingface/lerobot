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

"""Local environments behave like the same environments behind the env server."""

import time
from dataclasses import dataclass

import numpy as np
import pytest

from lerobot.robots.sim import SimRobot, SimRobotConfig, is_simulation_done
from lerobot.sims.types import SimConfig
from tests.env_server.toy_env import ToyEnvConfig


def _episode(config: ToyEnvConfig) -> list[tuple]:
    env = config.create_env("reach", 0, num_envs=2)
    try:
        obs, info = env.reset(seed=[10, 20])
        results = [(obs, np.zeros(2, np.float32), np.zeros(2, bool), np.zeros(2, bool), info)]
        for actions in ([[2, 0], [0, 0]], [[0.1, 0.1], [0.1, 0.1]], [[0.5, 0], [0.5, 0]]):
            results.append(env.step(np.array(actions, np.float32)))
        return results
    finally:
        env.close()


def test_local_environment_matches_the_server(server):
    local = _episode(ToyEnvConfig())
    served = _episode(ToyEnvConfig(endpoint=server[1]))
    for expected, actual in zip(local, served, strict=True):
        for expected_array, actual_array in zip(expected[1:4], actual[1:4], strict=True):
            np.testing.assert_array_equal(actual_array, expected_array)
        for name, value in expected[0].items():
            np.testing.assert_array_equal(actual[0][name], value)
        for key in ("is_success", "step"):
            np.testing.assert_array_equal(actual[4][key], expected[4][key])
        assert tuple(actual[4]["task"]) == tuple(expected[4]["task"])


@pytest.mark.parametrize("where", ["local", "served"])
def test_sim_robot_advances_once_per_action(where, request, tmp_path):
    if where == "served":
        config = SimRobotConfig(env=ToyEnvConfig(endpoint=request.getfixturevalue("server")[1]))
    else:
        config = SimRobotConfig(env=ToyEnvConfig())
    config.calibration_dir = tmp_path
    robot = SimRobot(config)
    try:
        robot.connect()
        action = dict.fromkeys(robot.action_features, 0.1)
        assert robot.send_action(action) == pytest.approx(action)
        assert robot.latest.step[0] == 1
        state = [name for name, kind in robot.observation_features.items() if kind is float]
        before = robot.get_observation()
        time.sleep(0.1)
        robot.hold()
        after = robot.get_observation()
        assert [after[name] for name in state] == [before[name] for name in state]
        assert robot.latest.step[0] == 1
        robot.send_action(action)
        assert robot.latest.step[0] == 2
    finally:
        robot.disconnect()


def test_sim_robot_config_needs_an_environment():
    with pytest.raises(ValueError, match="--robot.env.type"):
        SimRobotConfig()


def test_sim_robot_env_choices_exclude_envs_built_around_a_robot():
    import sys
    from dataclasses import field

    import draccus

    from lerobot.robots import RobotConfig

    @dataclass
    class Wrapper:
        robot: RobotConfig = field(default_factory=lambda: SimRobotConfig(env=ToyEnvConfig()))

    # gym_manipulator embeds a RobotConfig: offering it here would make draccus expand robot and env
    # choices into each other forever.
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(300)
    try:
        args = ["--robot.type=sim", "--robot.env.type=toy_test"]
        cfg = draccus.parse(Wrapper, args=args)
    finally:
        sys.setrecursionlimit(limit)
    assert isinstance(cfg.robot.env, ToyEnvConfig)
    with pytest.raises(SystemExit):
        draccus.parse(Wrapper, args=["--robot.type=sim", "--robot.env.type=gym_manipulator"])


def _served(deployment: str):
    import socket
    import threading

    from lerobot.env_server.server import EnvServer, ServerConfig
    from lerobot.transport.zenoh import ZenohConfig

    pytest.importorskip("zenoh")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        endpoint = f"tcp/127.0.0.1:{sock.getsockname()[1]}"
    server = EnvServer(ServerConfig(deployment=deployment, zenoh=ZenohConfig(listen_endpoints=[endpoint])))
    thread = threading.Thread(target=server.serve)
    thread.start()
    assert server.ready.wait(5)
    return server, thread, endpoint


@pytest.mark.parametrize("where", ["local", "served"])
def test_env_config_creates_vector_envs_locally_or_served(where):
    from lerobot.env_server.client import BatchedEnv
    from lerobot.sims.vector_env import VectorEnv

    config = ToyEnvConfig()
    server = thread = None
    if where == "served":
        server, thread, config.endpoint = _served(config.type)
    try:
        envs = config.create_envs(n_envs=2)
        env = envs["reach"][0]
        expected_type = BatchedEnv if where == "served" else VectorEnv
        assert isinstance(env, expected_type)
        # Canonical observation names, as the server describes them.
        assert set(config.features) == {"observation.state", "observation.image", "action"}
        obs, _ = env.reset(seed=[1, 2])
        assert obs["observation.state"].shape == (2, 2)
        env.step(np.zeros((2, 2), np.float32))
        env.close()
    finally:
        if server is not None:
            server.stop.set()
            thread.join(5)


def test_served_env_must_be_the_env_the_command_describes():
    config = ToyEnvConfig()
    server, thread, config.endpoint = _served(config.type)
    config.fps = 30  # the server's toy runs at 20 fps
    try:
        with pytest.raises(ValueError, match="fps: server 20"):
            config.create_envs(n_envs=1)
    finally:
        server.stop.set()
        thread.join(5)


def test_sim_robot_records_the_body_its_env_declares(tmp_path):
    from lerobot.env_server.contracts import decode_descriptor, encode_descriptor

    for declared, expected in ((None, "sim"), ("so_follower", "so_follower")):
        robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(robot_type=declared), calibration_dir=tmp_path))
        try:
            assert (robot.name, robot.robot_type) == ("sim", expected)
            # A newer server may declare more: this client keeps what it knows.
            wire = {**encode_descriptor(robot.descriptor), "joint_units": ["rad", "rad"]}
            assert decode_descriptor(wire).robot_type == robot.descriptor.robot_type
        finally:
            robot.disconnect()
    # Served, a server simulating another body is refused: its datasets would record the wrong robot.
    config = ToyEnvConfig(robot_type="so_follower")
    server, thread, config.endpoint = _served(config.type)
    try:
        with pytest.raises(ValueError, match="robot_type: server ''"):
            SimRobot(SimRobotConfig(env=config, calibration_dir=tmp_path))
    finally:
        server.stop.set()
        thread.join(5)


def test_environments_in_separate_processes_match_sequential_environments():
    from lerobot.sims.vector_env import VectorEnv

    results = []
    for use_async_envs in (False, True):
        env = VectorEnv(SimConfig(), num_envs=3, use_async_envs=use_async_envs)
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
        np.testing.assert_array_equal(parallel.step, sequential.step)
        assert parallel.task == sequential.task


def test_local_tasks_build_their_simulators_on_first_use():
    # lerobot-eval creates every task's environment up front, then prefetches the next one with _ensure.
    env = ToyEnvConfig().create_envs(n_envs=2)["reach"][0]
    try:
        assert not env._envs
        env.reset(seed=[1, 2])
        assert len(env._envs) == 2
    finally:
        env.close()


def test_sim_robot_names_the_actions_it_takes(tmp_path):
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(), calibration_dir=tmp_path))
    try:
        robot.connect()
        with pytest.raises(ValueError, match=r"missing \['command_1'\], unknown \['shoulder.pos'\]"):
            robot.send_action({"command_0": 0.0, "shoulder.pos": 0.0})
    finally:
        robot.disconnect()


def test_batched_env_works_under_gymnasium_vector_wrappers():
    from gymnasium.wrappers.vector import RecordEpisodeStatistics

    env = RecordEpisodeStatistics(ToyEnvConfig().create_envs(n_envs=2)["reach"][0])
    try:
        env.reset(seed=[1, 2])
        env.step(np.zeros((2, 2), np.float32))
    finally:
        env.close()


@pytest.mark.parametrize("where", ["local", "served"])
def test_sim_robot_preserves_episode_end_until_explicit_reset(where, request, tmp_path):
    config = ToyEnvConfig(endpoint=request.getfixturevalue("server")[1] if where == "served" else None)
    robot = SimRobot(SimRobotConfig(env=config, calibration_dir=tmp_path))
    try:
        robot.connect()
        assert robot.task_description == "Move to the target"

        # The toy terminates once x passes 1. Preserve that transition for the episode owner.
        robot.send_action({"command_0": 2.0, "command_1": 0.0})
        assert robot.latest.terminated[0]
        assert robot.latest.is_success[0]
        assert robot.latest.reward[0] == 1
        assert robot.latest.step[0] == 1
        terminal_observation = robot.get_observation()
        assert terminal_observation["state_0"] >= 1

        # Further commands cannot advance a terminal environment.
        robot.send_action({"command_0": -1.0, "command_1": 0.0})
        assert is_simulation_done(robot)
        assert robot.latest.step[0] == 1
        frozen_observation = robot.get_observation()
        for name, value in terminal_observation.items():
            np.testing.assert_array_equal(frozen_observation[name], value)

        initial_observation = robot.reset(seed=3)
        assert not is_simulation_done(robot)
        assert robot.latest.step[0] == 0
        assert initial_observation["state_0"] < 1

        # Truncation is preserved in the same way.
        for _ in range(5):
            robot.send_action({"command_0": 0.0, "command_1": 0.0})
        assert robot.latest.truncated[0]
        assert robot.latest.step[0] == 5
    finally:
        robot.disconnect()


@pytest.mark.parametrize("where", ["local", "served"])
def test_recorded_sim_episodes_can_train_and_reload_a_policy(where, request, tmp_path, monkeypatch, clock):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import torch

    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.datasets import LeRobotDataset
    from lerobot.processor import make_default_processors
    from lerobot.scripts import lerobot_record
    from lerobot.teleoperators import Teleoperator
    from tests.mocks.mock_teleop import MockTeleopConfig

    env = ToyEnvConfig(endpoint=request.getfixturevalue("server")[1] if where == "served" else None)
    robot = SimRobot(SimRobotConfig(env=env, calibration_dir=tmp_path / "calibration"))
    teleop = MagicMock(spec=Teleoperator)
    teleop.get_action.return_value = {"command_0": 2.0, "command_1": 0.0}
    monkeypatch.setattr(lerobot_record, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(lerobot_record, "make_teleoperator_from_config", lambda _: teleop)
    monkeypatch.setattr(lerobot_record, "init_logging", lambda: None)
    monkeypatch.setattr(lerobot_record, "log_say", lambda *args, **kwargs: None)
    monkeypatch.setattr(lerobot_record, "time", clock)
    monkeypatch.setattr(
        lerobot_record,
        "init_keyboard_listener",
        lambda: (None, {"exit_early": False, "stop_recording": False, "rerecord_episode": False}),
    )
    cfg = lerobot_record.RecordConfig(
        robot=robot.config,
        teleop=MockTeleopConfig(),
        play_sounds=False,
        dataset=DatasetRecordConfig(
            repo_id="local/sim_demo",
            root=tmp_path / "dataset",
            single_task="Move to the target",
            fps=20,
            num_episodes=2,
            episode_time_s=10,
            reset_time_s=10,
            video=False,
            push_to_hub=False,
            no_stamp=True,
        ),
    )
    processors = [MagicMock(wraps=pipeline) for pipeline in make_default_processors()]
    dataset = lerobot_record.record(cfg, *processors)
    assert dataset.num_episodes == 2
    assert teleop.get_action.call_count == 2
    for processor in processors:
        assert processor.reset.call_count == 2

    reader = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root)
    assert len(reader) == 2
    samples = [reader[index] for index in range(2)]
    assert all(sample["observation.state"][0] < 1 for sample in samples)
    checkpoint_config = SimpleNamespace(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )
    robot.descriptor.validate_checkpoint(checkpoint_config, {})
    # Native benchmark camera names remain unchanged.
    assert "observation.image" in env.features
    legacy_config = SimpleNamespace(
        input_features={"observation.image": checkpoint_config.input_features["observation.images.image"]},
        output_features=checkpoint_config.output_features,
    )
    robot.descriptor.validate_checkpoint(legacy_config, {"observation.images.image": "observation.image"})

    pytest.importorskip("torchvision")
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy(
        ACTConfig(
            input_features=checkpoint_config.input_features,
            output_features=checkpoint_config.output_features,
            device="cpu",
            pretrained_backbone_weights=None,
            chunk_size=1,
            n_action_steps=1,
            dim_model=16,
            n_heads=4,
            dim_feedforward=32,
            n_encoder_layers=1,
            n_decoder_layers=1,
            use_vae=False,
        )
    )
    batch = {name: torch.stack([sample[name] for sample in samples]) for name in policy.config.input_features}
    batch["action"] = torch.stack([sample["action"] for sample in samples]).unsqueeze(1)
    batch["action_is_pad"] = torch.zeros((2, 1), dtype=torch.bool)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    checkpoint = tmp_path / "policy"
    policy.save_pretrained(checkpoint)
    policy = ACTPolicy.from_pretrained(checkpoint)
    robot.descriptor.validate_checkpoint(policy.config, {})
    try:
        robot.connect()
        for _ in range(5):
            observation = robot.get_observation()
            action = policy.select_action(
                {
                    "observation.state": torch.tensor([[observation["state_0"], observation["state_1"]]]),
                    "observation.images.image": torch.from_numpy(observation["image"])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255,
                }
            )
            robot.send_action(dict(zip(robot.action_features, action[0].tolist(), strict=True)))
            if is_simulation_done(robot):
                break
        assert is_simulation_done(robot)
    finally:
        robot.disconnect()


def test_teleop_resets_simulation_and_processors(tmp_path, monkeypatch, clock):
    from unittest.mock import MagicMock

    from lerobot.processor import make_default_processors
    from lerobot.scripts import lerobot_teleoperate

    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(), calibration_dir=tmp_path))
    teleop = MagicMock()
    teleop.get_action.return_value = {"command_0": 2.0, "command_1": 0.0}
    processors = [MagicMock(wraps=pipeline) for pipeline in make_default_processors()]
    monkeypatch.setattr(lerobot_teleoperate, "time", clock)
    try:
        robot.connect()
        lerobot_teleoperate.teleop_loop(teleop, robot, 20, *processors, duration=0.12)
        assert teleop.get_action.call_count >= 2
        for processor in processors:
            assert processor.reset.call_count == teleop.get_action.call_count - 1
    finally:
        robot.disconnect()
