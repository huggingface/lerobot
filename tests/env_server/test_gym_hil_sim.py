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


"""Opt-in native rendering tests: LEROBOT_TEST_NATIVE_SIM=1 uv run pytest tests/env_server/test_gym_hil_sim.py."""

import multiprocessing
import os
import socket
from dataclasses import replace

import numpy as np
import pytest

from lerobot.robots.sim import SimRobot, SimRobotConfig
from tests.env_server.native_workflow import PANDA_ACTION_NAMES, GymHilEnvConfig, panda_keyboard_processor

pytestmark = pytest.mark.skipif(
    os.environ.get("LEROBOT_TEST_NATIVE_SIM") != "1",
    reason="Native MuJoCo rendering requires gym-hil and an available graphics context",
)


def _serve(endpoint, ready, stop):
    from lerobot.env_server.server import EnvServer, ServerConfig
    from lerobot.transport.zenoh import ZenohConfig

    server = EnvServer(
        ServerConfig(
            deployment="gym_hil_test",
            sim=GymHilEnvConfig(episode_length=3).backend_config(),
            zenoh=ZenohConfig(listen_endpoints=[endpoint]),
        )
    )
    server.ready = ready
    server.stop = stop
    server.serve()


@pytest.fixture
def native_server():
    pytest.importorskip("gym_hil")
    pytest.importorskip("zenoh")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        endpoint = f"tcp/127.0.0.1:{sock.getsockname()[1]}"
    context = multiprocessing.get_context("spawn")
    ready, stop = context.Event(), context.Event()
    process = context.Process(target=_serve, args=(endpoint, ready, stop))
    process.start()
    try:
        assert ready.wait(30), f"Native simulator server failed to start (exit={process.exitcode})"
        yield endpoint
    finally:
        stop.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert not process.is_alive()


def test_native_local_and_server_episode_parity(native_server, tmp_path):
    config = GymHilEnvConfig(episode_length=3)
    robots = [
        SimRobot(SimRobotConfig(env=replace(config, endpoint=endpoint), calibration_dir=tmp_path))
        for endpoint in (None, native_server)
    ]
    try:
        for robot in robots:
            robot.connect()
            robot.reset_episode(seed=42)
        initial_observations = [robot.get_observation() for robot in robots]
        assert robots[0].task_description == robots[1].task_description == config.task
        processor = panda_keyboard_processor()
        command = processor(({}, {}))
        for _ in range(3):
            observations = [r.get_observation() for r in robots]
            for key in observations[0]:
                np.testing.assert_allclose(observations[0][key], observations[1][key], atol=1e-6)
            for robot in robots:
                assert robot.send_action(command) == command
            assert robots[0].episode_status == robots[1].episode_status
        for robot, initial in zip(robots, initial_observations, strict=True):
            assert robot.episode_status.truncated
            terminal = robot.get_observation()
            robot.hold()
            np.testing.assert_array_equal(robot.get_observation()["front"], terminal["front"])
            with pytest.raises(RuntimeError, match="episode ended"):
                robot.send_action(command)
            robot.reset_episode(seed=42)
            assert robot.episode_status.step == 0
            for name, value in robot.get_observation().items():
                np.testing.assert_allclose(value, initial[name], atol=1e-6)
    finally:
        for robot in robots:
            robot.disconnect()


@pytest.mark.parametrize("remote", [False, True])
def test_native_keyboard_recording_uses_normal_entrypoint(native_server, tmp_path, monkeypatch, remote):
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.scripts import lerobot_record
    from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig

    teleop = MockTeleop(MockTeleopConfig(n_motors=4))
    monkeypatch.setattr(
        teleop,
        "get_action",
        lambda: {},
    )
    monkeypatch.setattr(lerobot_record, "make_teleoperator_from_config", lambda _: teleop)
    monkeypatch.setattr(
        lerobot_record,
        "init_keyboard_listener",
        lambda: (
            None,
            {
                "exit_early": False,
                "stop_recording": False,
                "rerecord_episode": False,
            },
        ),
    )
    dataset = lerobot_record.record(
        lerobot_record.RecordConfig(
            robot=SimRobotConfig(
                env=GymHilEnvConfig(endpoint=native_server if remote else None, episode_length=3),
                calibration_dir=tmp_path / "calibration",
            ),
            teleop=teleop.config,
            play_sounds=False,
            dataset=DatasetRecordConfig(
                repo_id="local/gym_hil_smoke",
                root=tmp_path / "dataset",
                fps=10,
                num_episodes=2,
                episode_time_s=10,
                reset_time_s=0,
                video=False,
                push_to_hub=False,
            ),
        ),
        teleop_action_processor=panda_keyboard_processor(),
    )
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 6
    assert dataset.features["action"]["names"] == list(PANDA_ACTION_NAMES)
    np.testing.assert_array_equal(dataset[0]["action"].numpy(), np.zeros(7))
    assert not (dataset.root / "meta" / "simulation.json").exists()
    if not remote:
        from tests.env_server.native_workflow import train_and_rollout

        train_and_rollout(dataset, GymHilEnvConfig(episode_length=3), native_server, tmp_path, monkeypatch)


def test_native_teleop_stops_cleanly_at_episode_boundary(tmp_path):
    pytest.importorskip("gym_hil")
    from lerobot.processor import make_default_processors
    from lerobot.scripts.lerobot_teleoperate import teleop_loop
    from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig

    robot = SimRobot(SimRobotConfig(env=GymHilEnvConfig(episode_length=2), calibration_dir=tmp_path))
    teleop = MockTeleop(MockTeleopConfig(n_motors=4))
    teleop.get_action = lambda: {}
    _, action_processor, observation_processor = make_default_processors()
    try:
        robot.connect()
        teleop.connect()
        teleop_loop(
            teleop,
            robot,
            fps=10,
            duration=10,
            teleop_action_processor=panda_keyboard_processor(),
            robot_action_processor=action_processor,
            robot_observation_processor=observation_processor,
        )
        assert robot.episode_status.step == 2 and robot.episode_status.truncated
    finally:
        robot.disconnect()
        teleop.disconnect()


@pytest.mark.parametrize("remote", [False, True])
def test_native_rtc_executes_queued_actions_during_delayed_inference(native_server, tmp_path, remote):
    from tests.inference.rtc_helpers import assert_delayed_chunk_overlap, sim_rtc

    robot = SimRobot(
        SimRobotConfig(
            env=GymHilEnvConfig(episode_length=3, endpoint=native_server if remote else None),
            seed=42,
            calibration_dir=tmp_path,
        )
    )
    try:
        robot.connect()
        with sim_rtc(robot) as rig:
            assert_delayed_chunk_overlap(rig, delay_steps=1)
            assert robot.episode_status.truncated
            terminal = robot.get_observation()["front"]
            rig.engine.pause()
            rig.wrapper.hold()
            np.testing.assert_array_equal(robot.get_observation()["front"], terminal)
    finally:
        robot.disconnect()
