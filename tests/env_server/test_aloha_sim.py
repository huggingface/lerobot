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

"""A second native provider: bimanual absolute joints rather than Panda Cartesian deltas."""

import multiprocessing
import os
import socket
from dataclasses import replace
from importlib.metadata import version

import numpy as np
import pytest

from lerobot.envs.configs import AlohaEnv
from lerobot.envs.factory import make_env
from lerobot.robots.sim import SimRobot, SimRobotConfig
from tests.env_server.native_workflow import aloha_keyboard_processor

pytestmark = pytest.mark.skipif(
    os.environ.get("LEROBOT_TEST_NATIVE_SIM") != "1",
    reason="Native Aloha tests require gym-aloha and an available graphics context",
)


def _serve(endpoint, ready, stop):
    from lerobot.env_server.server import EnvServer, ServerConfig
    from lerobot.transport.zenoh import ZenohConfig

    server = EnvServer(
        ServerConfig(
            deployment="aloha",
            sim=AlohaEnv(task="AlohaTransferCube-v0", episode_length=3).backend_config(),
            zenoh=ZenohConfig(listen_endpoints=[endpoint]),
        )
    )
    server.ready = ready
    server.stop = stop
    server.serve()


@pytest.fixture
def aloha_server():
    pytest.importorskip("gym_aloha")
    pytest.importorskip("zenoh")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        endpoint = f"tcp/127.0.0.1:{sock.getsockname()[1]}"
    context = multiprocessing.get_context("spawn")
    ready, stop = context.Event(), context.Event()
    process = context.Process(target=_serve, args=(endpoint, ready, stop))
    process.start()
    try:
        assert ready.wait(30), f"Aloha server failed to start (exit={process.exitcode})"
        yield endpoint
    finally:
        stop.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert not process.is_alive()


def test_aloha_local_and_server_parity(aloha_server, tmp_path):
    config = AlohaEnv(task="AlohaTransferCube-v0", episode_length=3)
    robots = [
        SimRobot(SimRobotConfig(env=replace(config, endpoint=endpoint), seed=42, calibration_dir=tmp_path))
        for endpoint in (None, aloha_server)
    ]
    try:
        for robot in robots:
            robot.connect()
            assert robot.descriptor.max_episode_steps == 3
            assert robot.descriptor.sim_build["gym-aloha"] == version("gym-aloha")
            assert "mujoco" in robot.descriptor.sim_build
            assert len(robot.action_features) == 14
        assert robots[0].task_description == robots[1].task_description == "transfer_cube"
        processor = aloha_keyboard_processor()
        for _ in range(3):
            observations = [robot.get_observation() for robot in robots]
            for name in robots[0].action_features:
                assert observations[0][name] == pytest.approx(observations[1][name], abs=1e-5)
            assert observations[0]["top"].shape == observations[1]["top"].shape == (480, 640, 3)
            assert observations[0]["top"].dtype == observations[1]["top"].dtype == np.uint8
            for robot, observation in zip(robots, observations, strict=True):
                robot.send_action(processor(({"d": 1, "j": 1}, observation)))
            assert robots[0].episode_status == robots[1].episode_status
        for robot in robots:
            assert robot.episode_status.truncated
            terminal = robot.get_observation()
            robot.hold()
            np.testing.assert_array_equal(robot.get_observation()["top"], terminal["top"])
            with pytest.raises(RuntimeError, match="episode ended"):
                robot.send_action(processor(({}, terminal)))
            robot.reset_episode(seed=42)
            assert robot.episode_status.step == 0
    finally:
        for robot in robots:
            robot.disconnect()


@pytest.mark.parametrize("use_async_envs", [False, True])
def test_aloha_batched_single_world_can_render_on_its_owner_thread(use_async_envs):
    pytest.importorskip("gym_aloha")
    envs = make_env(
        AlohaEnv(task="AlohaTransferCube-v0", episode_length=3), n_envs=1, use_async_envs=use_async_envs
    )
    env = envs["AlohaTransferCube-v0"][0]
    try:
        observation, _ = env.reset(seed=42)
        assert observation["observation.images.top"].shape == (1, 480, 640, 3)
        env.step(observation["observation.state"].copy())
    finally:
        env.close()


@pytest.mark.parametrize("remote", [False, True])
def test_aloha_keyboard_recording(aloha_server, tmp_path, monkeypatch, remote):
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.scripts import lerobot_record
    from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig

    teleop = MockTeleop(MockTeleopConfig())
    monkeypatch.setattr(teleop, "get_action", lambda: {"d": 1, "j": 1})
    monkeypatch.setattr(lerobot_record, "make_teleoperator_from_config", lambda _: teleop)
    monkeypatch.setattr(
        lerobot_record,
        "init_keyboard_listener",
        lambda: (None, {"exit_early": False, "stop_recording": False, "rerecord_episode": False}),
    )
    dataset = lerobot_record.record(
        lerobot_record.RecordConfig(
            robot=SimRobotConfig(
                env=AlohaEnv(
                    task="AlohaTransferCube-v0", episode_length=3, endpoint=aloha_server if remote else None
                ),
                calibration_dir=tmp_path / "calibration",
                seed=42,
            ),
            teleop=teleop.config,
            play_sounds=False,
            dataset=DatasetRecordConfig(
                repo_id="local/aloha_smoke",
                root=tmp_path / "dataset",
                fps=50,
                num_episodes=2,
                episode_time_s=10,
                reset_time_s=0,
                video=False,
                push_to_hub=False,
                no_stamp=True,
            ),
        ),
        teleop_action_processor=aloha_keyboard_processor(),
    )
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 6
    assert dataset.features["action"]["names"] == [f"joint_{i}.pos" for i in range(14)]
    np.testing.assert_allclose(dataset[0]["action"][0], 0.01, atol=1e-6)
    np.testing.assert_allclose(dataset[0]["action"][7], -0.01, atol=1e-6)
    assert not (dataset.root / "meta" / "simulation.json").exists()
    if not remote:
        from tests.env_server.native_workflow import train_and_rollout

        train_and_rollout(
            dataset,
            AlohaEnv(task="AlohaTransferCube-v0", episode_length=3),
            aloha_server,
            tmp_path,
            monkeypatch,
        )


@pytest.mark.parametrize("remote", [False, True])
def test_aloha_rtc_executes_absolute_targets_during_delayed_inference(aloha_server, tmp_path, remote):
    from tests.inference.rtc_helpers import assert_delayed_chunk_overlap, sim_rtc

    robot = SimRobot(
        SimRobotConfig(
            env=AlohaEnv(
                task="AlohaTransferCube-v0", episode_length=3, endpoint=aloha_server if remote else None
            ),
            seed=42,
            calibration_dir=tmp_path,
        )
    )
    try:
        robot.connect()
        with sim_rtc(robot) as rig:
            assert_delayed_chunk_overlap(rig, delay_steps=1)
            assert robot.episode_status.truncated
            terminal = robot.get_observation()["top"]
            rig.engine.pause()
            rig.wrapper.hold()
            np.testing.assert_array_equal(robot.get_observation()["top"], terminal)
    finally:
        robot.disconnect()
