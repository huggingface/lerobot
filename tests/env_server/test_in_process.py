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

"""Environments driven in this process behave as the same environments behind the env server."""

import time
from dataclasses import dataclass

import numpy as np
import pytest

from lerobot.env_server.client import EnvClient
from lerobot.env_server.in_process import InProcessEnvClient
from lerobot.envs.configs import EnvConfig
from lerobot.robots.sim import SimRobot, SimRobotConfig
from lerobot.sims.backend import BackendConfig


@EnvConfig.register_subclass("toy_test")
@dataclass
class ToyEnvConfig(EnvConfig):
    fps: int = 20

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def sim_settings(self) -> dict:
        return {
            "type": "toy",
            "task": "reach",
            "fps": self.fps,
            "control": "delta",
            "semantics": "toy-delta-v1",
        }


def _episode(client: EnvClient | InProcessEnvClient) -> list[dict]:
    try:
        client.open(2, "lockstep", seeds=[10, 20])
        results = [client.request("reset", seeds=[10, 20])["result"]]
        for actions in ([[2, 0], [0, 0]], [[0.1, 0.1], [0.1, 0.1]], [[0.5, 0], [0.5, 0]]):
            results.append(client.request("step", actions=np.array(actions, np.float32))["result"])
        return results
    finally:
        client.close()


def test_in_process_client_matches_the_server(server):
    served = _episode(EnvClient(server[1]))
    local = _episode(InProcessEnvClient(BackendConfig()))
    for expected, actual in zip(served, local, strict=True):
        for key in ("reward", "terminated", "truncated", "is_success", "step"):
            np.testing.assert_array_equal(actual[key], expected[key])
        for name, value in expected["obs"].items():
            np.testing.assert_array_equal(actual["obs"][name], value)
        assert tuple(actual["task"]) == tuple(expected["task"])


def test_in_process_realtime_world_keeps_ticking_while_idle(tmp_path):
    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(), calibration_dir=tmp_path))
    try:
        robot.connect()
        robot.send_action(dict.fromkeys(robot.action_features, 0.0))
        commanded = robot.episode_status.step
        # 20 fps: the world advances on its own while no new command arrives, as hardware does.
        time.sleep(0.15)
        robot.get_observation()
        assert robot.episode_status.step > commanded
    finally:
        robot.disconnect()


@pytest.mark.parametrize("where", ["in_process", "served"])
def test_lockstep_sim_robot_advances_once_per_action(where, request, tmp_path):
    if where == "served":
        config = SimRobotConfig(endpoint=request.getfixturevalue("server")[1], clock="lockstep")
    else:
        config = SimRobotConfig(env=ToyEnvConfig(), clock="lockstep")
    config.calibration_dir = tmp_path
    robot = SimRobot(config)
    try:
        robot.connect()
        action = dict.fromkeys(robot.action_features, 0.1)
        assert robot.send_action(action) == pytest.approx(action)
        assert robot.episode_status.step == 1
        state = [name for name, kind in robot.observation_features.items() if kind is float]
        before = robot.get_observation()
        time.sleep(0.1)
        robot.hold()
        after = robot.get_observation()
        assert [after[name] for name in state] == [before[name] for name in state]
        assert robot.episode_status.step == 1
        robot.send_action(action)
        assert robot.episode_status.step == 2
        robot.reset_world(seed=3)
        assert robot.episode_status.step == 0
    finally:
        robot.disconnect()


def test_sim_robot_config_needs_an_env_or_an_endpoint():
    with pytest.raises(ValueError, match="--robot.env.type"):
        SimRobotConfig()
    with pytest.raises(ValueError, match="--robot.clock"):
        SimRobotConfig(env=ToyEnvConfig(), clock="fast")


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
        args = ["--robot.type=sim", "--robot.env.type=toy_test", "--robot.clock=lockstep"]
        cfg = draccus.parse(Wrapper, args=args)
    finally:
        sys.setrecursionlimit(limit)
    assert isinstance(cfg.robot.env, ToyEnvConfig) and cfg.robot.clock == "lockstep"
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


@pytest.mark.parametrize("where", ["in_process", "served"])
def test_env_config_creates_batched_envs_in_process_or_served(where):
    from lerobot.envs.batched import BatchedEnv

    config = ToyEnvConfig()
    server = thread = None
    if where == "served":
        server, thread, config.endpoint = _served(config.type)
    try:
        envs = config.create_envs(n_envs=2)
        env = envs["reach"][0]
        assert isinstance(env, BatchedEnv)
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


def test_worlds_in_their_own_processes_match_sequential_worlds():
    from lerobot.sims.backend import Backend

    results = []
    for async_worlds in (False, True):
        backend = Backend(BackendConfig(), num_envs=3, async_worlds=async_worlds)
        try:
            steps = [backend.reset([4, 5, 6])]
            for actions in ([[2, 0], [0.1, 0], [0, 0]], [[0.3, 0.1]] * 3):
                steps.append(backend.step(np.array(actions, np.float32)))
            results.append(steps)
        finally:
            backend.close()
    for sequential, parallel in zip(*results, strict=True):
        np.testing.assert_array_equal(parallel.obs["observation.state"], sequential.obs["observation.state"])
        np.testing.assert_array_equal(parallel.terminated, sequential.terminated)
        np.testing.assert_array_equal(parallel.step, sequential.step)
        assert parallel.task == sequential.task


def test_teleop_needs_a_realtime_sim_robot():
    from lerobot.robots.sim import check_teleop_clock

    check_teleop_clock(SimRobotConfig(env=ToyEnvConfig()), teleop=object())
    with pytest.raises(ValueError, match="--robot.clock=realtime"):
        check_teleop_clock(SimRobotConfig(env=ToyEnvConfig(), clock="lockstep"), teleop=object())


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


def test_recording_a_sim_robot_ends_the_episode_with_the_world(tmp_path):
    from lerobot.scripts.lerobot_record import record_loop
    from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig

    robot = SimRobot(SimRobotConfig(env=ToyEnvConfig(), calibration_dir=tmp_path))
    teleop = MockTeleop(MockTeleopConfig(n_motors=2))
    try:
        robot.connect()
        teleop.connect()
        start = time.perf_counter()
        record_loop(
            robot=robot,
            events={"exit_early": False},
            fps=20,
            # The toy succeeds once x passes 1: one step of 2 ends the episode.
            teleop_action_processor=lambda _: {"command_0": 2.0, "command_1": 0.0},
            robot_action_processor=lambda transition: transition[0],
            robot_observation_processor=lambda obs: obs,
            teleop=teleop,
            control_time_s=30,
        )
        assert time.perf_counter() - start < 5
        assert robot.episode_status.terminated
    finally:
        teleop.disconnect()
        robot.disconnect()
