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

"""The EnvHub environment of `examples/praxis/sim_env`: the praxis-sim rig behind LeRobot's simulator API.

The tests run on the CPU with a stand-in for the `praxis_sim` package (no MuJoCo, no rendering): they
check what `env.py` adds, the conventions the `praxis` policy needs and the scenes of unseeded resets.
One test reads the real package when it is installed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from lerobot.envs import HubEnvConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import validate_checkpoint
from lerobot.policies.praxis.configuration_praxis import PRAXIS_ACTION_NAMES
from lerobot.robots.sim_robot import SimRobot, SimRobotConfig
from lerobot.sim.integrations import envhub

from .tiny_model import tiny_lerobot_config

SIM_ENV = Path(__file__).resolve().parents[3] / "examples" / "praxis" / "sim_env"
IMAGE = (36, 64, 3)
TASKS = {"lift_cube": "Pick up the red cube.", "move_block": "Move the red block to the {side}."}
SUITES = {"droid_guard": ["lift_cube", "move_block"]}


class StandInSimEnv(gym.Env):
    """What `env.py` relies on in `praxis_sim.gym_env.PraxisSimGymEnv`, without a simulator.

    As in the real environment, the reset seed is the scene seed, a reset without a seed repeats the
    last scene, and some prompts depend on the scene.
    """

    def __init__(self, task: str, *, after_success_s: float = 1.0, time_limit_s: float | None = None) -> None:
        self.task_name = task
        self.after_success_s = after_success_s
        self.task_description = TASKS[task].format(side="right")
        self._max_episode_steps = 30 + round(15 * after_success_s)
        self._scene_seed = 0
        self._ticks = 0
        image = spaces.Box(0, 255, IMAGE, np.uint8)
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict({"exterior": image, "wrist": image}),
                "agent_pos": spaces.Box(-np.inf, np.inf, (7,), np.float32),
                "scene_seed": spaces.Box(0, 2**24, (1,), np.int64),
            }
        )
        self.action_space = spaces.Box(-np.inf, np.inf, (7,), np.float32)

    @property
    def task(self) -> str:
        return self.task_name

    def _observation(self) -> dict:
        image = np.full(IMAGE, self._scene_seed % 256, np.uint8)
        pose = np.array([0.4, 0.0, 0.5, np.pi, 0.0, 0.0, 0.0], np.float32)
        return {
            "pixels": {"exterior": image, "wrist": image},
            "agent_pos": pose,
            "scene_seed": np.array([self._scene_seed], np.int64),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._scene_seed = int(seed) if seed is not None else self._scene_seed
        self._ticks = 0
        self.task_description = TASKS[self.task_name].format(side="left" if self._scene_seed % 2 else "right")
        return self._observation(), {"is_success": False}

    def step(self, action):
        self._ticks += 1
        success = self._ticks >= 3
        info = {
            "is_success": success,
            "progress": min(self._ticks / 3, 1.0),
            "success_tick": 3 if success else -1,
        }
        return self._observation(), info["progress"], self._ticks >= 5, False, info


def _resolve_tasks(task: str) -> tuple[str, list[str]]:
    if task in SUITES:
        return task, list(SUITES[task])
    return "praxis_sim", [name.strip() for name in task.split(",")]


@pytest.fixture
def sim_env(monkeypatch) -> str:
    """The path of `examples/praxis/sim_env`, loaded against the stand-in package."""
    package = types.ModuleType("praxis_sim")
    gym_env = types.ModuleType("praxis_sim.gym_env")
    gym_env.PraxisSimGymEnv = StandInSimEnv
    gym_env.resolve_tasks = _resolve_tasks
    package.gym_env = gym_env
    monkeypatch.setitem(sys.modules, "praxis_sim", package)
    monkeypatch.setitem(sys.modules, "praxis_sim.gym_env", gym_env)
    # env.py binds the package it finds on import: load it anew here, and anew after this test.
    monkeypatch.setattr(envhub, "_MODULES", {})
    return str(SIM_ENV)


def _config(path: str, **make_env_kwargs) -> HubEnvConfig:
    return HubEnvConfig(hub_path=path, trust_remote_code=True, make_env_kwargs=make_env_kwargs)


def _close(envs: dict) -> None:
    for tasks in envs.values():
        for env in tasks.values():
            env.close()


def test_sim_env_has_the_conventions_of_the_policy(sim_env):
    """The rig is the single-arm rig of the policy: its names, its rate, a checkpoint that is admitted."""
    envs = make_env(_config(sim_env), n_envs=1)
    try:
        assert {suite: sorted(tasks) for suite, tasks in envs.items()} == {"droid_guard": [0, 1]}
        descriptor = envs["droid_guard"][0].descriptor
        assert descriptor.fps == 15
        assert descriptor.control_mode == "eef_pose"
        assert descriptor.robot_type == "Franka"
        assert descriptor.action_feature.names == PRAXIS_ACTION_NAMES
        features = {feature.name: feature for feature in descriptor.features}
        assert features["observation.state"].names == PRAXIS_ACTION_NAMES
        policy = tiny_lerobot_config()
        assert {policy.exterior_image_key, policy.wrist_image_key} <= set(features)
        validate_checkpoint(descriptor, policy, {})
    finally:
        _close(envs)


def test_sim_env_takes_a_suite_or_a_list_of_tasks(sim_env):
    envs = make_env(_config(sim_env, suite="move_block,lift_cube", after_success_s=8.0), n_envs=2)
    try:
        assert {suite: sorted(tasks) for suite, tasks in envs.items()} == {"praxis_sim": [0, 1]}
        env = envs["praxis_sim"][0]
        # The other options reach the simulator's environment: here, the time an episode runs after success.
        assert env.call("_max_episode_steps") == (150, 150)
        env.reset(seed=0)
        assert env.call("task_description") == (
            "Move the red block to the right.",
            "Move the red block to the left.",
        )
    finally:
        _close(envs)


def test_eval_seeds_are_the_scene_seeds_and_metrics_reach_the_episode(sim_env):
    """`lerobot-eval` seeds every episode; the scene seed reaches the policy, the task's numbers the report."""
    envs = make_env(_config(sim_env), n_envs=2)
    try:
        env = envs["droid_guard"][1]
        observation, _ = env.reset(seed=[3, 4])
        np.testing.assert_array_equal(observation["observation.scene_seed"], [[3.0], [4.0]])
        assert env.call("task_description") == (
            "Move the red block to the left.",
            "Move the red block to the right.",
        )
        action = observation["observation.state"]
        for _ in range(3):
            _, _, _, _, info = env.step(action)
        assert info["is_success"].tolist() == [True, True]
        assert info["metrics"]["progress"].tolist() == [1.0, 1.0]
        assert info["metrics"]["success_tick"].tolist() == [3.0, 3.0]
        # A seeded reset after an episode is that scene again, not the next one.
        observation, _ = env.reset(seed=[3, 4])
        np.testing.assert_array_equal(observation["observation.scene_seed"], [[3.0], [4.0]])
    finally:
        _close(envs)


def _scene(robot: SimRobot) -> int:
    return int(robot._env.latest.obs["observation.scene_seed"][0, 0])


def test_a_sim_robot_takes_the_next_scene_each_episode(sim_env):
    """A sim robot resets without a seed: each episode that ran is followed by the next scene."""
    robot = SimRobot(SimRobotConfig(env=_config(sim_env, first_scene=5), task_id=1))
    try:
        assert robot.robot_type == "Franka"
        assert tuple(robot.action_features) == PRAXIS_ACTION_NAMES
        assert set(robot.cameras) == {"exterior", "wrist"}
        robot.connect()
        assert _scene(robot) == 5
        # The strategies reset once more before the first episode: the same scene, not the next.
        robot.start_episode()
        assert _scene(robot) == 5
        hold = {name: robot.get_observation()[name] for name in robot.action_features}
        robot.send_action(hold)
        robot.start_episode()
        assert _scene(robot) == 6
        robot.send_action(hold)
        robot.start_episode(seed=0)
        assert _scene(robot) == 0
        robot.send_action(hold)
        robot.start_episode()
        assert _scene(robot) == 1
    finally:
        robot.disconnect()


def test_sim_env_describes_the_real_package(monkeypatch):
    """With praxis-sim installed: the suites and the cameras of the rig (no simulator is built to describe it)."""
    pytest.importorskip("praxis_sim.gym_env")
    monkeypatch.setattr(envhub, "_MODULES", {})
    envs = make_env(_config(str(SIM_ENV)), n_envs=1)
    try:
        assert sorted(envs["droid_guard"]) == list(range(7))
        descriptor = envs["droid_guard"][0].descriptor
        features = {feature.name: feature for feature in descriptor.features}
        assert features["observation.images.exterior"].shape == (288, 512, 3)
        assert features["observation.images.wrist"].shape == (288, 512, 3)
        assert features["observation.state"].names == PRAXIS_ACTION_NAMES
        assert "observation.scene_seed" in features
        validate_checkpoint(descriptor, tiny_lerobot_config(image_shape=(3, 288, 512)), {})
    finally:
        _close(envs)
