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

"""A deterministic toy simulator integration and its env config, for local and served execution tests."""

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from lerobot.envs.configs import EnvConfig
from lerobot.sim.integrations.utils import canonical_observation

__all__ = [
    "DictObservationEnv",
    "ToyEnv",
    "ToyEnvConfig",
    "ToyVectorEnv",
    "batched_env_fns",
    "canonical_observation",
    "env_fns",
]


class ToyEnv:
    """A 2-D point that starts near the origin and succeeds past x = 1."""

    builds = 0

    def __init__(self) -> None:
        ToyEnv.builds += 1
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(-100, 100, (2,), np.float32),
                "pixels": spaces.Box(0, 255, (8, 8, 3), np.uint8),
            }
        )
        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        self._max_episode_steps = 5
        self.task = "reach"
        self.task_description = "Move to the target"
        self.state: np.ndarray = np.zeros(2, dtype=np.float32)

    def _obs(self) -> dict[str, np.ndarray]:
        return {"agent_pos": self.state.copy(), "pixels": self.render()}

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict]:
        """Place the point near the origin, from ``seed``."""
        self.state = np.random.default_rng(seed).uniform(-0.1, 0.1, 2).astype(np.float32)
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Move the point by ``action``."""
        self.state += action
        success = bool(self.state[0] >= 1)
        return self._obs(), float(success), success, False, {"is_success": success}

    def render(self) -> np.ndarray:
        """Return an image whose brightness encodes x."""
        return np.full((8, 8, 3), int(np.clip(self.state[0] * 10 + 100, 0, 255)), dtype=np.uint8)

    def close(self) -> None:
        """Nothing to release."""


def env_fns(
    task: str,
    n_envs: int,
    task_ids: tuple[int, ...] = (0,),
    **_: Any,
) -> dict[str, dict[int, list[Callable[[], Any]]]]:
    """The toy environments' factories, as a simulator integration returns them."""
    if task != "reach":
        raise ValueError(f"Unknown toy task: {task!r}")
    return {"reach": {task_id: [ToyEnv] * n_envs for task_id in task_ids}}


class ToyVectorEnv(gym.vector.VectorEnv):
    """``ToyEnv`` for a whole batch at once, as a GPU simulator steps it; it keeps stepping finished envs."""

    def __init__(self, num_envs: int, autoreset_mode: str = "DISABLED") -> None:
        self.num_envs = num_envs
        self.metadata = {"autoreset_mode": gym.vector.AutoresetMode[autoreset_mode]}
        self.single_observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(-100, 100, (2,), np.float32),
                "pixels": spaces.Box(0, 255, (8, 8, 3), np.uint8),
            }
        )
        self.single_action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self._max_episode_steps = 5
        self.task_description = "Move to the target"
        self.state: np.ndarray = np.zeros((num_envs, 2), dtype=np.float32)

    def _obs(self) -> dict[str, np.ndarray]:
        brightness = np.clip(self.state[:, 0] * 10 + 100, 0, 255).astype(np.uint8)
        pixels = np.broadcast_to(brightness[:, None, None, None], (self.num_envs, 8, 8, 3))
        return {"agent_pos": self.state.copy(), "pixels": pixels.copy()}

    def reset(self, *, seed: list[int | None] | None = None, options: dict | None = None):
        """Place each point near the origin, from its seed, as ``ToyEnv`` does."""
        seeds = seed if seed is not None else [None] * self.num_envs
        self.state = np.stack([np.random.default_rng(s).uniform(-0.1, 0.1, 2) for s in seeds]).astype(
            np.float32
        )
        return self._obs(), {}

    def step(self, actions: np.ndarray):
        """Move every point, finished ones included."""
        self.state += actions
        success = self.state[:, 0] >= 1
        return (
            self._obs(),
            success.astype(np.float32),
            success,
            np.zeros(self.num_envs, bool),
            {"is_success": success},
        )


def batched_env_fns(
    task: str,
    n_envs: int,
    task_ids: tuple[int, ...] = (0,),
    autoreset_mode: str = "DISABLED",
    **_: Any,
) -> dict[str, dict[int, Callable[[], Any]]]:
    """The batched toy environments' factories, as a GPU simulator integration returns them."""
    if task != "reach":
        raise ValueError(f"Unknown toy task: {task!r}")
    return {"reach": {task_id: partial(ToyVectorEnv, n_envs, autoreset_mode) for task_id in task_ids}}


class DictObservationEnv(gym.Env):
    """A Gymnasium env with the dict observations LeRobot environments provide, for gym.make() registration."""

    observation_space = spaces.Dict({"agent_pos": spaces.Box(-1, 1, (2,), np.float32)})
    action_space = spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return {"agent_pos": np.zeros(2, np.float32)}, {}

    def step(self, action):
        return {"agent_pos": np.zeros(2, np.float32)}, 0.0, False, False, {}


# Simulator integrations are imported as `lerobot.sim.integrations.<type>`; the toys are registered there for tests.
sys.modules.setdefault("lerobot.sim.integrations.toy", sys.modules[__name__])
toy_vector = types.ModuleType("lerobot.sim.integrations.toy_vector")
toy_vector.vector_env_fns = batched_env_fns
toy_vector.canonical_observation = canonical_observation
sys.modules.setdefault("lerobot.sim.integrations.toy_vector", toy_vector)


@EnvConfig.register_subclass("toy")
@dataclass
class ToyEnvConfig(EnvConfig):
    fps: int = 20
    robot_type: str | None = None
    # The suite's task ids, as LIBERO's --env.task_ids selects them.
    task_ids: tuple[int, ...] = (0,)

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def sim_settings(self) -> dict:
        return {
            "type": self.type,
            "task": "reach",
            "fps": self.fps,
            "control_mode": "delta",
            "semantics": "toy-delta-v1",
            "robot_type": self.robot_type,
            "kwargs": {"task_ids": list(self.task_ids)},
        }
