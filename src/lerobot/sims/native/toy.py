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

"""Deterministic lightweight environment integration for tests."""

from collections.abc import Callable
from typing import Any

import numpy as np
from gymnasium import spaces

from .utils import canonical_observation

__all__ = ["canonical_observation", "create_envs"]


class ToyEnv:
    """A 2-D point that starts near the origin and succeeds past x = 1."""

    def __init__(self) -> None:
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


def create_envs(
    task: str,
    n_envs: int,
    env_cls: Callable[[list[Callable[[], Any]]], Any],
    **_: Any,
) -> dict[str, dict[int, Any]]:
    """Create the test environment using the standard native-integration contract."""
    if task != "reach":
        raise ValueError(f"Unknown toy task: {task!r}")
    return {"reach": {0: env_cls([ToyEnv for _ in range(n_envs)])}}
