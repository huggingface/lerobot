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

"""Local Gymnasium vector environment for LeRobot simulator providers."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import batch_space

from lerobot.sims.backend import Backend, BackendConfig
from lerobot.sims.types import StepResult


class VectorEnv(gym.vector.VectorEnv):
    """Run a batch of simulator environments directly, without deployment machinery."""

    canonical_observations = True

    def __init__(
        self,
        config: BackendConfig,
        num_envs: int = 1,
        task_group: str | None = None,
        task_id: int = 0,
        use_async_envs: bool = False,
    ) -> None:
        """Create a local vector environment backed by the selected simulator."""
        self.backend = Backend(config, num_envs, task_group, task_id, use_async_envs)
        self.num_envs = num_envs
        self.task_group = self.backend.group
        self.task_id = task_id
        self.single_observation_space, self.single_action_space = self.backend.single_spaces()
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.metadata = {
            "render_fps": config.fps,
            "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
        }
        self.render_mode = "rgb_array"
        self.latest: StepResult | None = None
        self.closed = False

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset every environment and return its initial observation."""
        if options:
            raise ValueError("This environment does not define reset options")
        seeds: list[int | None] | None = (
            [seed + index for index in range(self.num_envs)] if isinstance(seed, int) else seed
        )
        self.latest = self.backend.reset(seeds)
        return self.latest.obs, self._info(self.latest)

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        """Advance every environment by one simulation step."""
        self.latest = self.backend.step(np.asarray(actions, dtype=np.float32))
        return (
            self.latest.obs,
            self.latest.reward,
            self.latest.terminated,
            self.latest.truncated,
            self._info(self.latest),
        )

    def render(self) -> list[np.ndarray]:
        """Render every environment."""
        return self.backend.render()

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Return a supported per-environment attribute."""
        if args or kwargs:
            raise TypeError("Environment attributes do not take arguments")
        if name == "_max_episode_steps":
            return (self.backend.max_episode_steps,) * self.num_envs
        if name == "task":
            return (self.task_group,) * self.num_envs
        if name == "task_description":
            if self.latest is None:
                raise RuntimeError("Reset before reading the task description")
            return self.latest.task
        if name == "render":
            return tuple(self.render())
        raise AttributeError(f"Environment attribute is not exposed: {name}")

    def get_attr(self, name: str) -> tuple[Any, ...]:
        """Return a supported per-environment attribute."""
        return self.call(name)

    def close_extras(self, **kwargs: Any) -> None:
        """Release simulator resources."""
        self.backend.close()
        self.closed = True

    @staticmethod
    def _info(result: StepResult) -> dict[str, Any]:
        return {
            "is_success": result.is_success,
            "task": result.task,
            "step": result.step,
            "sim_time": result.sim_time,
        }
