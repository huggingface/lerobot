from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
import torch


class IsaacLabArenaEnvWrapper(gym.vector.AsyncVectorEnv):
    """Wrapper adapting IsaacLab batched GPU env to AsyncVectorEnv.
    IsaacLab handles vectorization internally on GPU. We inherit from
    AsyncVectorEnv for compatibility with LeRobot."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env,
        episode_length: int = 500,
        task: str | None = None,
        render_mode: str | None = "rgb_array",
        simulation_app=None,
    ):
        self._env = env
        self._num_envs = env.num_envs
        self._episode_length = episode_length
        self._closed = False
        self.render_mode = render_mode
        self._simulation_app = simulation_app

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.task = task

        if hasattr(env, "metadata") and env.metadata:
            self.metadata = {**self.metadata, **env.metadata}

    @property
    def unwrapped(self):
        return self

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def _max_episode_steps(self) -> int:
        return self._episode_length

    @property
    def device(self) -> str:
        return getattr(self._env, "device", "cpu")

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset all environments."""
        # IsaacLab expects a single seed
        if isinstance(seed, (list, tuple, range)):
            seed = seed[0] if len(seed) > 0 else None

        obs, info = self._env.reset(seed=seed, options=options)

        if "final_info" not in info:
            zeros = np.zeros(self._num_envs, dtype=bool)
            info["final_info"] = {"is_success": zeros}

        return obs, info

    def step(
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self._env.device)

        obs, reward, terminated, truncated, info = self._env.step(actions)

        # Convert to numpy for gym compatibility
        reward = reward.cpu().numpy().astype(np.float32)
        terminated = terminated.cpu().numpy().astype(bool)
        truncated = truncated.cpu().numpy().astype(bool)

        is_success = self._get_success(terminated, truncated)
        info["final_info"] = {"is_success": is_success}

        return obs, reward, terminated, truncated, info

    def _get_success(self, terminated: np.ndarray, truncated: np.ndarray) -> np.ndarray:
        is_success = np.zeros(self._num_envs, dtype=bool)

        if not hasattr(self._env, "termination_manager"):
            return is_success & (terminated | truncated)

        term_manager = self._env.termination_manager
        if not hasattr(term_manager, "get_term"):
            return is_success & (terminated | truncated)

        success_tensor = term_manager.get_term("success")
        if success_tensor is None:
            return is_success & (terminated | truncated)

        is_success = success_tensor.cpu().numpy().astype(bool)

        return is_success & (terminated | truncated)

    def call(self, method_name: str, *args, **kwargs) -> list[Any]:
        if method_name == "_max_episode_steps":
            return [self._episode_length] * self._num_envs
        if method_name == "task":
            return [self.task] * self._num_envs
        if method_name == "render":
            return self.render_all()

        if hasattr(self._env, method_name):
            attr = getattr(self._env, method_name)
            result = attr(*args, **kwargs) if callable(attr) else attr
            if isinstance(result, list):
                return result
            return [result] * self._num_envs

        raise AttributeError(f"IsaacLab-Arena has no method/attribute '{method_name}'")

    def render_all(self) -> list[np.ndarray]:
        """Render all environments and return list of frames."""
        frames = self.render()
        if frames is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            return [placeholder] * self._num_envs

        if frames.ndim == 4:
            return [frames[i] for i in range(min(len(frames), self._num_envs))]

        return [np.zeros((480, 640, 3), dtype=np.uint8)] * self._num_envs

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None

        frames = self._env.render() if hasattr(self._env, "render") else None
        if frames is None:
            return None

        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        return frames[0] if frames.ndim == 4 else frames

    def close(self) -> None:
        if not self._closed:
            logging.info("Closing IsaacLab environment...")
            self._env.close()
            if self._simulation_app is not None:
                self._simulation_app.app.close()
            self._closed = True

    @property
    def envs(self) -> list[IsaacLabArenaEnvWrapper]:
        return [self] * self._num_envs

    def __del__(self):
        self.close()
