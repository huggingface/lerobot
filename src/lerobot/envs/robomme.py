"""RoboMME environment wrapper for LeRobot evaluation.

Wraps the RoboMME ``BenchmarkEnvBuilder`` into a Gymnasium-compatible
``VectorEnv`` suitable for ``lerobot_eval``.

RoboMME tasks:
  Counting:    BinFill, PickXtimes, SwingXtimes, StopCube
  Permanence:  VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap
  Reference:   PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder
  Imitation:   MoveCube, InsertPeg, PatternLock, RouteStick

Dataset: pepijn223/robomme_data_lerobot (LeRobot v3.0, 1,600 episodes)
Install: pip install 'lerobot[robomme]'  (Linux only)
Benchmark: https://github.com/RoboMME/robomme_benchmark
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.envs.lazy_vec_env import LazyVectorEnv

ROBOMME_TASKS = [
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
]


class RoboMMEGymEnv(gym.Env):
    """Thin Gymnasium wrapper around a single RoboMME episode env."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task: str = "PickXtimes",
        action_space_type: str = "joint_angle",
        dataset: str = "test",
        episode_idx: int = 0,
        max_steps: int = 300,
    ):
        super().__init__()
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        self._task = task
        self._action_space_type = action_space_type
        self._dataset = dataset
        self._episode_idx = episode_idx
        self._max_steps = max_steps

        self._builder = BenchmarkEnvBuilder(
            env_id=task,
            dataset=dataset,
            action_space=action_space_type,
            gui_render=False,
            max_steps=max_steps,
        )
        self._env = None

        action_dim = 8 if action_space_type == "joint_angle" else 7
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "front_rgb": spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8),
                "wrist_rgb": spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._env = self._builder.make_env_for_episode(
            episode_idx=self._episode_idx,
            max_steps=self._max_steps,
        )
        obs, info = self._env.reset()
        return self._convert_obs(obs), self._convert_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)

        terminated_bool = bool(terminated.item()) if hasattr(terminated, "item") else bool(terminated)
        truncated_bool = bool(truncated.item()) if hasattr(truncated, "item") else bool(truncated)

        status = info.get("status", "ongoing")
        is_success = status == "success"
        conv_info = self._convert_info(info)
        conv_info["is_success"] = is_success

        return self._convert_obs(obs), float(reward), terminated_bool, truncated_bool, conv_info

    def _convert_obs(self, obs: dict) -> dict:
        front_rgb = (
            obs["front_rgb_list"][-1] if isinstance(obs["front_rgb_list"], list) else obs["front_rgb_list"]
        )
        wrist_rgb = (
            obs["wrist_rgb_list"][-1] if isinstance(obs["wrist_rgb_list"], list) else obs["wrist_rgb_list"]
        )
        joint_state = (
            obs["joint_state_list"][-1]
            if isinstance(obs["joint_state_list"], list)
            else obs["joint_state_list"]
        )
        gripper_state = (
            obs["gripper_state_list"][-1]
            if isinstance(obs["gripper_state_list"], list)
            else obs["gripper_state_list"]
        )

        front_rgb = np.asarray(front_rgb, dtype=np.uint8)
        wrist_rgb = np.asarray(wrist_rgb, dtype=np.uint8)
        joint = np.asarray(joint_state, dtype=np.float32).flatten()[:7]
        gripper = np.asarray(gripper_state, dtype=np.float32).flatten()[:1]
        state = np.concatenate([joint, gripper])

        return {
            "front_rgb": front_rgb,
            "wrist_rgb": wrist_rgb,
            "state": state,
        }

    def _convert_info(self, info: dict) -> dict:
        return {
            "status": info.get("status", "ongoing"),
            "task_goal": info.get("task_goal", ""),
        }


def _make_env_fns(
    *,
    task: str,
    n_envs: int,
    action_space_type: str,
    dataset: str,
    episode_length: int,
    task_id: int,
) -> list[Callable[[], RoboMMEGymEnv]]:
    """Build n_envs factory callables for one RoboMME task id."""

    def _make_one(episode_index: int) -> RoboMMEGymEnv:
        return RoboMMEGymEnv(
            task=task,
            action_space_type=action_space_type,
            dataset=dataset,
            episode_idx=episode_index,
            max_steps=episode_length,
        )

    return [partial(_make_one, task_id + i) for i in range(n_envs)]


def create_robomme_envs(
    task: str,
    n_envs: int = 1,
    action_space_type: str = "joint_angle",
    dataset: str = "test",
    episode_length: int = 300,
    task_ids: list[int] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create vectorized RoboMME environments for evaluation.

    Returns {suite_name: {task_id: VectorEnv}} matching lerobot's expected format.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of env factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    if task_ids is None:
        task_ids = [0]

    suite_name = "robomme"
    envs_by_task = {}
    lazy = len(task_ids) > 50
    if lazy:
        print(f"Using lazy env creation for {len(task_ids)} tasks (envs created on demand)")

    for task_id in task_ids:
        fns = _make_env_fns(
            task=task,
            n_envs=n_envs,
            action_space_type=action_space_type,
            dataset=dataset,
            episode_length=episode_length,
            task_id=task_id,
        )
        envs_by_task[task_id] = LazyVectorEnv(env_cls, fns) if lazy else env_cls(fns)

    return {suite_name: envs_by_task}
