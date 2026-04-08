#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import importlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.types import RobotObservation

# Camera names as used by RoboTwin 2.0. The wrapper appends "_rgb" when looking
# up keys in get_obs() output (e.g. "head_camera" → "head_camera_rgb").
ROBOTWIN_CAMERA_NAMES: tuple[str, ...] = (
    "head_camera",
    "front_camera",
    "left_wrist",
    "right_wrist",
)

ACTION_DIM = 14  # 7 DOF × 2 arms
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
DEFAULT_EPISODE_LENGTH = 300

# Complete task list from RoboTwin 2.0 (60 tasks, as listed on the leaderboard).
ROBOTWIN_TASKS: tuple[str, ...] = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "close_laptop",
    "close_microwave",
    "dump_bin",
    "grab_roller",
    "handover_block",
    "handover_cup",
    "handover_diverse_bottles",
    "handover_mic",
    "hanging_mug",
    "insert_pin",
    "lift_pot",
    "make_tea",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_basket",
    "place_block",
    "place_cable",
    "place_can",
    "place_chopsticks",
    "place_cloth",
    "place_container",
    "place_cup",
    "place_diverse_bottles",
    "place_dual_bottles",
    "place_fork",
    "place_knife",
    "place_object_basket",
    "place_ring",
    "place_ruler",
    "place_shoes_left",
    "place_shoes_right",
    "place_spoon",
    "place_toy",
    "pour_water",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "put_shoes_box",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
    "wipe_board",
    "arrange_tools",
    "build_tower",
    "fold_cloth",
)


def _load_robotwin_task(task_name: str) -> type:
    """Dynamically import and return a RoboTwin 2.0 task class.

    RoboTwin tasks live in ``envs/<task_name>.py`` relative to the repository
    root and are expected to be on ``sys.path`` after installation.
    """
    try:
        module = importlib.import_module(f"envs.{task_name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Could not import RoboTwin task '{task_name}'. "
            "Ensure RoboTwin 2.0 is installed and its 'envs/' directory is on PYTHONPATH. "
            "See the RoboTwin installation guide: https://robotwin-platform.github.io/doc/usage/robotwin-install.html"
        ) from e
    task_cls = getattr(module, task_name, None)
    if task_cls is None:
        raise AttributeError(f"Task class '{task_name}' not found in envs/{task_name}.py")
    return task_cls


class RoboTwinEnv(gym.Env):
    """Gymnasium wrapper around a single RoboTwin 2.0 task.

    RoboTwin uses a custom SAPIEN-based API (``setup_demo`` / ``get_obs`` /
    ``take_action`` / ``check_success``) rather than the standard gym interface.
    This class bridges that API to Gymnasium so that ``lerobot-eval`` can drive
    RoboTwin exactly like LIBERO or Meta-World.

    The underlying SAPIEN environment is created lazily on the first ``reset()``
    call *inside the worker process*.  This is required for
    ``gym.vector.AsyncVectorEnv`` compatibility: SAPIEN allocates EGL/GPU
    contexts that must not be forked from the parent process.

    Observations
    ------------
    The ``pixels`` dict uses the raw RoboTwin camera names as keys (e.g.
    ``"head_camera"``, ``"left_wrist"``).  ``preprocess_observation`` in
    ``envs/utils.py`` then converts these to ``observation.images.<cam>``.

    Actions
    -------
    14-dim float32 array in ``[-1, 1]`` (joint-space, 7 DOF per arm).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        task_name: str,
        episode_index: int = 0,
        n_envs: int = 1,
        camera_names: Sequence[str] = ROBOTWIN_CAMERA_NAMES,
        observation_height: int = 480,
        observation_width: int = 640,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
        render_mode: str = "rgb_array",
    ):
        super().__init__()
        self.task_name = task_name
        self.task = task_name  # used by add_envs_task() in utils.py
        self.task_description = task_name.replace("_", " ")
        self.episode_index = episode_index
        self._reset_stride = n_envs
        self.camera_names = list(camera_names)
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.episode_length = episode_length
        self.render_mode = render_mode

        self._env: Any | None = None  # deferred — created on first reset() inside worker
        self._step_count: int = 0

        image_spaces = {
            cam: spaces.Box(
                low=0,
                high=255,
                shape=(observation_height, observation_width, 3),
                dtype=np.uint8,
            )
            for cam in self.camera_names
        }
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(image_spaces),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(ACTION_DIM,), dtype=np.float64),
            }
        )
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def _ensure_env(self) -> None:
        """Create the SAPIEN environment on first use.

        Called inside the worker subprocess after fork(), so each worker gets
        its own EGL/GPU context rather than inheriting a stale one from the
        parent process (which causes crashes with AsyncVectorEnv).
        """
        if self._env is not None:
            return
        task_cls = _load_robotwin_task(self.task_name)
        self._env = task_cls()

    def _get_obs(self) -> RobotObservation:
        assert self._env is not None, "_get_obs called before _ensure_env()"
        raw = self._env.get_obs()

        images: dict[str, np.ndarray] = {}
        for cam in self.camera_names:
            # RoboTwin 2.0 camera keys follow the "<name>_rgb" convention.
            # Fall back to the bare name if the suffixed key is absent.
            key = f"{cam}_rgb" if f"{cam}_rgb" in raw else cam
            if key in raw:
                img = np.asarray(raw[key], dtype=np.uint8)
                # Ensure HWC and exactly 3 channels
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                elif img.shape[-1] != 3:
                    img = img[..., :3]
                if img.shape[0] != self.observation_height or img.shape[1] != self.observation_width:
                    # Resize is best done outside hot-loop; return as-is and let
                    # the processor handle it if needed.
                    pass
                images[cam] = img
            else:
                # Camera not exposed by this task — return a black frame so the
                # observation shape stays consistent across all tasks.
                images[cam] = np.zeros((self.observation_height, self.observation_width, 3), dtype=np.uint8)

        # Joint state: try common key names used by RoboTwin base_task.
        joint_state: np.ndarray | None = None
        for key in ("joint_action", "qpos", "robot_qpos", "joint_state", "obs"):
            if key in raw:
                arr = np.asarray(raw[key], dtype=np.float64).flatten()
                if arr.size >= ACTION_DIM:
                    joint_state = arr[:ACTION_DIM]
                    break
        if joint_state is None:
            joint_state = np.zeros(ACTION_DIM, dtype=np.float64)

        return {"pixels": images, "agent_pos": joint_state}

    def reset(self, seed: int | None = None, **kwargs) -> tuple[RobotObservation, dict]:
        self._ensure_env()
        super().reset(seed=seed)
        assert self._env is not None  # set by _ensure_env() above

        actual_seed = self.episode_index if seed is None else seed
        self._env.setup_demo(seed=actual_seed, is_test=True)
        self.episode_index += self._reset_stride
        self._step_count = 0

        obs = self._get_obs()
        return obs, {"is_success": False, "task": self.task_name}

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        assert self._env is not None, "step() called before reset()"
        if action.ndim != 1 or action.shape[0] != ACTION_DIM:
            raise ValueError(f"Expected 1-D action of shape ({ACTION_DIM},), got {action.shape}")

        # RoboTwin 2.0 uses take_action(); fall back to step() for older forks.
        if hasattr(self._env, "take_action"):
            self._env.take_action(action)
        else:
            self._env.step(action)

        self._step_count += 1

        is_success = bool(getattr(self._env, "eval_success", False))
        if not is_success and hasattr(self._env, "check_success"):
            is_success = bool(self._env.check_success())

        obs = self._get_obs()
        reward = float(is_success)
        terminated = is_success
        truncated = self._step_count >= self.episode_length

        info: dict[str, Any] = {
            "task": self.task_name,
            "is_success": is_success,
            "step": self._step_count,
        }
        if terminated or truncated:
            info["final_info"] = {
                "task": self.task_name,
                "is_success": is_success,
            }
            self.reset()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        self._ensure_env()
        obs = self._get_obs()
        # Prefer head camera for rendering; fall back to first available.
        if "head_camera" in obs["pixels"]:
            return obs["pixels"]["head_camera"]
        return next(iter(obs["pixels"].values()))

    def close(self) -> None:
        if self._env is not None:
            if hasattr(self._env, "close_env"):
                self._env.close_env(clearance=False)
            self._env = None


# ---- Multi-task factory --------------------------------------------------------


def _make_env_fns(
    *,
    task_name: str,
    n_envs: int,
    camera_names: list[str],
    observation_height: int,
    observation_width: int,
    episode_length: int,
) -> list[Callable[[], RoboTwinEnv]]:
    """Return n_envs factory callables for a single task."""

    def _make_one(episode_index: int) -> RoboTwinEnv:
        return RoboTwinEnv(
            task_name=task_name,
            episode_index=episode_index,
            n_envs=n_envs,
            camera_names=camera_names,
            observation_height=observation_height,
            observation_width=observation_width,
            episode_length=episode_length,
        )

    return [partial(_make_one, i) for i in range(n_envs)]


def create_robotwin_envs(
    task: str,
    n_envs: int,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    camera_names: Sequence[str] = ROBOTWIN_CAMERA_NAMES,
    observation_height: int = 480,
    observation_width: int = 640,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
) -> dict[str, dict[int, Any]]:
    """Create vectorized RoboTwin 2.0 environments.

    Returns:
        ``dict[task_name][0] -> VectorEnv`` — one entry per task, each wrapping
        ``n_envs`` parallel rollouts.

    Args:
        task: Comma-separated list of task names (e.g. ``"beat_block_hammer"``
            or ``"beat_block_hammer,click_bell"``).
        n_envs: Number of parallel rollouts per task.
        env_cls: Vector env constructor (e.g. ``gym.vector.AsyncVectorEnv``).
        camera_names: Cameras to include in observations.
        observation_height: Pixel height for all cameras.
        observation_width: Pixel width for all cameras.
        episode_length: Max steps before truncation.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be callable (e.g. gym.vector.AsyncVectorEnv).")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    task_names = [t.strip() for t in str(task).split(",") if t.strip()]
    if not task_names:
        raise ValueError("`task` must contain at least one RoboTwin task name.")

    unknown = [t for t in task_names if t not in ROBOTWIN_TASKS]
    if unknown:
        raise ValueError(f"Unknown RoboTwin tasks: {unknown}. Available tasks: {sorted(ROBOTWIN_TASKS)}")

    print(f"Creating RoboTwin envs | tasks={task_names} | n_envs(per task)={n_envs}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for task_name in task_names:
        fns = _make_env_fns(
            task_name=task_name,
            n_envs=n_envs,
            camera_names=list(camera_names),
            observation_height=observation_height,
            observation_width=observation_width,
            episode_length=episode_length,
        )
        out[task_name][0] = env_cls(fns)
        print(f"Built vec env | task={task_name} | n_envs={n_envs}")

    return {k: dict(v) for k, v in out.items()}
