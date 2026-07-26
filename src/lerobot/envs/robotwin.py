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
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from lerobot.types import RobotObservation

from .utils import _LazyAsyncVectorEnv

logger = logging.getLogger(__name__)

# Camera names as used by RoboTwin 2.0. The wrapper appends "_rgb" when looking
# up keys in get_obs() output (e.g. "head_camera" → "head_camera_rgb").
ROBOTWIN_CAMERA_NAMES: tuple[str, ...] = (
    "head_camera",
    "left_camera",
    "right_camera",
)

ACTION_DIM = 14  # 7 DOF × 2 arms
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
DEFAULT_EPISODE_LENGTH = 300
# D435 dims from task_config/_camera_config.yml (what demo_clean.yml selects).
DEFAULT_CAMERA_H = 240
DEFAULT_CAMERA_W = 320

# Task list from RoboTwin 2.0's `envs/` directory — mirrors upstream exactly
# (50 tasks as of main; earlier revisions had 60 with a different split).
# Keep this in sync with:
#   gh api /repos/RoboTwin-Platform/RoboTwin/contents/envs --paginate \
#     | jq -r '.[].name' | grep -E '\.py$' | grep -v '^_' | sed 's/\.py$//'
ROBOTWIN_TASKS: tuple[str, ...] = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
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
)


_ROBOTWIN_SETUP_CACHE: dict[str, dict[str, Any]] = {}


def _load_robotwin_setup_kwargs(task_name: str) -> dict[str, Any]:
    """Build the kwargs dict RoboTwin's setup_demo expects.

    Mirrors the config loading done by RoboTwin's ``script/eval_policy.py``:
    reads ``task_config/demo_clean.yml``, resolves the embodiment file from
    ``_embodiment_config.yml``, loads the robot's own ``config.yml``, and
    reads camera dimensions from ``_camera_config.yml``.

    Uses ``aloha-agilex`` single-robot dual-arm by default (the only embodiment
    used by beat_block_hammer and most smoke-test tasks).
    """
    if task_name in _ROBOTWIN_SETUP_CACHE:
        return dict(_ROBOTWIN_SETUP_CACHE[task_name])

    import os

    import yaml  # type: ignore[import-untyped]
    from envs import CONFIGS_PATH  # type: ignore[import-not-found]

    task_config = "demo_clean"
    with open(os.path.join(CONFIGS_PATH, f"{task_config}.yml"), encoding="utf-8") as f:
        args = yaml.safe_load(f)

    # Resolve embodiment — demo_clean.yml uses [aloha-agilex] (dual-arm single robot)
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)
    embodiment = args.get("embodiment", ["aloha-agilex"])
    if len(embodiment) == 1:
        robot_file = embodiment_types[embodiment[0]]["file_path"]
        args["left_robot_file"] = robot_file
        args["right_robot_file"] = robot_file
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"] = embodiment_types[embodiment[0]]["file_path"]
        args["right_robot_file"] = embodiment_types[embodiment[1]]["file_path"]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"embodiment must have 1 or 3 items, got {len(embodiment)}")

    with open(os.path.join(args["left_robot_file"], "config.yml"), encoding="utf-8") as f:
        args["left_embodiment_config"] = yaml.safe_load(f)
    with open(os.path.join(args["right_robot_file"], "config.yml"), encoding="utf-8") as f:
        args["right_embodiment_config"] = yaml.safe_load(f)

    # Camera dimensions
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as f:
        camera_config = yaml.safe_load(f)
    head_cam = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_cam]["h"]
    args["head_camera_w"] = camera_config[head_cam]["w"]

    # Headless overrides
    args["render_freq"] = 0
    args["task_name"] = task_name
    args["task_config"] = task_config

    _ROBOTWIN_SETUP_CACHE[task_name] = args
    return dict(args)


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
    ``"head_camera"``, ``"left_camera"``). ``preprocess_observation`` in
    ``envs/utils.py`` then converts these to ``observation.images.<cam>``.

    Actions
    -------
    14-dim float32 array in ``[-1, 1]`` (joint-space, 7 DOF per arm).

    Autograd
    --------
    ``setup_demo`` and ``take_action`` drive CuRobo's Newton trajectory
    optimizer, which calls ``cost.backward()`` internally. lerobot_eval wraps
    the rollout in ``torch.no_grad()``, so both call sites re-enable grad.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        task_name: str,
        episode_index: int = 0,
        n_envs: int = 1,
        camera_names: Sequence[str] = ROBOTWIN_CAMERA_NAMES,
        observation_height: int | None = None,
        observation_width: int | None = None,
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
        # Default to D435 dims (the camera type baked into task_config/demo_clean.yml).
        # The YAML-driven lookup is deferred to reset() so construction doesn't
        # import RoboTwin's `envs` module — fast-tests run without RoboTwin installed.
        self.observation_height = observation_height or DEFAULT_CAMERA_H
        self.observation_width = observation_width or DEFAULT_CAMERA_W
        self.episode_length = episode_length
        self._max_episode_steps = episode_length  # lerobot_eval.rollout reads this
        self.render_mode = render_mode

        self._env: Any | None = None  # deferred — created on first reset() inside worker
        self._step_count: int = 0
        self._black_frame = np.zeros((self.observation_height, self.observation_width, 3), dtype=np.uint8)

        image_spaces = {
            cam: spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )
            for cam in self.camera_names
        }
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(image_spaces),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(ACTION_DIM,), dtype=np.float32),
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
        cameras_raw = raw.get("observation", {})

        images: dict[str, np.ndarray] = {}
        for cam in self.camera_names:
            cam_data = cameras_raw.get(cam)
            img = cam_data.get("rgb") if cam_data else None
            if img is None:
                images[cam] = self._black_frame
                continue
            img = np.asarray(img, dtype=np.uint8)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] != 3:
                img = img[..., :3]
            images[cam] = img

        ja = raw.get("joint_action") or {}
        vec = ja.get("vector")
        if vec is not None:
            arr = np.asarray(vec, dtype=np.float32).ravel()
            joint_state = (
                arr[:ACTION_DIM] if arr.size >= ACTION_DIM else np.zeros(ACTION_DIM, dtype=np.float32)
            )
        else:
            joint_state = np.zeros(ACTION_DIM, dtype=np.float32)

        return {"pixels": images, "agent_pos": joint_state}

    def reset(self, seed: int | None = None, **kwargs) -> tuple[RobotObservation, dict]:
        self._ensure_env()
        super().reset(seed=seed)
        assert self._env is not None  # set by _ensure_env() above

        actual_seed = self.episode_index if seed is None else seed
        setup_kwargs = _load_robotwin_setup_kwargs(self.task_name)
        setup_kwargs.update(seed=actual_seed, is_test=True)
        with torch.enable_grad():
            self._env.setup_demo(**setup_kwargs)
        self.episode_index += self._reset_stride
        self._step_count = 0

        obs = self._get_obs()
        return obs, {"is_success": False, "task": self.task_name}

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        assert self._env is not None, "step() called before reset()"
        if action.ndim != 1 or action.shape[0] != ACTION_DIM:
            raise ValueError(f"Expected 1-D action of shape ({ACTION_DIM},), got {action.shape}")

        with torch.enable_grad():
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
                import contextlib

                with contextlib.suppress(TypeError):
                    self._env.close_env()
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
    observation_height: int = DEFAULT_CAMERA_H,
    observation_width: int = DEFAULT_CAMERA_W,
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

    logger.info(
        "Creating RoboTwin envs | tasks=%s | n_envs(per task)=%d",
        task_names,
        n_envs,
    )

    is_async = env_cls is gym.vector.AsyncVectorEnv
    cached_obs_space: spaces.Space | None = None
    cached_act_space: spaces.Space | None = None
    cached_metadata: dict[str, Any] | None = None

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
        if is_async:
            lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
            if cached_obs_space is None:
                cached_obs_space = lazy.observation_space
                cached_act_space = lazy.action_space
                cached_metadata = lazy.metadata
            out[task_name][0] = lazy
        else:
            out[task_name][0] = env_cls(fns)
        logger.info("Built vec env | task=%s | n_envs=%d", task_name, n_envs)

    return {k: dict(v) for k, v in out.items()}
