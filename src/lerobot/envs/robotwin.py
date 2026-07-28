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
import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from lerobot.types import RobotObservation
from lerobot.utils.import_utils import _scipy_available

from .utils import _LazyAsyncVectorEnv

# scipy is only used for end-effector-pose composition (``--env.action_mode=ee``); guard it so this
# module (and its base-env unit tests, which mock the RoboTwin runtime) imports without scipy installed.
if _scipy_available:
    from scipy.spatial.transform import Rotation
else:
    Rotation = None

logger = logging.getLogger(__name__)

# Camera names as used by RoboTwin 2.0. The wrapper appends "_rgb" when looking
# up keys in get_obs() output (e.g. "head_camera" → "head_camera_rgb").
ROBOTWIN_CAMERA_NAMES: tuple[str, ...] = (
    "head_camera",
    "left_camera",
    "right_camera",
)

ACTION_DIM = 14  # 7 DOF × 2 arms (joint-space control mode)
# End-effector-pose control mode: per arm [x, y, z, qx, qy, qz, qw, gripper] = 8, dual-arm = 16.
# Used by world-model policies (e.g. LingBot-VA) that predict eef-pose deltas executed via CuRobo IK.
EEF_ACTION_DIM = 16
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
DEFAULT_EPISODE_LENGTH = 1200
OFFICIAL_INSTRUCTION_ENV = "LEROBOT_ROBOTWIN_OFFICIAL_INSTRUCTION"
OFFICIAL_INSTRUCTION_TYPE_ENV = "LEROBOT_ROBOTWIN_INSTRUCTION_TYPE"
OFFICIAL_INSTRUCTION_MAX_ENV = "LEROBOT_ROBOTWIN_INSTRUCTION_MAX"


def _compose_eef_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    """Compose a single-arm predicted delta pose onto the initial pose.

    ``new_pose`` / ``init_pose`` are 8-vectors ``[x, y, z, qx, qy, qz, qw, gripper]``. Translation
    is added, rotation is composed (``init_R * new_R``), and the gripper is taken from the
    prediction. Mirrors ``add_eef_pose`` in the upstream LingBot-VA RoboTwin client.
    """
    new_r = Rotation.from_quat(new_pose[3:7])
    init_r = Rotation.from_quat(init_pose[3:7])
    out_rot = (init_r * new_r).as_quat()
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])


def _add_init_eef_pose(delta_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    """Compose a dual-arm (16-d) predicted delta pose onto the initial eef pose, normalizing quats."""
    left = _compose_eef_pose(delta_pose[:8], init_pose[:8])
    right = _compose_eef_pose(delta_pose[8:], init_pose[8:])
    out = np.concatenate([left, right])
    # Normalize the two quaternions (indices 3:7 and 11:15) as the upstream client does.
    out[3:7] = out[3:7] / (np.linalg.norm(out[3:7]) + 1e-8)
    out[11:15] = out[11:15] / (np.linalg.norm(out[11:15]) + 1e-8)
    return out


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _arm_for_block(block: Any) -> str:
    return "left" if float(block.get_pose().p[0]) < 0 else "right"


def _robotwin_blocks_episode_info(task_name: str, env: Any) -> dict[str, str] | None:
    """Infer the episode-info dict used by RoboTwin's official instruction generator for block ranking."""
    if task_name == "blocks_ranking_rgb":
        return {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": _arm_for_block(env.block1),
            "{b}": _arm_for_block(env.block2),
            "{c}": _arm_for_block(env.block3),
        }
    if task_name == "blocks_ranking_size":
        return {
            "{A}": "large block",
            "{B}": "medium block",
            "{C}": "small block",
            "{a}": _arm_for_block(env.block1),
            "{b}": _arm_for_block(env.block2),
            "{c}": _arm_for_block(env.block3),
        }
    return None


def _generate_robotwin_official_instruction(task_name: str, env: Any) -> str:
    """Generate language with RoboTwin's official task templates, matching its eval client."""
    fallback = task_name.replace("_", " ")
    episode_info = _robotwin_blocks_episode_info(task_name, env)
    if episode_info is None:
        logger.warning(
            "Official RoboTwin instruction is not implemented for task=%s; using %r.", task_name, fallback
        )
        return fallback

    try:
        # Part of the robotwin simulator repo, this is being pulled by the docker image running robotwin
        # see https://github.com/RoboTwin-Platform/RoboTwin/tree/main/description
        # Used to generate the official instructions
        from description.utils.generate_episode_instructions import generate_episode_descriptions
    except Exception:
        logger.warning(
            "Failed to import RoboTwin official instruction generator; using %r.", fallback, exc_info=True
        )
        return fallback

    instruction_type = os.environ.get(OFFICIAL_INSTRUCTION_TYPE_ENV, "seen")
    try:
        max_descriptions = int(os.environ.get(OFFICIAL_INSTRUCTION_MAX_ENV, "1000000"))
    except ValueError:
        max_descriptions = 1000000

    results = generate_episode_descriptions(task_name, [episode_info], max_descriptions=max_descriptions)
    if not results:
        logger.warning(
            "RoboTwin generated no official instructions for task=%s; using %r.", task_name, fallback
        )
        return fallback

    options = results[0].get(instruction_type) or results[0].get("seen") or results[0].get("unseen")
    if not options:
        logger.warning(
            "RoboTwin generated no %s official instructions for task=%s; using %r.",
            instruction_type,
            task_name,
            fallback,
        )
        return fallback

    return str(np.random.choice(options))


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
        action_mode: str = "joint",
    ):
        super().__init__()
        self.task_name = task_name
        self.task = task_name  # used by add_envs_task() in utils.py
        self.task_description = task_name.replace("_", " ")
        self.episode_index = episode_index
        self._reset_stride = n_envs
        # "joint": 14-d joint-space actions via take_action(action). "ee": 16-d end-effector-pose
        # deltas (added onto the episode's initial eef pose) executed via take_action(.., "ee") + IK.
        if action_mode not in ("joint", "ee"):
            raise ValueError(f"action_mode must be 'joint' or 'ee'; got {action_mode!r}")
        self.action_mode = action_mode
        self._action_dim = EEF_ACTION_DIM if action_mode == "ee" else ACTION_DIM
        self._init_eef_pose: np.ndarray | None = None
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
            low=ACTION_LOW, high=ACTION_HIGH, shape=(self._action_dim,), dtype=np.float32
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

    def _read_eef_pose(self) -> np.ndarray:
        """Read the current 16-d dual-arm eef pose [left(xyz+quat)+grip, right(xyz+quat)+grip]."""
        assert self._env is not None, "_read_eef_pose called before _ensure_env()"
        ep = self._env.get_obs()["endpose"]
        pose = (
            list(ep["left_endpose"])
            + [ep["left_gripper"]]
            + list(ep["right_endpose"])
            + [ep["right_gripper"]]
        )
        return np.asarray(pose, dtype=np.float64)

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

        use_official_instruction = self.task_name in {"blocks_ranking_rgb", "blocks_ranking_size"}
        if _env_flag(OFFICIAL_INSTRUCTION_ENV, default=use_official_instruction):
            self.task_description = _generate_robotwin_official_instruction(self.task_name, self._env)
            if hasattr(self._env, "set_instruction"):
                self._env.set_instruction(instruction=self.task_description)
            logger.info("RoboTwin official instruction | task=%s | %s", self.task_name, self.task_description)
        else:
            self.task_description = self.task_name.replace("_", " ")

        # In eef mode the policy predicts pose deltas relative to the initial eef pose.
        if self.action_mode == "ee":
            self._init_eef_pose = self._read_eef_pose()

        obs = self._get_obs()
        return obs, {"is_success": False, "task": self.task_name}

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        assert self._env is not None, "step() called before reset()"
        if action.ndim != 1 or action.shape[0] != self._action_dim:
            raise ValueError(f"Expected 1-D action of shape ({self._action_dim},), got {action.shape}")

        with torch.enable_grad():
            if self.action_mode == "ee":
                ee_action = _add_init_eef_pose(np.asarray(action, dtype=np.float64), self._init_eef_pose)
                self._env.take_action(ee_action, action_type="ee")
            elif hasattr(self._env, "take_action"):
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
    action_mode: str = "joint",
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
            action_mode=action_mode,
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
    action_mode: str = "joint",
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
            action_mode=action_mode,
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
