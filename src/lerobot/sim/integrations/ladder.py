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

"""LADDER long-horizon manipulation benchmark integration for LeRobot simulation.

LADDER (https://github.com/insait-institute/ladder, https://ladderrobotics.ai, arXiv:2610.18053)
evaluates policies across 145 long-horizon manipulation tasks (1 to 10 subgoals) in three suites:
- ``simpler`` (``SimplerEnv-Long``): 54 WidowX tasks on SAPIEN 3 / ManiSkill 3, 24 episodes/task.
- ``robot_arena`` (``RobotArena-∞-Long``): 47 WidowX tasks on Genesis, 8 episodes/task.
- ``robolab`` (``RoboLab-Long``): 44 Franka DROID tasks on Isaac Sim 6 / Isaac Lab 3, 8 episodes/task.

Each suite steps all ``n_envs`` parallel worlds of a task in one batched simulator instance, so this
integration defines ``vector_env_fns``.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from .utils import canonical_observation

__all__ = ["canonical_observation", "conventions", "vector_env_fns"]

DEFAULT_LADDER_ROOT = "/app/ladder"
_NO_AUTORESET = getattr(getattr(gym.vector, "AutoresetMode", None), "DISABLED", "disabled")
_CANONICAL_TO_WIDOWX_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
)

WIDOWX_STATE_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")
WIDOWX_DELTA_ACTION_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")
WIDOWX_ABS_ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")
DROID_JOINT_NAMES = (*(f"joint_{i}.pos" for i in range(7)), "gripper.pos")

# fmt: off
SIMPLER_TASKS: tuple[str, ...] = (
    "all10_plate", "all10_towel", "all1_plate", "all1_towel", "all2_plate", "all2_towel",
    "all3_plate", "all3_towel", "all4_plate", "all4_towel", "all5_plate", "all5_towel",
    "all6_plate", "all6_towel", "all7_plate", "all7_towel", "all8_plate", "all8_towel",
    "all9_plate", "all9_towel", "except_eggplant_2", "except_eggplant_3", "except_eggplant_4",
    "except_eggplant_5", "move_plate_to_towel_2", "move_plate_to_towel_3", "move_plate_to_towel_4",
    "move_plate_to_towel_5", "move_towel_to_plate_3", "ordered_plate_2", "ordered_plate_3",
    "relay10e", "relay2e", "relay3e", "relay4e", "relay5e", "relay6e", "relay7e", "relay8e",
    "relay9e", "shuttle10", "shuttle7", "shuttle8", "shuttle9", "split2", "split3", "split4",
    "split5", "split6", "split7", "spoons2", "spoons3", "spoons4", "spoons5",
)

ROBOT_ARENA_TASKS: tuple[str, ...] = (
    "scene36_1goal", "scene36_2goal", "scene45_1goal", "scene45_2goal", "scene45_3goal",
    "scene45_4goal", "scene45_5goal", "scene45_6goal", "scene45_all_2", "scene45_all_3",
    "scene45_all_4", "scene45_all_5", "scene45_all_6", "scene45_except_2", "scene45_except_3",
    "scene45_except_4", "scene45_except_5", "scene45_except_6", "scene50_3goal", "scene54_1goal",
    "scene54_2goal", "scene54_2goal_ordered", "scene54_3goal", "scene54_3goal_ordered",
    "scene54_duplicates_2", "scene54_duplicates_3", "scene54_except", "scene54_except_3",
    "scene54_relay_2", "scene54_relay_4", "scene54_revisit_10", "scene54_revisit_3",
    "scene54_revisit_6", "scene54_revisit_7", "scene54_revisit_8", "scene54_revisit_9",
    "scene54_route_2", "scene54_route_4", "scene54_route_6", "scene54_route_7",
    "scene54_route_8", "scene54_split_2", "scene54_split_3", "scene54_transfer_2",
    "scene76_1goal", "scene76_2goal", "scene76_2goal_ordered",
)

ROBOLAB_TASKS: tuple[str, ...] = (
    "comp_bin_1", "comp_bin_10", "comp_bin_2", "comp_bin_2_distractor", "comp_bin_4_distractor",
    "comp_bin_6", "comp_identical_2", "comp_identical_4", "comp_ordered_bin_10",
    "comp_ordered_bin_2", "comp_ordered_bin_3", "comp_ordered_bin_4", "comp_ordered_bin_5",
    "comp_ordered_bin_6", "comp_ordered_bin_7", "comp_ordered_bin_8", "comp_ordered_bin_9",
    "comp_plate_1", "comp_relay_2", "comp_relay_3", "comp_relay_4", "comp_relay_5",
    "comp_revisit_1", "comp_revisit_2", "comp_revisit_3", "comp_route_10", "comp_route_2",
    "comp_route_2_binfirst", "comp_route_3", "comp_route_4", "comp_route_5", "comp_route_6",
    "comp_route_7", "comp_route_8", "comp_route_9", "comp_split_plate_2", "comp_split_plate_3",
    "comp_split_plate_4", "comp_split_plate_5", "comp_split_plate_6", "comp_split_plate_7",
    "comp_split_plate_8", "comp_transfer_2", "comp_transfer_3",
)

ROBOT_ARENA_K8_OFFSETS: tuple[tuple[float, float], ...] = (
    (-0.03, -0.02), (-0.03, +0.02), (-0.01, -0.02), (-0.01, +0.02),
    (+0.01, -0.02), (+0.01, +0.02), (+0.03, -0.02), (+0.03, +0.02),
)
# fmt: on

SUITE_TASKS: dict[str, tuple[str, ...]] = {
    "simpler": SIMPLER_TASKS,
    "robot_arena": ROBOT_ARENA_TASKS,
    "robolab": ROBOLAB_TASKS,
}


def _resolve_action_mode(
    suite: str, action_mode: str | None = "auto", control_mode: str | None = None
) -> str:
    if control_mode and "target_base_pose" in control_mode:
        return "abs_ee"
    if action_mode in (None, "", "auto"):
        return "joint_pos" if suite == "robolab" else "delta_ee"
    if action_mode not in ("delta_ee", "abs_ee", "joint_pos", "joint_vel"):
        raise ValueError(
            f"Invalid action_mode={action_mode!r}. Expected 'auto', 'delta_ee', 'abs_ee', 'joint_pos', or 'joint_vel'."
        )
    return action_mode


def _axisangle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    kx, ky, kz = rv / theta
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _matrix_to_axisangle(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    theta = float(np.arccos(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)))
    if theta < 1e-7:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    return ((axis / norm) * theta).astype(np.float32) if norm >= 1e-12 else np.zeros(3, dtype=np.float32)


def _intrinsic_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    x, y, z = float(euler[0]), float(euler[1]), float(euler[2])
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    return np.array(
        [
            [cy * cz, -cy * sz, sy],
            [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
            [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
        ],
        dtype=np.float64,
    )


def _ensure_ladder_on_path(root: Path) -> None:
    for p in (
        root / "openpi_src" / "packages" / "openpi-client" / "src",
        root / "robolab_src",
        root,
        root / "build_support" / "python",
    ):
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
    try:
        import ladder.components  # noqa: F401

        return
    except ImportError:
        sys.modules.pop("ladder", None)
    if not (root / "ladder").is_dir():
        raise FileNotFoundError(
            f"LADDER source tree not found at {root}. Set LADDER_ROOT or pass root=... in env config."
        )


class _RoboLabKit:
    app: Any = None
    thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ladder-robolab")
    env: Any = None
    env_cfg: Any = None
    oracle: Any = None
    key: tuple[str, int] | None = None


class _RobotArenaRuntime:
    thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ladder-robot-arena")
    env: Any = None


class LadderVectorEnv(gym.vector.VectorEnv):
    """Native batched ``gym.vector.VectorEnv`` wrapping a LADDER task across ``n_envs`` parallel worlds."""

    def __init__(
        self,
        suite: str = "simpler",
        task: str = "all1_plate",
        n_envs: int = 1,
        action_mode: str = "auto",
        control_mode: str | None = None,
        episode_length: int | None = None,
        front_camera_name: str = "main",
        wrist_camera_name: str = "wrist",
        observation_height: int | None = None,
        observation_width: int | None = None,
        root: str | None = None,
        assets_dir: str | None = None,
        seed: int = 42,
    ) -> None:
        if suite not in SUITE_TASKS:
            raise ValueError(f"Unknown LADDER suite {suite!r}. Expected one of {sorted(SUITE_TASKS)}.")
        if task not in SUITE_TASKS[suite]:
            raise ValueError(f"Unknown task {task!r} for suite {suite!r}.")

        resolved_action_mode = _resolve_action_mode(suite, action_mode, control_mode)
        self.suite = suite
        self.task = task
        self.task_description = task
        self.num_envs = int(n_envs)
        self._action_mode = resolved_action_mode
        self._control_mode = control_mode
        self._max_episode_steps = int(episode_length or (1500 if suite == "robolab" else 1000))
        self._front_camera_name = front_camera_name
        self._wrist_camera_name = wrist_camera_name
        self._root = Path(root or os.environ.get("LADDER_ROOT", DEFAULT_LADDER_ROOT)).expanduser().resolve()
        self._assets_dir = Path(assets_dir).expanduser().resolve() if assets_dir else None
        self._seed = int(seed)

        default_h, default_w = (224, 224) if suite == "robolab" else (480, 640)
        self._obs_h = int(observation_height or default_h)
        self._obs_w = int(observation_width or default_w)
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": 15 if suite == "robolab" else 5,
            "autoreset_mode": _NO_AUTORESET,
        }
        self.render_mode = "rgb_array"
        self.closed = False

        action_dim = 8 if resolved_action_mode in ("joint_pos", "joint_vel") else 7
        state_dim = 8 if suite == "robolab" else 7
        cam_spaces: dict[str, spaces.Space] = {
            front_camera_name: spaces.Box(0, 255, shape=(self._obs_h, self._obs_w, 3), dtype=np.uint8),
        }
        if suite == "robolab":
            cam_spaces[wrist_camera_name] = spaces.Box(
                0, 255, shape=(self._obs_h, self._obs_w, 3), dtype=np.uint8
            )

        self.single_observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(cam_spaces),
                "agent_pos": spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32),
            }
        )
        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self._env: Any = None
        self._env_cfg: Any = None
        self._converter: Any = None
        self._raw_obs: Any = None
        self._oracle: Any = None
        self._padded_single = False
        self._rollout_idx = 0
        self._step_count = 0
        self._last_eef_poses = [np.eye(4, dtype=np.float32) for _ in range(self.num_envs)]
        self._last_target_poses = [np.eye(4, dtype=np.float32) for _ in range(self.num_envs)]
        self._last_joints: list[np.ndarray | None] = [None for _ in range(self.num_envs)]
        self._last_frames = [
            np.zeros((self._obs_h, self._obs_w, 3), dtype=np.uint8) for _ in range(self.num_envs)
        ]
        self._success_latched = np.zeros(self.num_envs, dtype=bool)

    @property
    def unwrapped(self) -> LadderVectorEnv:
        return self

    def reset(
        self, *, seed: int | list[int | None] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        episode_ids = [self._rollout_idx * self.num_envs + i for i in range(self.num_envs)]
        self._rollout_idx += 1
        self._step_count = 0
        self._success_latched = np.zeros(self.num_envs, dtype=bool)
        obs, _, _, _, info = self._pack_batch(self._reset_backend(episode_ids))
        return obs, info

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        act_arr = np.asarray(actions, dtype=np.float32)
        if act_arr.ndim == 1:
            act_arr = act_arr[None, :]
        decoded = [self._decode_action(i, act_arr[i]) for i in range(self.num_envs)]
        raw_obs_list = self._step_backend(decoded)
        self._step_count += 1
        return self._pack_batch(raw_obs_list)

    def render(self) -> tuple[np.ndarray, ...]:
        return tuple(self._last_frames)

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        if name == "render":
            return self.render()
        attr = getattr(self, name)
        val = attr(*args, **kwargs) if callable(attr) else attr
        return tuple(val for _ in range(self.num_envs))

    def get_attr(self, name: str) -> list[Any]:
        if name in ("task", "task_description", "_max_episode_steps"):
            return [getattr(self, name) for _ in range(self.num_envs)]
        raise AttributeError(name)

    def close(self, **kwargs: Any) -> None:
        if self._env is not None and self.suite == "simpler":
            with contextlib.suppress(Exception):
                self._env.close()
        self._env = None
        self.closed = True

    def _pack_batch(
        self, raw_obs_list: list[Any]
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        front_imgs: list[np.ndarray] = []
        wrist_imgs: list[np.ndarray] = []
        states: list[np.ndarray] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        num_complete = np.zeros(self.num_envs, dtype=np.int32)

        for i, raw_obs in enumerate(raw_obs_list):
            if isinstance(raw_obs, dict):
                images = {k: np.asarray(v, dtype=np.uint8) for k, v in (raw_obs.get("images") or {}).items()}
                st = raw_obs.get("state") or {}
                eef_pose = np.asarray(st.get("eef_pose", np.eye(4)), dtype=np.float32)
                target_pose = np.asarray(st.get("eef_pose_from_target", eef_pose), dtype=np.float32)
                gripper = float(np.asarray(st.get("gripper_state", [1.0])).reshape(-1)[0])
                joints = st.get("joints")
                metadata = raw_obs.get("metadata") or {}
            else:
                images = {k: np.asarray(v, dtype=np.uint8) for k, v in (raw_obs.images or {}).items()}
                eef_pose = np.asarray(raw_obs.state.eef_pose, dtype=np.float32)
                target_raw = getattr(raw_obs.state, "eef_pose_from_target", None)
                target_pose = np.asarray(target_raw if target_raw is not None else eef_pose, dtype=np.float32)
                gripper = float(np.asarray(raw_obs.state.gripper_state).reshape(-1)[0])
                joints = getattr(raw_obs.state, "joints", None)
                metadata = raw_obs.metadata or {}

            self._last_eef_poses[i] = eef_pose
            self._last_target_poses[i] = target_pose
            self._last_joints[i] = (
                np.asarray(joints, dtype=np.float32).reshape(-1)[:7] if joints is not None else None
            )

            instruction = str(metadata.get("instruction", "")).strip()
            if instruction:
                self.task_description = instruction

            main_img = images.get("main")
            if main_img is None:
                main_img = next(
                    iter(images.values()), np.zeros((self._obs_h, self._obs_w, 3), dtype=np.uint8)
                )
            front_imgs.append(main_img)
            self._last_frames[i] = main_img
            if self.suite == "robolab":
                wrist_imgs.append(images.get("wrist", np.zeros_like(main_img)))

            if self.suite == "robolab" and joints is not None:
                j = np.asarray(joints, dtype=np.float32).reshape(-1)[:7]
                states.append(np.concatenate([j, np.array([1.0 - gripper], dtype=np.float32)]))
            else:
                rotvec = _matrix_to_axisangle(eef_pose[:3, :3])
                states.append(
                    np.concatenate([eef_pose[:3, 3], rotvec, np.array([gripper], dtype=np.float32)])
                )

            if bool(metadata.get("done", False)):
                self._success_latched[i] = True
            is_succ = bool(self._success_latched[i])
            oracle = metadata.get("oracle") if isinstance(metadata.get("oracle"), dict) else None
            achieved = [bool(v) for v in oracle.get("achieved", [])] if oracle else []
            n_goals = len(achieved)
            completed = int(metadata.get("num_subgoals_complete", sum(achieved)))
            if is_succ and n_goals > 0:
                completed = max(completed, n_goals)

            rewards[i] = float(completed) / float(n_goals) if n_goals > 0 else float(is_succ)
            terminated[i] = is_succ
            truncated[i] = bool(metadata.get("truncated", False)) or (
                self._step_count >= self._max_episode_steps
            )
            num_complete[i] = completed

        pixels = {self._front_camera_name: np.stack(front_imgs, axis=0)}
        if self.suite == "robolab":
            pixels[self._wrist_camera_name] = np.stack(wrist_imgs, axis=0)
        obs = {"pixels": pixels, "agent_pos": np.stack(states, axis=0).astype(np.float32)}
        info = {
            "is_success": self._success_latched.copy(),
            "subgoal_progress": rewards.copy(),
            "num_subgoals_complete": num_complete,
        }
        return obs, rewards, terminated, truncated, info

    def _decode_action(self, env_i: int, raw: np.ndarray) -> Any:
        from ladder.components.inference.api.core.action import (
            EndEffectorPoseAction,
            JointAnglesAction,
            ManipulatorAction,
        )

        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        if raw.size == 8 and self._action_mode in ("joint_pos", "joint_vel"):
            arm = raw[:7].astype(np.float32)
            cur_j = self._last_joints[env_i]
            joints = (
                (cur_j + np.clip(arm, -1.0, 1.0) * 0.2).astype(np.float32)
                if (self._action_mode == "joint_vel" and cur_j is not None)
                else arm
            )
            return ManipulatorAction(
                arm_action=JointAnglesAction(angles=joints),
                gripper_action=0.0 if float(raw[7]) > 0.5 else 1.0,
            )

        if raw.size < 7:
            raise ValueError(f"LadderVectorEnv expected at least a 7-D action, got shape {raw.shape}")
        pos = raw[:3].astype(np.float32)
        rot_in = raw[3:6].astype(np.float64)
        gripper = float(1.0 if raw[6] > 0.0 else 0.0) if raw[6] < 0.0 else float(np.clip(raw[6], 0.0, 1.0))
        target_eef = self._last_target_poses[env_i]
        if self._action_mode == "delta_ee":
            translation = (target_eef[:3, 3] + pos).astype(np.float32)
            rotation = (_axisangle_to_matrix(rot_in) @ target_eef[:3, :3]).astype(np.float32)
            new_target = np.eye(4, dtype=np.float32)
            new_target[:3, 3], new_target[:3, :3] = translation, rotation
            self._last_target_poses[env_i] = new_target
        elif self._control_mode and "target_base_pose" in self._control_mode:
            translation = pos
            rotation = (_intrinsic_xyz_to_matrix(rot_in) @ _CANONICAL_TO_WIDOWX_ROTATION.T).astype(np.float32)
        else:
            translation = pos
            rotation = _intrinsic_xyz_to_matrix(rot_in).astype(np.float32)
        return ManipulatorAction(
            arm_action=EndEffectorPoseAction(translation=translation, rotation=rotation),
            gripper_action=gripper,
        )

    def _reset_backend(self, episode_ids: list[int]) -> list[Any]:
        if self.suite == "simpler":
            return self._reset_simpler(episode_ids)
        if self.suite == "robot_arena":
            return _RobotArenaRuntime.thread.submit(self._reset_robot_arena, episode_ids).result()
        return _RoboLabKit.thread.submit(self._reset_robolab).result()

    def _step_backend(self, actions: list[Any]) -> list[Any]:
        assert self._env is not None
        if self.suite == "simpler":
            return self._env.step(actions)
        if self.suite == "robot_arena":
            step_actions = [actions[0], actions[0]] if self._padded_single else actions
            obs_list = _RobotArenaRuntime.thread.submit(self._env.step, step_actions).result()
            return obs_list[:1] if self._padded_single else obs_list
        return _RoboLabKit.thread.submit(self._step_robolab, actions).result()

    def _reset_simpler(self, episode_ids: list[int]) -> list[Any]:
        _ensure_ladder_on_path(self._root)
        if "MS_ASSET_DIR" not in os.environ:
            for cand in (
                self._assets_dir,
                Path(os.environ["LADDER_SIMPLER_ASSETS"]) if "LADDER_SIMPLER_ASSETS" in os.environ else None,
                self._root / "ladder" / "components" / "inference" / "robots" / "simpler" / "maniskill3",
            ):
                if cand is not None and cand.is_dir():
                    os.environ["MS_ASSET_DIR"] = str(cand)
                    break

        from ladder.components.inference.robots.simpler.env_builder import ABS_CONTROL_MODE
        from ladder.components.inference.robots.simpler.simpler_env import SimplerEnv, SimplerEnvConfig

        if self._env is None:
            self._env = SimplerEnv(
                SimplerEnvConfig(
                    env_name=f"{self.task}-v0",
                    robot_name="widowx",
                    scene_name="bridge_table_1_v1",
                    rgb_overlay_path="",
                    sim_freq=500,
                    control_mode=ABS_CONTROL_MODE,
                    control_freq=5,
                    max_episode_steps=0,
                    raytracing=False,
                    n_envs=self.num_envs,
                    episode_layouts=max(24, max(episode_ids, default=0) + 1),
                )
            )
        return self._env.reset(episode_ids)

    def _reset_robot_arena(self, episode_ids: list[int]) -> list[Any]:
        _ensure_ladder_on_path(self._root)
        from ladder.components.inference.robots.robot_arena.composition.registry import COMPOSITION_TASKS
        from ladder.components.inference.robots.robot_arena.robot_arena_env import (
            RobotArenaEnv,
            RobotArenaEnvConfig,
        )
        from ladder.components.inference.robots.robot_arena.scene_builder import set_genesis_seed
        from ladder.core.compute.seeder import SEEDER

        data_root = next(
            (
                str(c)
                for c in (
                    self._assets_dir,
                    Path(os.environ["LADDER_ROBOT_ARENA_ASSETS"])
                    if "LADDER_ROBOT_ARENA_ASSETS" in os.environ
                    else None,
                    self._root
                    / "ladder/components/inference/robots/robot_arena/robot_arena_data/examples/data",
                )
                if c is not None and ((c / "bridge").is_dir() or (c / "scenes").is_dir())
            ),
            None,
        )
        if data_root is None:
            raise FileNotFoundError("RobotArena assets not found. Set LADDER_ROBOT_ARENA_ASSETS.")

        SEEDER.seed_everything(self._seed)
        set_genesis_seed(self._seed)
        comp_task = COMPOSITION_TASKS[self.task]
        offsets = [ROBOT_ARENA_K8_OFFSETS[ep % len(ROBOT_ARENA_K8_OFFSETS)] for ep in episode_ids]
        self._padded_single = len(offsets) == 1
        if self._padded_single:
            offsets = [offsets[0], ROBOT_ARENA_K8_OFFSETS[0]]

        cfg = RobotArenaEnvConfig(
            data_root=data_root,
            scene_name=comp_task.scene,
            robot_name="widowx",
            max_episode_steps=comp_task.max_episode_steps,
            control_freq=5,
            sim_freq=200,
            sim_substeps=1,
            rigid_constraint_timeconst=0.02,
            camera_width=self._obs_w,
            camera_height=self._obs_h,
            per_env_object_offsets=offsets,
            composition_task=self.task,
        )
        if _RobotArenaRuntime.env is not None and _RobotArenaRuntime.env.can_reuse_for(cfg):
            _RobotArenaRuntime.env.reconfigure(cfg)
            self._env = _RobotArenaRuntime.env
        else:
            if _RobotArenaRuntime.env is not None:
                _RobotArenaRuntime.env.close()
            _RobotArenaRuntime.env = RobotArenaEnv(config=cfg)
            self._env = _RobotArenaRuntime.env
        obs_list = self._env.reset()
        return obs_list[:1] if self._padded_single else obs_list

    def _reset_robolab(self) -> list[Any]:
        _ensure_ladder_on_path(self._root)
        for cand in (
            self._assets_dir / "assets" if self._assets_dir is not None else None,
            self._assets_dir,
            Path(os.environ["LADDER_ROBOLAB_ASSETS"]) if "LADDER_ROBOLAB_ASSETS" in os.environ else None,
            self._root / "ladder" / "components" / "inference" / "robots" / "robolab" / "assets",
        ):
            if cand is not None and (cand / "robots").is_dir():
                os.environ["ROBOLAB_ASSET_DIR"] = str(cand)
                break

        from ladder.components.inference.robots.robolab.composition.specs import (
            SPECS,
            task_class_name,
            task_target,
        )
        from ladder.components.inference.robots.robolab.droid_conversions import DroidJointposConverter
        from ladder.components.inference.robots.robolab.oracle import build_oracle

        if _RoboLabKit.app is None:
            import signal
            import threading

            import cv2  # noqa: F401
            from isaaclab.app import AppLauncher

            os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            root_logger = logging.getLogger()
            levels = {handler: handler.level for handler in root_logger.handlers}
            orig_signal = signal.signal
            if threading.current_thread() is not threading.main_thread():
                signal.signal = lambda *a, **kw: None  # type: ignore[assignment]
            try:
                _RoboLabKit.app = AppLauncher(headless=True, enable_cameras=True).app
            finally:
                signal.signal = orig_signal  # type: ignore[assignment]
                for handler, level in levels.items():
                    handler.setLevel(level)
            atexit.register(
                lambda: (
                    sys.stdout.flush(),
                    sys.stderr.flush(),
                    os._exit(
                        0
                        if getattr(sys, "last_exc", None) is None and getattr(sys, "last_value", None) is None
                        else 1
                    ),
                )
            )

            import carb.settings
            import robolab.constants

            robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
            carb.settings.get_settings().set("/persistent/app/usd/muteUsdDiagnostics", False)

        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

        auto_register_droid_envs(
            task_dirs=None,
            task=[task_target(self.task) if self.task in SPECS else self.task],
            cameras=None,
        )
        cls_name = task_class_name(self.task) if self.task in SPECS else self.task
        key = (cls_name, self.num_envs)
        if _RoboLabKit.env is not None and _RoboLabKit.key == key:
            self._env = _RoboLabKit.env
            self._env_cfg = _RoboLabKit.env_cfg
            self._oracle = _RoboLabKit.oracle
            self._env.reset_eval_state()
        else:
            if _RoboLabKit.env is not None:
                _RoboLabKit.env.close()
                _RoboLabKit.env = None
            self._env, self._env_cfg = create_env(
                cls_name,
                device="cuda:0",
                num_envs=self.num_envs,
                instruction_type="default",
                policy="ladder_runner",
                renderer="realtime",
                rendering_mode="balanced",
            )
            self._oracle = build_oracle(getattr(self._env.cfg, "subtasks", None), self._env_cfg.instruction)
            _RoboLabKit.env = self._env
            _RoboLabKit.env_cfg = self._env_cfg
            _RoboLabKit.oracle = self._oracle
            _RoboLabKit.key = key

        obs, _ = self._env.reset()
        obs, _ = self._env.reset()
        self._raw_obs = obs
        self._converter = DroidJointposConverter(
            num_envs=self.num_envs, image_size=(self._obs_h, self._obs_w)
        )
        return self._build_robolab_obs_list(cls_name)

    def _step_robolab(self, actions: list[Any]) -> list[Any]:
        import omni.kit.app
        import omni.timeline
        import torch
        from ladder.components.inference.robots.robolab.composition.specs import SPECS, task_class_name

        assert self._env is not None and self._raw_obs is not None
        joints = self._raw_obs["proprio_obs"]["arm_joint_pos"].detach().cpu().numpy()
        if hasattr(self._converter, "actions_to_jointpos"):
            commands = self._converter.actions_to_jointpos(actions, joints)
        else:
            commands = np.stack(
                [self._converter.action_to_jointpos(i, actions[i], joints[i]) for i in range(self.num_envs)],
                axis=0,
            )
        timeline = omni.timeline.get_timeline_interface()
        kit_app = omni.kit.app.get_app()
        while not timeline.is_playing():
            kit_app.update()
        obs, _, _, _, _ = self._env.step(torch.as_tensor(commands, device=self._env.device))
        self._raw_obs = obs
        return self._build_robolab_obs_list(task_class_name(self.task) if self.task in SPECS else self.task)

    def _build_robolab_obs_list(self, cls_name: str) -> list[dict[str, Any]]:
        image_obs = self._raw_obs["image_obs"]
        proprio = self._raw_obs["proprio_obs"]
        results = self._env.get_env_results()
        active = set(self._env.active_env_ids)
        max_steps_reached = self._step_count >= int(self._env.max_episode_length)

        achieved_all: list[list[bool]] | None = None
        if self._oracle:
            with contextlib.suppress(Exception):
                achieved_all = [
                    [bool(g.check(self._env, env_id=i)) for g in self._oracle]
                    for i in range(self._env.num_envs)
                ]

        out: list[dict[str, Any]] = []
        for i in range(self._env.num_envs):
            success = bool(results[i]["success"])
            metadata = {
                "env_name": cls_name,
                "robot_name": "franka",
                "instruction": self._env_cfg.instruction,
                "done": success,
                "truncated": max_steps_reached or i not in active,
                "oracle": {"achieved": achieved_all[i] if achieved_all is not None else [success]},
            }
            out.append(
                self._converter.observation_dict(
                    env_i=i,
                    main_image=image_obs["over_shoulder_left_camera"][i].detach().cpu().numpy(),
                    wrist_image=image_obs["wrist_cam"][i].detach().cpu().numpy(),
                    joints=proprio["arm_joint_pos"][i].detach().cpu().numpy(),
                    gripper_position=proprio["gripper_pos"][i].detach().cpu().numpy(),
                    metadata=metadata,
                )
            )
        return out


def vector_env_fns(
    task: str = "",
    n_envs: int = 1,
    suite: str = "simpler",
    action_mode: str = "auto",
    control_mode: str | None = None,
    episode_length: int | None = None,
    front_camera_name: str = "main",
    wrist_camera_name: str = "wrist",
    observation_height: int | None = None,
    observation_width: int | None = None,
    root: str | None = None,
    assets_dir: str | None = None,
    seed: int = 42,
) -> dict[str, dict[int, Callable[[], gym.vector.VectorEnv]]]:
    """Return one batched ``LadderVectorEnv`` factory per task, as ``{task_name: {0: factory}}``."""
    if suite not in SUITE_TASKS:
        raise ValueError(f"Unknown LADDER suite {suite!r}. Expected one of {sorted(SUITE_TASKS)}.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    known = SUITE_TASKS[suite]
    names = list(known) if task in ("", "all", "*") else [t.strip() for t in task.split(",") if t.strip()]
    if not names:
        names = list(known)
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"Unknown LADDER tasks for suite {suite!r}: {unknown}. Available: {list(known)}")

    make = partial(
        LadderVectorEnv,
        suite=suite,
        n_envs=n_envs,
        action_mode=action_mode,
        control_mode=control_mode,
        episode_length=episode_length,
        front_camera_name=front_camera_name,
        wrist_camera_name=wrist_camera_name,
        observation_height=observation_height,
        observation_width=observation_width,
        root=root,
        assets_dir=assets_dir,
        seed=seed,
    )
    return {name: {0: partial(make, task=name)} for name in names}


def conventions(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the LADDER suite's default control rate, control mode, component names, and robot type."""
    suite = str(kwargs.get("suite") or "simpler")
    if suite not in SUITE_TASKS:
        raise ValueError(f"Unknown LADDER suite {suite!r}. Expected one of {sorted(SUITE_TASKS)}.")
    action_mode = _resolve_action_mode(suite, kwargs.get("action_mode"), kwargs.get("control_mode"))
    if suite == "robolab":
        joint_ctrl = action_mode in ("joint_pos", "joint_vel")
        return {
            "fps": 15,
            "control_mode": "position"
            if joint_ctrl
            else ("delta" if action_mode == "delta_ee" else "eef_pose"),
            "state_names": DROID_JOINT_NAMES,
            "action_names": DROID_JOINT_NAMES
            if joint_ctrl
            else (WIDOWX_DELTA_ACTION_NAMES if action_mode == "delta_ee" else WIDOWX_ABS_ACTION_NAMES),
            "robot_type": "Franka",
        }
    return {
        "fps": 5,
        "control_mode": "delta" if action_mode == "delta_ee" else "eef_pose",
        "state_names": WIDOWX_STATE_NAMES,
        "action_names": WIDOWX_DELTA_ACTION_NAMES if action_mode == "delta_ee" else WIDOWX_ABS_ACTION_NAMES,
        "robot_type": "widowx",
    }
