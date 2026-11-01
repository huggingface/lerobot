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

"""NVIDIA RoboLab: 120 tabletop tasks for a DROID robot (Franka + Robotiq 2F-85) on Isaac Lab.

Each task is one suite of one task id. Isaac Lab steps every env of a task in one GPU world, so this integration
defines ``vector_env_fns``. The tasks are listed from RoboLab's task metadata, without starting Isaac Sim.

Install: the ``robolab_native`` stage of docker/Dockerfile.sim (Isaac Sim 6.0.1, Isaac Lab 3.0 beta 2).
Benchmark: https://github.com/NVlabs/RoboLab
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from .utils import canonical_observation

__all__ = ["canonical_observation", "conventions", "vector_env_fns"]

# Isaac Lab steps physics at 120 Hz and applies each action for 8 steps (RoboLab's DROID registration).
CONTROL_FPS = 15
DEFAULT_CAMERAS = "WRIST_LEFT"
JOINT_NAMES = (*(f"joint_{i}.pos" for i in range(7)), "gripper.pos")

# The Kit app Isaac Lab runs in: started once per process, before any isaaclab or omni import, and kept for the
# next task's env.
_APP: Any = None


def _task_metadata() -> dict[str, dict[str, Any]]:
    """Return RoboLab's metadata for each benchmark task, by task name."""
    import robolab.constants

    path = Path(robolab.constants.TASK_DIR) / "_metadata" / "task_metadata.json"
    return {task["task_name"]: task for task in json.loads(path.read_text())}


def _launch_app() -> None:
    global _APP
    if _APP is None:
        # RoboLab's runners import OpenCV before Isaac Lab.
        import cv2  # noqa: F401
        from isaaclab.app import AppLauncher

        root = logging.getLogger()
        levels = {handler: handler.level for handler in root.handlers}
        _APP = AppLauncher(headless=True, enable_cameras=True).app
        # Isaac Lab raises the handlers already installed to WARNING, which hides LeRobot's own logs.
        for handler, level in levels.items():
            handler.setLevel(level)


class RoboLabVectorEnv(gym.vector.VectorEnv):
    """``num_envs`` envs of one RoboLab task in one Isaac Lab world, with no autoreset.

    RoboLab freezes an env once it terminates and records whether it succeeded; ``step`` reports a frozen env as
    terminated (success) or truncated (time out) from that record, so its rule for physics artifacts (an env that
    terminates within two steps is reset instead) applies unchanged.
    """

    def __init__(
        self,
        task: str,
        num_envs: int,
        cameras: str = DEFAULT_CAMERAS,
        instruction_type: str = "default",
        observation_height: int | None = None,
        observation_width: int | None = None,
        device: str = "cuda:0",
        seed: int = 0,
    ) -> None:
        _launch_app()
        import gymnasium as gym_registry
        import robolab.constants
        from robolab.core.environments.config import parse_env_cfg
        from robolab.core.environments.runtime import create_env
        from robolab.core.task.instruction import instruction_text
        from robolab.registrations.droid import camera_presets
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

        # RoboLab writes each env's config to its output directory.
        robolab.constants.set_output_dir(tempfile.mkdtemp(prefix="lerobot-robolab-"))
        if task not in gym_registry.registry:
            auto_register_droid_envs(task=task, cameras=getattr(camera_presets, cameras))
        env_cfg = parse_env_cfg(task, device=device, seed=seed, num_envs=num_envs, use_fabric=True)
        # No HDF5 recordings: LeRobot records episodes itself.
        env_cfg.recorders = None
        self.env, env_cfg = create_env(env_cfg, instruction_type=instruction_type)
        self.task_description = instruction_text(env_cfg.instruction)
        self.num_envs = num_envs
        self.metadata = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED, "render_fps": CONTROL_FPS}
        self._max_episode_steps = int(self.env.max_episode_length)
        self._image_size = (
            (observation_height, observation_width) if observation_height and observation_width else None
        )
        self._cameras = list(self.env.observation_manager.active_terms["image_obs"])
        height, width = self._image_size or self._native_size()
        self.single_observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {name: spaces.Box(0, 255, (height, width, 3), np.uint8) for name in self._cameras}
                ),
                "agent_pos": spaces.Box(-np.inf, np.inf, (len(JOINT_NAMES),), np.float32),
            }
        )
        self.single_action_space = spaces.Box(-np.inf, np.inf, (len(JOINT_NAMES),), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def _native_size(self) -> tuple[int, int]:
        sensor = self.env.scene[self._cameras[0]]
        return int(sensor.cfg.height), int(sensor.cfg.width)

    def _observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Return batched NumPy observations from Isaac Lab's CUDA tensors."""
        import torch

        pixels = {}
        for name in self._cameras:
            images = obs["image_obs"][name]
            if self._image_size is not None:
                images = torch.nn.functional.interpolate(
                    images.permute(0, 3, 1, 2).float(), self._image_size, mode="bilinear", antialias=True
                )
                images = images.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            pixels[name] = images.cpu().numpy()
        proprio = obs["proprio_obs"]
        state = torch.cat([proprio["arm_joint_pos"], proprio["gripper_pos"]], dim=-1)
        return {"pixels": pixels, "agent_pos": state.float().cpu().numpy()}

    def reset(
        self, *, seed: int | list[int | None] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset every env. One world has one random generator: only the first seed is used."""
        first = seed[0] if isinstance(seed, list) else seed
        # Unfreeze the envs that finished the previous episode, as RoboLab's runner does between runs.
        self.env.reset_eval_state()
        self.env.reset(seed=first)
        # RoboLab's episode runner resets twice before each episode. On Isaac Sim 6.0.1 one reset already renders
        # the reset scene; the second keeps episodes identical to RoboLab's own runs.
        obs, _ = self.env.reset()
        return self._observation(obs), {}

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step every env; RoboLab holds finished envs in their final state."""
        import torch

        obs, reward, _, _, _ = self.env.step(
            torch.as_tensor(actions, dtype=torch.float32, device=self.env.device)
        )
        frozen = self.env._frozen_envs.cpu().numpy()
        results = self.env._env_results
        success = np.array([bool(frozen[i] and results.get(i)) for i in range(self.num_envs)])
        info = {"is_success": success}
        return self._observation(obs), reward.cpu().numpy(), success, frozen & ~success, info

    def close(self, **kwargs: Any) -> None:
        """Close the task's world; the Kit app stays up for the next task."""
        self.env.close()


def vector_env_fns(
    task: str,
    n_envs: int,
    cameras: str = DEFAULT_CAMERAS,
    instruction_type: str = "default",
    observation_height: int | None = None,
    observation_width: int | None = None,
) -> dict[str, dict[int, Callable[[], gym.vector.VectorEnv]]]:
    """Return one factory per RoboLab task, as ``{task_name: {0: factory}}``.

    Args:
        task: Comma-separated task names (e.g. ``"BananaInBowlTask"``), or empty for all 120.
        n_envs: Envs per task, all in one Isaac Lab world.
        cameras: A camera preset from ``robolab.registrations.droid.camera_presets``.
        instruction_type: Which of each task's instructions to give: ``default``, ``vague`` or ``specific``.
        observation_height, observation_width: Resize camera frames on the GPU to this size; native if unset.
    """
    known = _task_metadata()
    names = [name.strip() for name in task.split(",") if name.strip()] or sorted(known)
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"Unknown RoboLab tasks: {unknown}. Available tasks: {sorted(known)}")
    make = partial(
        RoboLabVectorEnv,
        num_envs=n_envs,
        cameras=cameras,
        instruction_type=instruction_type,
        observation_height=observation_height,
        observation_width=observation_width,
    )
    return {name: {0: partial(make, name)} for name in names}


def conventions(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return RoboLab's control rate, joint-position control and DROID's names; LeRobot's DROID records ``Franka``."""
    return {
        "fps": CONTROL_FPS,
        "control_mode": "position",
        "state_names": JOINT_NAMES,
        "action_names": JOINT_NAMES,
        "robot_type": "Franka",
    }
