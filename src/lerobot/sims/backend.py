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

"""Environment batches with explicit reset and absorbing terminal transitions."""

import importlib
import importlib.metadata
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces

from lerobot.env_server.contracts import EnvDescriptor, StepResult
from lerobot.sims.adapters import canonical_observation
from lerobot.transport.wire.features import FeatureSpec


@dataclass
class BackendConfig:
    """Select a native simulator, canonical conventions, and factory options."""

    type: str = "toy"
    task: str = "reach"
    fps: float = 20
    control: str = "delta"
    semantics: str = "toy-delta-v1"
    kwargs: dict[str, Any] = field(default_factory=dict)
    state_names: tuple[str, ...] = ()
    action_names: tuple[str, ...] = ()
    gripper_indices: tuple[int, ...] = ()
    command_retention_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Normalize component order and reject invalid controller declarations."""
        self.state_names = tuple(self.state_names)
        self.action_names = tuple(self.action_names)
        self.gripper_indices = tuple(self.gripper_indices)
        self.command_retention_indices = tuple(self.command_retention_indices)
        if (
            self.control not in {"position", "delta", "eef_pose", "velocity"}
            or not self.semantics
            or not np.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("Backend requires explicit semantics, controller, and positive FPS")


class _Factories:
    def __init__(self, env_fns):
        """Initialize owned resources without advancing simulation time."""
        self.env_fns = env_fns


class Backend:
    """Own a batch of worlds and retain absorbing terminal transitions."""

    def __init__(self, config: BackendConfig, num_envs: int = 1, group: str | None = None, task_id: int = 0):
        """Initialize owned resources without advancing simulation time."""
        self.config = config
        self.num_envs = num_envs
        self.envs: list[Any] = []
        self.last: list[Any] = []
        self.steps: np.ndarray = np.zeros(num_envs, dtype=np.int64)
        self.sim_time = 0.0
        if config.type == "toy":
            factories = {"reach": {0: _Factories([lambda: ToyEnv() for _ in range(num_envs)])}}
        else:
            module_name = "libero" if config.type in {"libero_plus", "robocerebra"} else config.type
            module = importlib.import_module(f"lerobot.sims.native.{module_name}")
            kwargs = dict(config.kwargs)
            if config.type == "libero_plus":
                kwargs["is_libero_plus"] = True
            factories = getattr(module, f"create_{module_name}_envs")(
                task=config.task, n_envs=num_envs, env_cls=_Factories, **kwargs
            )
        self.tasks = {key: [str(i) for i in tasks] for key, tasks in factories.items()}
        group = group or next(iter(factories))
        try:
            self.fns = factories[group][task_id].env_fns
        except KeyError as exc:
            raise ValueError(f"Unknown task: {group}/{task_id}") from exc
        self.group = group
        self.task_id = task_id

    def _ensure(self) -> None:
        if not self.envs:
            try:
                for fn in self.fns:
                    self.envs.append(fn())
            except BaseException:
                self.close()
                raise

    def descriptor(self) -> EnvDescriptor:
        """Describe this task using its observation and action spaces."""
        self._ensure()
        env = self.envs[0]
        obs = canonical_observation(env.observation_space.sample(), self.config.type)
        features = []
        for name, array in obs.items():
            rgb = array.dtype == np.uint8
            names = self.config.state_names if name == "observation.state" else ()
            if not names and array.ndim == 1:
                names = tuple(f"state_{i}" for i in range(array.shape[0]))
            features.append(
                FeatureSpec(
                    name,
                    array.shape,
                    array.dtype.name,
                    "rgb" if rgb else "tensor",
                    names,
                    self.config.semantics,
                )
            )
        dim = env.action_space.shape[0]
        action_names = self.config.action_names or tuple(f"command_{i}" for i in range(dim))
        return EnvDescriptor(
            self.config.type,
            self.config.type,
            self.config.semantics,
            tuple(features),
            FeatureSpec("action", (dim,), "float32", names=action_names, semantics=self.config.semantics),
            self.config.control,
            "position" if self.config.control == "position" else "command",
            self.config.fps,
            ("lockstep", "realtime"),
            self.tasks,
            int(getattr(env, "_max_episode_steps", 500)),
            {"numpy": np.__version__, **{name: importlib.metadata.version(name) for name in ("gymnasium",)}},
            self.config.gripper_indices,
            self.config.command_retention_indices,
        )

    def reset(self, seeds: list[int | None] | None = None, env_ids: list[int] | None = None) -> StepResult:
        """Reset selected worlds with explicit seeds and clear terminal latches."""
        self._ensure()
        ids = list(range(self.num_envs)) if env_ids is None else env_ids
        if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < self.num_envs for i in ids):
            raise ValueError("Invalid reset environment indices")
        seeds = seeds if seeds is not None else [None] * len(ids)
        if len(seeds) != len(ids):
            raise ValueError("Seed count must match reset environment count")
        if not self.last and len(ids) != self.num_envs:
            raise ValueError("Initial reset must initialize the full batch")
        if not self.last:
            self.last = [None] * self.num_envs
        for i, seed in zip(ids, seeds, strict=True):
            obs, info = self.envs[i].reset(seed=seed)
            self.last[i] = (canonical_observation(obs, self.config.type), 0.0, False, False, False)
            self.steps[i] = 0
        if len(ids) == self.num_envs:
            self.sim_time = 0.0
        return self.snapshot()

    def step(self, actions: np.ndarray) -> StepResult:
        """Advance each unfinished world once and retain terminal transitions."""
        self._ensure()
        if not self.last:
            raise ValueError("Reset before stepping")
        actions = np.asarray(actions)
        expected = (self.num_envs, *self.envs[0].action_space.shape)
        if actions.shape != expected or actions.dtype.kind != "f" or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite floating-point actions of shape {expected}")
        for i, env in enumerate(self.envs):
            if self.last[i][2] or self.last[i][3]:
                continue
            obs, reward, terminated, truncated, info = env.step(actions[i])
            self.steps[i] += 1
            truncated = bool(truncated or self.steps[i] >= getattr(env, "_max_episode_steps", 500))
            self.last[i] = (
                canonical_observation(obs, self.config.type),
                float(reward),
                bool(terminated),
                truncated,
                bool(info.get("is_success", info.get("success", False))),
            )
        self.sim_time += 1 / self.config.fps
        return self.snapshot()

    def snapshot(self) -> StepResult:
        """Return owned batch arrays without advancing physics."""
        if not self.last:
            raise ValueError("Reset before observing")
        obs = {key: np.stack([v[0][key] for v in self.last]) for key in self.last[0][0]}
        task = tuple(
            str(getattr(env, "task_description", getattr(env, "task", self.group))) for env in self.envs
        )
        return StepResult(
            obs,
            task,
            np.array([v[1] for v in self.last], dtype=np.float32),
            np.array([v[2] for v in self.last]),
            np.array([v[3] for v in self.last]),
            np.array([v[4] for v in self.last]),
            self.sim_time,
            self.steps.copy(),
        )

    def render(self) -> list[np.ndarray]:
        """Return cached canonical RGB frames, including terminal frames."""
        if not self.last:
            raise ValueError("Reset before rendering")
        key = next((k for k, v in self.last[0][0].items() if v.dtype == np.uint8 and v.ndim == 3), None)
        if key is None:
            raise ValueError("Backend does not expose an RGB rendering")
        # Cached canonical frames preserve the terminal image without additional physics/render work.
        return [v[0][key].copy() for v in self.last]

    def close(self) -> None:
        """Release owned simulator resources and transport declarations."""
        for env in self.envs:
            env.close()
        self.envs.clear()


class ToyEnv:
    """Deterministic lightweight backend for protocol and rollout integration tests."""

    def __init__(self):
        """Initialize owned resources without advancing simulation time."""
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

    def _obs(self):
        return {"agent_pos": self.state.copy(), "pixels": self.render()}

    def reset(self, seed=None):
        """Reset selected worlds with explicit seeds and clear terminal latches."""
        self.state = np.random.default_rng(seed).uniform(-0.1, 0.1, 2).astype(np.float32)
        return self._obs(), {}

    def step(self, action):
        """Advance each unfinished world once and retain terminal transitions."""
        self.state += action
        success = bool(self.state[0] >= 1)
        return self._obs(), float(success), success, False, {"is_success": success}

    def render(self):
        """Return cached canonical RGB frames, including terminal frames."""
        return np.full((8, 8, 3), int(np.clip(self.state[0] * 10 + 100, 0, 255)), dtype=np.uint8)

    def close(self) -> None:
        """Release owned simulator resources and transport declarations."""
        pass
