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

import contextlib
import multiprocessing
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import CloudpickleWrapper

from lerobot.sims.native import load_native_integration
from lerobot.sims.types import StepResult


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
    robot_type: str = ""

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
        self.env_fns = env_fns


# Attributes an environment may expose, read back from its process after it is built and reset.
_ENV_ATTRS = ("task_description", "task", "_max_episode_steps")


def _env_worker(pipe: Connection, make: CloudpickleWrapper) -> None:
    """Build one environment and run its reset/step commands until told to close."""
    try:
        env = make()
    except BaseException:
        pipe.send((False, traceback.format_exc()))
        return
    pipe.send((True, None))
    try:
        while True:
            command, argument = pipe.recv()
            if command == "close":
                break
            try:
                if command == "reset":
                    result: Any = env.reset(seed=argument)
                elif command == "step":
                    result = env.step(argument)
                elif command == "spaces":
                    result = (env.observation_space, env.action_space)
                else:
                    result = {name: getattr(env, name) for name in _ENV_ATTRS if hasattr(env, name)}
                pipe.send((True, result))
            except Exception:
                pipe.send((False, traceback.format_exc()))
    finally:
        env.close()
        pipe.close()


class _ProcessEnv:
    """One environment in its own process. A batch steps in parallel: send all, then receive all."""

    def __init__(self, make: Callable[[], Any]):
        context = multiprocessing.get_context("forkserver")
        self._pipe, child = context.Pipe()
        self._process = context.Process(
            target=_env_worker, args=(child, CloudpickleWrapper(make)), daemon=True
        )
        self._process.start()
        child.close()

    def ready(self) -> None:
        """Wait until the environment is built, then read its spaces and attributes."""
        self._receive()
        self.observation_space, self.action_space = self._call("spaces")
        self._refresh()

    def _receive(self) -> Any:
        ok, value = self._pipe.recv()
        if not ok:
            raise RuntimeError(f"Environment process failed: {value}")
        return value

    def _call(self, command: str, argument: Any = None) -> Any:
        self._pipe.send((command, argument))
        return self._receive()

    def _refresh(self) -> None:
        for name, value in self._call("attrs").items():
            setattr(self, name, value)

    def send(self, command: str, argument: Any = None) -> None:
        self._pipe.send((command, argument))

    def receive(self, refresh: bool = False) -> Any:
        result = self._receive()
        if refresh:
            self._refresh()
        return result

    def reset(self, seed: int | None = None) -> Any:
        self.send("reset", seed)
        return self.receive(refresh=True)

    def step(self, action: np.ndarray) -> Any:
        self.send("step", action)
        return self.receive()

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._pipe.send(("close", None))
        self._process.join(10)
        if self._process.is_alive():
            self._process.terminate()
        self._pipe.close()


class Backend:
    """Own a batch of environments and retain absorbing terminal transitions.

    With ``use_async_envs``, each environment runs in its own process and the batch steps in parallel.
    """

    def __init__(
        self,
        config: BackendConfig,
        num_envs: int = 1,
        group: str | None = None,
        task_id: int = 0,
        use_async_envs: bool = False,
    ):
        """Initialize owned resources without advancing simulation time."""
        self.config = config
        self.use_async_envs = use_async_envs
        self.num_envs = num_envs
        self.envs: list[Any] = []
        self.last: list[Any] = []
        self.steps: np.ndarray = np.zeros(num_envs, dtype=np.int64)
        self.sim_time = 0.0
        integration = load_native_integration(config.type)
        self.canonical_observation = integration.canonical_observation
        factories = integration.create_envs(
            task=config.task,
            n_envs=num_envs,
            env_cls=_Factories,
            **dict(config.kwargs),
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
                if self.use_async_envs:
                    # Start every process before waiting, so the environments build in parallel.
                    self.envs.extend(_ProcessEnv(fn) for fn in self.fns)
                    for env in self.envs:
                        env.ready()
                else:
                    for fn in self.fns:
                        self.envs.append(fn())
            except BaseException:
                self.close()
                raise

    def single_spaces(self) -> tuple[spaces.Dict, spaces.Box]:
        """Return canonical single-environment observation and action spaces."""
        self._ensure()
        env = self.envs[0]
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError(
                "LeRobot simulation needs dict observations "
                f"(pixels, agent_pos, environment_state), got {env.observation_space}"
            )
        obs = self.canonical_observation(env.observation_space.sample())
        if not isinstance(env.action_space, spaces.Box) or len(env.action_space.shape) != 1:
            raise ValueError(f"LeRobot simulation needs a 1-D Box action space, got {env.action_space}")
        observation_space = spaces.Dict(
            {
                name: spaces.Box(
                    0 if array.dtype == np.uint8 else -np.inf,
                    255 if array.dtype == np.uint8 else np.inf,
                    array.shape,
                    array.dtype,
                )
                for name, array in obs.items()
            }
        )
        return observation_space, env.action_space

    @property
    def max_episode_steps(self) -> int:
        """Maximum task duration declared by the selected environment."""
        self._ensure()
        return int(getattr(self.envs[0], "_max_episode_steps", 500))

    def reset(self, seeds: list[int | None] | None = None) -> StepResult:
        """Reset every environment with explicit seeds and clear terminal latches."""
        self._ensure()
        seeds = seeds if seeds is not None else [None] * self.num_envs
        if len(seeds) != self.num_envs:
            raise ValueError("Seed count must match the batch size")
        self.last = []
        if self.use_async_envs:
            for env, seed in zip(self.envs, seeds, strict=True):
                env.send("reset", seed)
            results = [env.receive(refresh=True) for env in self.envs]
        else:
            results = [env.reset(seed=seed) for env, seed in zip(self.envs, seeds, strict=True)]
        for obs, _ in results:
            self.last.append((self.canonical_observation(obs), 0.0, False, False, False))
        self.steps[:] = 0
        self.sim_time = 0.0
        return self.snapshot()

    def step(self, actions: np.ndarray) -> StepResult:
        """Advance each unfinished environment once and retain terminal transitions."""
        self._ensure()
        if not self.last:
            raise ValueError("Reset before stepping")
        actions = np.asarray(actions)
        expected = (self.num_envs, *self.envs[0].action_space.shape)
        if actions.shape != expected or actions.dtype.kind != "f" or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite floating-point actions of shape {expected}")
        running = []
        for i, env in enumerate(self.envs):
            if self.last[i][2] or self.last[i][3]:
                # Absorbing terminal state: keep the observation and flags, earn no further reward.
                self.last[i] = (self.last[i][0], 0.0, *self.last[i][2:])
                continue
            if self.use_async_envs:
                env.send("step", actions[i])
            running.append(i)
        for i in running:
            env = self.envs[i]
            obs, reward, terminated, truncated, info = (
                env.receive() if self.use_async_envs else env.step(actions[i])
            )
            self.steps[i] += 1
            truncated = bool(truncated or self.steps[i] >= getattr(env, "_max_episode_steps", 500))
            self.last[i] = (
                self.canonical_observation(obs),
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
        """Close every environment, even when one of them fails to close."""
        envs, self.envs = self.envs, []
        with contextlib.ExitStack() as stack:
            for env in envs:
                stack.callback(env.close)
