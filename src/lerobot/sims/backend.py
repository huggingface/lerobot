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
import importlib
import importlib.metadata
import multiprocessing
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Connection
from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import CloudpickleWrapper

from lerobot.env_server.contracts import EnvDescriptor, StepResult
from lerobot.sims.adapters import (
    canonical_observation,
    canonical_observation_space,
    check_explicit_reset,
    check_step_info,
    env_attribute,
    episode_limit,
)
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
    def __init__(self, env_fns: list[Callable[[], gym.Env]]) -> None:
        self.env_fns = env_fns


def make_env_factories(
    config: BackendConfig, num_envs: int = 1
) -> dict[str, dict[int, list[Callable[[], gym.Env]]]]:
    """Collect native factories without building vector environments or starting a session."""
    if config.type == "toy":
        return {"reach": {0: [ToyEnv] * num_envs}}
    module_name = "libero" if config.type in {"libero_plus", "robocerebra"} else config.type
    module = importlib.import_module(f"lerobot.sims.native.{module_name}")
    kwargs = dict(config.kwargs)
    if config.type == "libero_plus":
        kwargs["is_libero_plus"] = True
    factories = getattr(module, f"create_{module_name}_envs")(
        task=config.task, n_envs=num_envs, env_cls=_Factories, **kwargs
    )
    return {
        group: {task_id: batch.env_fns for task_id, batch in tasks.items()}
        for group, tasks in factories.items()
    }


def make_scalar_env(config: BackendConfig, task_id: int = 0, group: str | None = None) -> gym.Env:
    """Build exactly one native world. Its caller owns reset, stepping and close."""
    factories = make_env_factories(config)
    group = group or next(iter(factories))
    try:
        env_fn = factories[group][task_id][0]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unknown task: {group}/{task_id}") from exc
    return env_fn()


def describe_env(config: BackendConfig, env: Any, tasks: dict[str, list[str]]) -> EnvDescriptor:
    """Describe native spaces without resetting, stepping or constructing an execution session."""
    if not isinstance(env.observation_space, spaces.Dict):
        raise ValueError(f"Simulation requires dict observations, got {env.observation_space}")
    check_explicit_reset(env)
    observation_space = canonical_observation_space(env.observation_space)
    features: list[FeatureSpec] = []
    for name, feature_space in observation_space.spaces.items():
        assert isinstance(feature_space, spaces.Box)
        rgb = feature_space.dtype == np.uint8 and len(feature_space.shape) == 3
        names = config.state_names if name == "observation.state" else ()
        if not names and len(feature_space.shape) == 1:
            names = tuple(f"state_{i}" for i in range(feature_space.shape[0]))
        features.append(
            FeatureSpec(
                name,
                feature_space.shape,
                feature_space.dtype.name,
                "rgb" if rgb else "tensor",
                names,
                config.semantics,
            )
        )
    if not isinstance(env.action_space, spaces.Box) or len(env.action_space.shape) != 1:
        raise ValueError(f"Simulation requires a 1-D Box action space, got {env.action_space}")
    action_dim = env.action_space.shape[0]
    action_names = config.action_names or tuple(f"command_{i}" for i in range(action_dim))
    return EnvDescriptor(
        config.type,
        config.semantics,
        tuple(features),
        FeatureSpec("action", (action_dim,), "float32", names=action_names, semantics=config.semantics),
        config.control,
        config.fps,
        ("lockstep", "realtime"),
        tasks,
        episode_limit(env),
        {
            "numpy": np.__version__,
            "gymnasium": importlib.metadata.version("gymnasium"),
            **env_attribute(env, "metadata", {}).get("sim_build", {}),
        },
        config.gripper_indices,
        config.command_retention_indices,
        config.robot_type,
        asdict(config),
        dict(env_attribute(env, "metadata", {}).get("asset_revisions", {})),
    )


# Read from the worker after construction and reset.
_WORLD_ATTRS = ("task_description", "task", "_max_episode_steps", "metadata")


class _EnvStep(NamedTuple):
    """Last result cached for one environment."""

    observation: dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    task_description: str


def _world_worker(pipe: Connection, make: CloudpickleWrapper) -> None:
    """Build one world and run its reset/step commands until told to close."""
    env = None
    try:
        env = make()
        check_explicit_reset(env)
    except BaseException:
        pipe.send((False, traceback.format_exc()))
        if env is not None:
            env.close()
        pipe.close()
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
                    result = {
                        name: value
                        for name in _WORLD_ATTRS
                        if (value := env_attribute(env, name)) is not None
                    }
                    result["_max_episode_steps"] = episode_limit(env)
                if command in {"reset", "step"}:
                    task = env_attribute(env, "task_description", env_attribute(env, "task"))
                    if task is not None:
                        *values, info = result
                        result = (*values, {"task_description": task, **info})
                pipe.send((True, result))
            except Exception:
                pipe.send((False, traceback.format_exc()))
    finally:
        env.close()
        pipe.close()


class _ProcessWorld:
    """One world in its own process. A batch of them steps in parallel: send all, then receive all."""

    def __init__(self, make: Callable[[], Any]) -> None:
        context = multiprocessing.get_context("forkserver")
        self._pipe, child = context.Pipe()
        self._process = context.Process(
            target=_world_worker, args=(child, CloudpickleWrapper(make)), daemon=True
        )
        self._process.start()
        child.close()

    def ready(self) -> None:
        """Wait until the world is built, then read its spaces and attributes."""
        self._receive()
        self.observation_space, self.action_space = self._call("spaces")
        self._refresh()

    def _receive(self) -> Any:
        ok, value = self._pipe.recv()
        if not ok:
            raise RuntimeError(f"World process failed: {value}")
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
            self._process.join(5)
        self._pipe.close()


class Backend:
    """Own a batch of worlds and retain absorbing terminal transitions.

    With ``async_worlds``, each world runs in its own process and the batch steps in parallel.
    """

    def __init__(
        self,
        config: BackendConfig,
        num_envs: int = 1,
        group: str | None = None,
        task_id: int = 0,
        async_worlds: bool = False,
    ) -> None:
        """Initialize owned resources without advancing simulation time."""
        self.config = config
        self.async_worlds = async_worlds
        self.num_envs = num_envs
        self.envs: list[Any] = []
        self.last: list[_EnvStep] = []
        self.steps: np.ndarray = np.zeros(num_envs, dtype=np.int64)
        self.sim_time = 0.0
        factories = make_env_factories(config, num_envs)
        self.tasks = {key: [str(i) for i in tasks] for key, tasks in factories.items()}
        group = group or next(iter(factories))
        try:
            self._env_fns = factories[group][task_id]
        except KeyError as exc:
            raise ValueError(f"Unknown task: {group}/{task_id}") from exc
        self.group = group
        self.task_id = task_id

    def _ensure(self) -> None:
        if not self.envs:
            try:
                if self.async_worlds:
                    # Start every process before waiting, so the worlds build in parallel.
                    self.envs.extend(_ProcessWorld(env_fn) for env_fn in self._env_fns)
                    for world in self.envs:
                        world.ready()
                else:
                    for env_fn in self._env_fns:
                        self.envs.append(env_fn())
                        check_explicit_reset(self.envs[-1])
            except BaseException:
                self.close()
                raise

    def descriptor(self) -> EnvDescriptor:
        """Describe this task using its observation and action spaces."""
        self._ensure()
        return describe_env(self.config, self.envs[0], self.tasks)

    def reset(self, seeds: list[int | None] | None = None) -> StepResult:
        """Reset every world with explicit seeds and clear terminal latches."""
        self._ensure()
        seeds = seeds if seeds is not None else [None] * self.num_envs
        if len(seeds) != self.num_envs:
            raise ValueError("Seed count must match the batch size")
        self.last = []
        if self.async_worlds:
            for env, seed in zip(self.envs, seeds, strict=True):
                env.send("reset", seed)
            results = [env.receive(refresh=True) for env in self.envs]
        else:
            results = [env.reset(seed=seed) for env, seed in zip(self.envs, seeds, strict=True)]
        for env, (obs, info) in zip(self.envs, results, strict=True):
            check_step_info(info)
            task = str(
                info.get(
                    "task_description",
                    env_attribute(env, "task_description", env_attribute(env, "task", self.group)),
                )
            )
            success = bool(info.get("is_success", info.get("success", info.get("succeed", False))))
            self.last.append(_EnvStep(canonical_observation(obs), 0.0, False, False, success, task))
        self.steps[:] = 0
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
        running = []
        for i, env in enumerate(self.envs):
            if self.last[i].terminated or self.last[i].truncated:
                # Absorbing terminal state: keep the observation and flags, earn no further reward.
                self.last[i] = self.last[i]._replace(reward=0.0)
                continue
            if self.async_worlds:
                env.send("step", actions[i])
            running.append(i)
        for i in running:
            env = self.envs[i]
            obs, reward, terminated, truncated, info = (
                env.receive() if self.async_worlds else env.step(actions[i])
            )
            check_step_info(info)
            self.steps[i] += 1
            limit = episode_limit(env)
            truncated = bool(truncated or (limit is not None and self.steps[i] >= limit))
            self.last[i] = _EnvStep(
                canonical_observation(obs),
                float(reward),
                bool(terminated),
                truncated,
                bool(info.get("is_success", info.get("success", info.get("succeed", False)))),
                str(
                    info.get(
                        "task_description",
                        env_attribute(env, "task_description", env_attribute(env, "task", self.group)),
                    )
                ),
            )
        self.sim_time += 1 / self.config.fps
        return self.snapshot()

    def snapshot(self) -> StepResult:
        """Return owned batch arrays without advancing physics."""
        if not self.last:
            raise ValueError("Reset before observing")
        observations = {
            key: np.stack([result.observation[key] for result in self.last])
            for key in self.last[0].observation
        }
        return StepResult(
            observations,
            tuple(result.task_description for result in self.last),
            np.array([result.reward for result in self.last], dtype=np.float32),
            np.array([result.terminated for result in self.last]),
            np.array([result.truncated for result in self.last]),
            np.array([result.success for result in self.last]),
            self.sim_time,
            self.steps.copy(),
        )

    def render(self) -> list[np.ndarray]:
        """Return cached canonical RGB frames, including terminal frames."""
        if not self.last:
            raise ValueError("Reset before rendering")
        key = next(
            (
                name
                for name, value in self.last[0].observation.items()
                if value.dtype == np.uint8 and value.ndim == 3
            ),
            None,
        )
        if key is None:
            raise ValueError("Backend does not expose an RGB rendering")
        return [result.observation[key].copy() for result in self.last]

    def close(self) -> None:
        """Close every world, even when one of them fails to close."""
        envs, self.envs = self.envs, []
        with contextlib.ExitStack() as stack:
            for env in envs:
                stack.callback(env.close)


class ToyEnv(gym.Env):
    """Deterministic lightweight backend for protocol and rollout integration tests."""

    def __init__(self) -> None:
        """A 2-D point that starts near the origin and succeeds past x = 1."""
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

    def _obs(self) -> dict[str, np.ndarray]:
        return {"agent_pos": self.state.copy(), "pixels": self.render()}

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Place the point near the origin, from ``seed``."""
        self.state = np.random.default_rng(seed).uniform(-0.1, 0.1, 2).astype(np.float32)
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Move the point by ``action``."""
        self.state += action
        success = bool(self.state[0] >= 1)
        return self._obs(), float(success), success, False, {"is_success": success}

    def render(self) -> np.ndarray:
        """An 8x8 image whose brightness encodes x."""
        return np.full((8, 8, 3), int(np.clip(self.state[0] * 10 + 100, 0, 255)), dtype=np.uint8)

    def close(self) -> None:
        """Nothing to release."""
