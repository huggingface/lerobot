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
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import CloudpickleWrapper

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


# Attributes a world may expose, read back from its process after it is built and reset.
_WORLD_ATTRS = ("task_description", "task", "_max_episode_steps")


def _world_worker(pipe: Connection, make: CloudpickleWrapper) -> None:
    """Build one world and run its reset/step commands until told to close."""
    try:
        env = make()
    except BaseException as exc:
        pipe.send((False, repr(exc)))
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
                    result = {name: getattr(env, name) for name in _WORLD_ATTRS if hasattr(env, name)}
                pipe.send((True, result))
            except Exception as exc:
                pipe.send((False, repr(exc)))
    finally:
        env.close()
        pipe.close()


class _ProcessWorld:
    """One world in its own process. A batch of them steps in parallel: send all, then receive all."""

    def __init__(self, make: Callable[[], Any]):
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
    ):
        """Initialize owned resources without advancing simulation time."""
        self.config = config
        self.async_worlds = async_worlds
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
                if self.async_worlds:
                    # Start every process before waiting, so the worlds build in parallel.
                    self.envs.extend(_ProcessWorld(fn) for fn in self.fns)
                    for world in self.envs:
                        world.ready()
                else:
                    for fn in self.fns:
                        self.envs.append(fn())
            except BaseException:
                self.close()
                raise

    def descriptor(self) -> EnvDescriptor:
        """Describe this task using its observation and action spaces."""
        self._ensure()
        env = self.envs[0]
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError(
                "The simulator server needs dict observations, as in-process LeRobot does "
                f"(pixels, agent_pos, environment_state), got {env.observation_space}"
            )
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
        if not isinstance(env.action_space, spaces.Box) or len(env.action_space.shape) != 1:
            raise ValueError(f"The simulator server needs a 1-D Box action space, got {env.action_space}")
        dim = env.action_space.shape[0]
        action_names = self.config.action_names or tuple(f"command_{i}" for i in range(dim))
        return EnvDescriptor(
            self.config.type,
            self.config.semantics,
            tuple(features),
            FeatureSpec("action", (dim,), "float32", names=action_names, semantics=self.config.semantics),
            self.config.control,
            self.config.fps,
            ("lockstep", "realtime"),
            self.tasks,
            int(getattr(env, "_max_episode_steps", 500)),
            {"numpy": np.__version__, **{name: importlib.metadata.version(name) for name in ("gymnasium",)}},
            self.config.gripper_indices,
            self.config.command_retention_indices,
            asdict(self.config),
        )

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
        for obs, _ in results:
            self.last.append((canonical_observation(obs, self.config.type), 0.0, False, False, False))
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
            if self.last[i][2] or self.last[i][3]:
                # Absorbing terminal state: keep the observation and flags, earn no further reward.
                self.last[i] = (self.last[i][0], 0.0, *self.last[i][2:])
                continue
            if self.async_worlds:
                env.send("step", actions[i])
            running.append(i)
        for i in running:
            env = self.envs[i]
            obs, reward, terminated, truncated, info = (
                env.receive() if self.async_worlds else env.step(actions[i])
            )
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
