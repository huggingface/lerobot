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

"""Batches of simulator environments, stepped explicitly; finished environments stay terminal until reset."""

import contextlib
import importlib
import logging
import multiprocessing
import traceback
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from multiprocessing.connection import Connection
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import CloudpickleWrapper, batch_space

from lerobot.sim.config import SimConfig
from lerobot.sim.integrations import simulator_required
from lerobot.sim.types import EnvDescriptor, StepResult
from lerobot.utils.feature_spec import FeatureSpec

logger = logging.getLogger(__name__)

# Environment attributes copied back from a worker process.
_ENV_ATTRS = ("task_description", "task")
# Episode length of an environment that declares none.
_DEFAULT_MAX_EPISODE_STEPS = 500


def _max_episode_steps(env: Any) -> int | None:
    """The episode limit an environment declares, looked up through Gymnasium wrappers."""
    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    try:
        limit = get_wrapper_attr("_max_episode_steps") if get_wrapper_attr else env._max_episode_steps
    except AttributeError:
        return None
    return None if limit is None else int(limit)


def _build_env(sim_type: str, make: Callable[[], Any]) -> Any:
    with simulator_required(sim_type):
        return make()


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
                    result["_max_episode_steps"] = _max_episode_steps(env)
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

    def _exited(self) -> RuntimeError:
        self._process.join(1)
        return RuntimeError(
            f"Environment process exited with code {self._process.exitcode}. If its simulator cannot run in worker "
            "processes on this platform, run it in the main process (use_async_envs=false, e.g. "
            "--eval.use_async_envs=false)"
        )

    def _receive(self) -> Any:
        try:
            ok, value = self._pipe.recv()
        except EOFError:
            raise self._exited() from None
        if not ok:
            raise RuntimeError(f"Environment process failed: {value}")
        return value

    def _call(self, command: str, argument: Any = None) -> Any:
        self.send(command, argument)
        return self._receive()

    def _refresh(self) -> None:
        for name, value in self._call("attrs").items():
            setattr(self, name, value)

    def send(self, command: str, argument: Any = None) -> None:
        try:
            self._pipe.send((command, argument))
        except BrokenPipeError:
            raise self._exited() from None

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


class VectorEnv(gym.vector.VectorEnv):
    """``num_envs`` environments of one simulator task, stepped explicitly.

    The simulator integration ``lerobot.sim.integrations.<config.type>`` supplies the environments and their
    canonical observations. A finished environment keeps its terminal observation and flags, and earns no
    reward, until ``reset``. With ``use_async_envs``, each environment runs in its own process and the batch
    steps in parallel.
    """

    canonical_observations = True

    def __init__(
        self,
        config: SimConfig,
        num_envs: int = 1,
        task_group: str | None = None,
        task_id: int | None = None,
        use_async_envs: bool = False,
        descriptor: EnvDescriptor | None = None,
    ) -> None:
        """Select the task. Given its ``descriptor``, the simulators are built on first use rather than now."""
        with simulator_required(config.type):
            integration = importlib.import_module(f"lerobot.sim.integrations.{config.type}")
            # env_cls=list returns each task's environment factories without building them.
            factories = integration.create_envs(
                task=config.task, n_envs=num_envs, env_cls=list, **config.kwargs
            )
        self._canonical_observation = integration.canonical_observation
        self.config = config
        self.num_envs = num_envs
        self.use_async_envs = use_async_envs
        self.tasks = {group: [str(task) for task in tasks] for group, tasks in factories.items()}
        self.task_group = task_group or next(iter(factories))
        tasks = factories.get(self.task_group, {})
        # The suite's first task unless one is named: a suite may be filtered (LIBERO's task_ids).
        self.task_id: int = next(iter(tasks), 0) if task_id is None else task_id
        if self.task_id not in tasks:
            raise ValueError(f"Unknown task: {self.task_group}/{self.task_id}")
        self._env_fns: list[Callable[[], Any]] = [
            partial(_build_env, config.type, fn) for fn in tasks[self.task_id]
        ]
        self._envs: list[Any] = []
        self._last: list[tuple[dict[str, np.ndarray], float, bool, bool, bool]] = []
        self._elapsed_steps: np.ndarray = np.zeros(num_envs, dtype=np.int64)
        self._max_steps: list[int] = []
        self._sim_time = 0.0
        self._latest: StepResult | None = None
        self.closed = False
        try:
            self.descriptor = descriptor or self._describe()
        except BaseException:
            self._close_envs()
            raise
        self.single_observation_space = self.descriptor.observation_space()
        self.single_action_space = self.descriptor.action_space()
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.metadata = {"render_fps": config.fps, "autoreset_mode": gym.vector.AutoresetMode.DISABLED}
        self.render_mode = "rgb_array"

    def build(self) -> None:
        """Build the simulators unless already built; ``lerobot-eval`` calls it to prefetch the next task."""
        if self._envs:
            return
        try:
            if self.use_async_envs:
                # Start every process before waiting, so the environments build in parallel.
                self._envs.extend(_ProcessEnv(fn) for fn in self._env_fns)
                for env in self._envs:
                    env.ready()
            else:
                for fn in self._env_fns:
                    self._envs.append(fn())
        except BaseException:
            self._close_envs()
            raise
        limits = [_max_episode_steps(env) for env in self._envs]
        if None in limits:
            logger.warning(
                "%s declares no episode limit; episodes are truncated after %d steps",
                self.config.type,
                _DEFAULT_MAX_EPISODE_STEPS,
            )
        self._max_steps = [limit or _DEFAULT_MAX_EPISODE_STEPS for limit in limits]

    def _describe(self) -> EnvDescriptor:
        """Describe the canonical observations and actions of the built environments."""
        self.build()
        env = self._envs[0]
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError(
                "LeRobot simulation needs dict observations "
                f"(pixels, agent_pos, environment_state), got {env.observation_space}"
            )
        if not isinstance(env.action_space, spaces.Box) or len(env.action_space.shape) != 1:
            raise ValueError(f"LeRobot simulation needs a 1-D Box action space, got {env.action_space}")
        config = self.config
        features = []
        for name, value in self._canonical_observation(env.observation_space.sample()).items():
            names = config.state_names if name == "observation.state" else ()
            if not names and value.ndim == 1:
                names = tuple(f"state_{index}" for index in range(value.shape[0]))
            kind = "rgb" if value.dtype == np.uint8 else "tensor"
            features.append(FeatureSpec(name, value.shape, value.dtype.name, kind, names, config.semantics))
        dim = env.action_space.shape[0]
        action_names = config.action_names or tuple(f"command_{index}" for index in range(dim))
        return EnvDescriptor(
            config.type,
            config.semantics,
            tuple(features),
            FeatureSpec("action", (dim,), "float32", names=action_names, semantics=config.semantics),
            config.control_mode,
            config.fps,
            self.tasks,
            self._max_steps[0],
            config.robot_type,
            asdict(config),
        )

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset every environment with explicit seeds and clear terminal latches."""
        if options:
            raise ValueError("This environment does not define reset options")
        if isinstance(seed, int):
            seeds: list[int | None] = [seed + index for index in range(self.num_envs)]
        else:
            seeds = list(seed) if seed is not None else [None] * self.num_envs
        if len(seeds) != self.num_envs:
            raise ValueError("Seed count must match the batch size")
        self.build()
        if self.use_async_envs:
            for env, env_seed in zip(self._envs, seeds, strict=True):
                env.send("reset", env_seed)
            results = [env.receive(refresh=True) for env in self._envs]
        else:
            results = [env.reset(seed=env_seed) for env, env_seed in zip(self._envs, seeds, strict=True)]
        self._last = [(self._canonical_observation(obs), 0.0, False, False, False) for obs, _ in results]
        self._elapsed_steps[:] = 0
        self._sim_time = 0.0
        result = self._snapshot()
        return result.obs, result.info()

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Advance each unfinished environment once; finished ones keep their terminal transition."""
        if not self._last:
            raise ValueError("Reset before stepping")
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, *self.descriptor.action_feature.shape)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite floating-point actions of shape {expected}")
        running = []
        for i, env in enumerate(self._envs):
            if self._last[i][2] or self._last[i][3]:
                # A finished environment keeps its terminal observation and flags, and earns no reward.
                self._last[i] = (self._last[i][0], 0.0, *self._last[i][2:])
                continue
            if self.use_async_envs:
                env.send("step", actions[i])
            running.append(i)
        for i in running:
            env = self._envs[i]
            obs, reward, terminated, truncated, info = (
                env.receive() if self.use_async_envs else env.step(actions[i])
            )
            self._elapsed_steps[i] += 1
            truncated = bool(truncated or self._elapsed_steps[i] >= self._max_steps[i])
            self._last[i] = (
                self._canonical_observation(obs),
                float(reward),
                bool(terminated),
                truncated,
                bool(info.get("is_success", info.get("success", False))),
            )
        self._sim_time += 1 / self.config.fps
        result = self._snapshot()
        return result.obs, result.reward, result.terminated, result.truncated, result.info()

    def _snapshot(self) -> StepResult:
        obs = {key: np.stack([last[0][key] for last in self._last]) for key in self._last[0][0]}
        task = tuple(
            str(getattr(env, "task_description", getattr(env, "task", self.task_group))) for env in self._envs
        )
        self._latest = StepResult(
            obs,
            task,
            np.array([last[1] for last in self._last], dtype=np.float32),
            np.array([last[2] for last in self._last]),
            np.array([last[3] for last in self._last]),
            np.array([last[4] for last in self._last]),
            self._sim_time,
            self._elapsed_steps.copy(),
        )
        return self._latest

    @property
    def latest(self) -> StepResult:
        """The result of the last reset or step."""
        if self._latest is None:
            raise RuntimeError("Reset before reading the environment")
        return self._latest

    def render(self) -> list[np.ndarray]:
        """Return the latest RGB frame of every environment, terminal frames included, without rendering."""
        return self.latest.frames()

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Return a supported per-environment attribute."""
        if args or kwargs:
            raise TypeError("Environment attributes do not take arguments")
        if name == "_max_episode_steps":
            self.build()
            return (self._max_steps[0],) * self.num_envs
        if name == "task":
            return (self.task_group,) * self.num_envs
        if name == "task_description":
            return self.latest.task
        if name == "render":
            return tuple(self.render())
        raise AttributeError(f"Environment attribute is not exposed: {name}")

    def get_attr(self, name: str) -> tuple[Any, ...]:
        """Return a supported per-environment attribute."""
        return self.call(name)

    def _close_envs(self) -> None:
        envs, self._envs = self._envs, []
        with contextlib.ExitStack() as stack:
            for env in envs:
                stack.callback(env.close)

    def close_extras(self, **kwargs: Any) -> None:
        """Release simulator resources."""
        self._close_envs()
        self.closed = True
