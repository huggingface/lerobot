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

"""Vector envs for simulators, with no autoreset: a finished env stays finished until `reset`."""

import contextlib
import importlib
import logging
import multiprocessing
import traceback
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
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
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.feature_spec import FeatureSpec, validate_array

logger = logging.getLogger(__name__)

# Env attributes copied from a worker process after it builds the env and after each reset.
_ENV_ATTRS = ("task_description", "task", "num_envs")
# Episode limit for an env that declares none.
_DEFAULT_MAX_EPISODE_STEPS = 500
# Gymnasium 0.29, which some simulators pin (RoboMME's ManiSkill), has no AutoresetMode; there a vector env always
# autoresets, so none from vector_env_fns is accepted.
_NO_AUTORESET = getattr(getattr(gym.vector, "AutoresetMode", None), "DISABLED", "disabled")


@dataclass(frozen=True)
class _LastStep:
    """One env's latest observation, reward, episode flags and metrics."""

    obs: dict[str, np.ndarray]
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    is_success: bool = False
    # The latest value of each metric the env reported in this episode.
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Whether the episode has ended."""
        return self.terminated or self.truncated


def _max_episode_steps(env: Any) -> int | None:
    """Return the env's episode limit, looked up through Gymnasium wrappers, or None if it has none."""
    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    try:
        limit = get_wrapper_attr("_max_episode_steps") if get_wrapper_attr else env._max_episode_steps
    except AttributeError:
        return None
    return None if limit is None else int(limit)


def _spaces(env: Any) -> tuple[spaces.Space, spaces.Space]:
    """Return the observation and action spaces of one env; for a vector env, those of each of its envs."""
    return (
        getattr(env, "single_observation_space", env.observation_space),
        getattr(env, "single_action_space", env.action_space),
    )


def _row(batch: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Return env `index`'s observation from a vector env's batched, possibly nested, observation."""
    return {
        key: _row(value, index) if isinstance(value, Mapping) else value[index]
        for key, value in batch.items()
    }


def _build_env(sim_type: str, make: Callable[[], Any]) -> Any:
    """Build one env, or one vector env, with `make`; reject a vector env that autoresets."""
    with simulator_required(sim_type):
        env = make()
    if isinstance(env, gym.vector.VectorEnv):
        mode = env.metadata.get("autoreset_mode")
        if mode is not _NO_AUTORESET:
            env.close()
            raise ValueError(
                f"A vector env from vector_env_fns must not autoreset; set metadata['autoreset_mode'] to "
                f"AutoresetMode.DISABLED (got {mode})"
            )
    return env


def _env_worker(pipe: Connection, make: CloudpickleWrapper) -> None:
    """Worker process: build one env, then run the commands sent on `pipe` until `close`."""
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
                    result = _spaces(env)
                else:  # "attrs"
                    result = {name: getattr(env, name) for name in _ENV_ATTRS if hasattr(env, name)}
                    result["_max_episode_steps"] = _max_episode_steps(env)
                pipe.send((True, result))
            except Exception:
                pipe.send((False, traceback.format_exc()))
    finally:
        env.close()
        pipe.close()


class _ProcessEnv:
    """One environment in its own process. A batch steps in parallel: send all, then receive all.

    Not Gymnasium's `AsyncVectorEnv`, which also steps finished environments (Gymnasium leaves that
    undefined) and builds an extra one in this process to read the spaces.
    """

    def __init__(self, make: Callable[[], Any]):
        context = multiprocessing.get_context("forkserver")
        self._pipe, child = context.Pipe()
        self._process = context.Process(
            target=_env_worker, args=(child, CloudpickleWrapper(make)), daemon=True
        )
        self._process.start()
        child.close()
        self._close_requested = False

    def ready(self) -> None:
        """Wait until the environment is built, then read its spaces and attributes."""
        self._receive()
        self.observation_space, self.action_space = self._call("spaces")
        self._refresh()

    def _exit_error(self) -> RuntimeError:
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
            raise self._exit_error() from None
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
            raise self._exit_error() from None

    def receive(self, refresh: bool = False) -> Any:
        result = self._receive()
        if refresh:
            self._refresh()
        return result

    def reset(self, seed: int | list[int | None] | None = None) -> Any:
        self.send("reset", seed)
        return self.receive(refresh=True)

    def step(self, action: np.ndarray) -> Any:
        self.send("step", action)
        return self.receive()

    def request_close(self) -> None:
        """Ask the process to close its environment and exit, without waiting. Asks only once."""
        if self._close_requested:
            return
        self._close_requested = True
        with contextlib.suppress(OSError):
            self._pipe.send(("close", None))

    def close(self) -> None:
        """Close the environment and its process, terminating the process if it still runs after 10 s."""
        self.request_close()
        self._process.join(10)
        if self._process.is_alive():
            self._process.terminate()
        self._pipe.close()


class LocalVectorEnv(gym.vector.VectorEnv):
    """`num_envs` envs of one simulator task, with no autoreset.

    A finished env is not stepped until `reset`: it keeps its last observation and gets zero reward. With
    `use_async_envs`, each env runs in its own worker process.

    An integration that defines `vector_env_fns` builds the whole batch as one Gymnasium vector env instead
    (a GPU simulator stepping every env at once), unless it returns None for this config, as EnvHub does for a
    repo without one. That env must not autoreset; it is stepped as a whole, and
    the rows of finished envs are ignored, so they stay finished as above. With `use_async_envs`, it runs in
    one worker process.
    """

    closed = False

    def __init__(
        self,
        config: SimConfig,
        num_envs: int = 1,
        task_group: str | None = None,
        task_id: int | None = None,
        use_async_envs: bool = False,
        descriptor: EnvDescriptor | None = None,
    ) -> None:
        """Select the task; given a `descriptor`, build the simulators on first use instead of now.

        Args:
            config (`SimConfig`):
                The simulator to run.
            num_envs (`int`, *optional*, defaults to `1`):
                Number of envs in the batch.
            task_group (`str`, *optional*):
                Suite to run, e.g. `libero_spatial`. Defaults to the integration's first one.
            task_id (`int`, *optional*):
                Task within the suite. Defaults to its first task.
            use_async_envs (`bool`, *optional*, defaults to `False`):
                Run each env in its own worker process.
            descriptor (`EnvDescriptor`, *optional*):
                The env's descriptor, if already known, as for another task of the same simulator.
        """
        with simulator_required(config.type):
            integration = importlib.import_module(f"lerobot.sim.integrations.{config.type}")
            vector_env_fns = getattr(integration, "vector_env_fns", None)
            # None from vector_env_fns: this config has no batched sim (e.g. an EnvHub repo without one).
            factories = None
            if vector_env_fns is not None:
                factories = vector_env_fns(task=config.task, n_envs=num_envs, **config.kwargs)
            # One vector env for the whole batch, instead of one env per batch entry.
            self._batched = factories is not None
            if factories is None:
                factories = integration.env_fns(task=config.task, n_envs=num_envs, **config.kwargs)
        self._canonical_observation = integration.canonical_observation
        self.config = config
        self.num_envs = num_envs
        self.use_async_envs = use_async_envs
        self.tasks: dict[str, list[int]] = {group: list(tasks) for group, tasks in factories.items()}
        self.task_group = task_group or next(iter(factories))
        task_factories = factories.get(self.task_group, {})
        # Default to the suite's first task id, not 0: a suite can be filtered (e.g. LIBERO's task_ids).
        self.task_id: int = next(iter(task_factories), 0) if task_id is None else task_id
        if self.task_id not in task_factories:
            raise ValueError(f"Unknown task: {self.task_group}/{self.task_id}")
        fns = [task_factories[self.task_id]] if self._batched else task_factories[self.task_id]
        self._env_fns: list[Callable[[], Any]] = [partial(_build_env, config.type, fn) for fn in fns]
        self._envs: list[Any] = []
        self._last: list[_LastStep] = []
        self._elapsed_steps: np.ndarray = np.zeros(num_envs, dtype=np.int64)
        self._max_steps: list[int] = []
        self._latest: StepResult | None = None
        try:
            self.descriptor = descriptor or self._describe()
        except BaseException:
            self._close_envs()
            raise
        self.single_observation_space = self.descriptor.observation_space()
        self.single_action_space = self.descriptor.action_space()
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.metadata = {
            "render_fps": config.fps,
            "autoreset_mode": _NO_AUTORESET,
            "canonical_observations": True,
        }
        self.render_mode = "rgb_array"

    @property
    def latest(self) -> StepResult:
        """The result of the last reset or step."""
        if self._latest is None:
            raise RuntimeError("Reset before reading the environment")
        return self._latest

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset every env; an int `seed` gives env `i` the seed `seed + i`."""
        if options:
            raise ValueError("This environment does not define reset options")
        if isinstance(seed, int):
            seeds: list[int | None] = [seed + index for index in range(self.num_envs)]
        else:
            seeds = list(seed) if seed is not None else [None] * self.num_envs
        if len(seeds) != self.num_envs:
            raise ValueError("Seed count must match the batch size")
        self._build()
        self._last = [_LastStep(self._canonical_observation(obs)) for obs in self._reset_envs(seeds)]
        self._elapsed_steps[:] = 0
        result = self._snapshot()
        return result.obs, result.info()

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step every unfinished env once; finished envs are not stepped until `reset`."""
        if not self._last:
            raise RuntimeError("Reset before stepping")
        actions = np.asarray(actions, dtype=np.float32)
        validate_array(actions, self.descriptor.action_feature, leading_shape=(self.num_envs,))
        # Finished envs keep their terminal observation and flags, with zero reward.
        self._last = [replace(last, reward=0.0) if last.done else last for last in self._last]
        running = [i for i, last in enumerate(self._last) if not last.done]
        for i, obs, reward, terminated, truncated, success, metrics in self._step_envs(actions, running):
            self._elapsed_steps[i] += 1
            self._last[i] = _LastStep(
                obs=self._canonical_observation(obs),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated or self._elapsed_steps[i] >= self._max_steps[i]),
                is_success=bool(success),
                metrics={**self._last[i].metrics, **{name: float(value) for name, value in metrics.items()}},
            )
        result = self._snapshot()
        return result.obs, result.reward, result.terminated, result.truncated, result.info()

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Return a supported per-environment attribute."""
        if args or kwargs:
            raise TypeError("Environment attributes do not take arguments")
        if name == "_max_episode_steps":
            # The descriptor may come from another task, whose episode limit can differ: read this task's own.
            self._build()
            return (self._max_steps[0],) * self.num_envs
        if name == "task":
            return (self.task_group,) * self.num_envs
        if name == "task_description":
            return self.latest.task
        if name == "render":
            return self.render()
        raise AttributeError(f"Environment attribute is not exposed: {name}")

    def get_attr(self, name: str) -> tuple[Any, ...]:
        """Return a supported per-environment attribute."""
        return self.call(name)

    def render(self) -> tuple[np.ndarray, ...]:
        """Return each env's latest RGB frame, taken from the last observation instead of a new render."""
        return self.latest.frames()

    def close(self, **kwargs: Any) -> None:
        """Release simulator resources, once.

        Defined here rather than through Gymnasium's `close_extras`: Gymnasium 0.29's `close` reads attributes
        its `VectorEnv.__init__` sets, which isn't called here.
        """
        if not self.closed:
            self._close_envs()
            self.closed = True

    def _build(self) -> None:
        """Build the simulators if not built yet, so each task builds on first use."""
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
            if self._batched and getattr(self._envs[0], "num_envs", None) != self.num_envs:
                raise ValueError(
                    f"vector_env_fns of {self.config.type} built {getattr(self._envs[0], 'num_envs', None)} "
                    f"envs, expected {self.num_envs}"
                )
        except BaseException:
            self._close_envs()
            raise
        # A vector env has one limit, for all of its envs.
        limits = [_max_episode_steps(env) for env in self._envs] * (self.num_envs if self._batched else 1)
        if None in limits:
            logger.warning(
                "%s declares no episode limit; episodes are truncated after %d steps",
                self.config.type,
                _DEFAULT_MAX_EPISODE_STEPS,
            )
        self._max_steps = [limit or _DEFAULT_MAX_EPISODE_STEPS for limit in limits]

    def _describe(self) -> EnvDescriptor:
        """Build the envs and describe the first one's observations (with LeRobot keys) and actions."""
        self._build()
        observation_space, action_space = _spaces(self._envs[0])
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError(
                "LeRobot simulation needs dict observations "
                f"(pixels, agent_pos, environment_state), got {observation_space}"
            )
        if not isinstance(action_space, spaces.Box) or len(action_space.shape) != 1:
            raise ValueError(f"LeRobot simulation needs a 1-D Box action space, got {action_space}")
        config = self.config
        features = []
        for name, value in self._canonical_observation(observation_space.sample()).items():
            names = config.state_names if name == OBS_STATE else ()
            if not names and value.ndim == 1:
                names = tuple(f"state_{index}" for index in range(value.shape[0]))
            kind = "rgb" if value.dtype == np.uint8 else "tensor"
            features.append(FeatureSpec(name, value.shape, value.dtype.name, kind, names, config.semantics))
        dim = action_space.shape[0]
        action_names = config.action_names or tuple(f"command_{index}" for index in range(dim))
        action = FeatureSpec(ACTION, (dim,), "float32", names=action_names, semantics=config.semantics)
        return EnvDescriptor(
            sim_type=config.type,
            semantics=config.semantics,
            features=tuple(features),
            action_feature=action,
            control_mode=config.control_mode,
            fps=config.fps,
            tasks=self.tasks,
            max_episode_steps=self._max_steps[0],
            robot_type=config.robot_type,
            config=asdict(config),
        )

    def _reset_envs(self, seeds: list[int | None]) -> list[Any]:
        """Reset the simulators, one seed per env, and return each env's raw observation."""
        if self._batched:
            batch, _ = self._envs[0].reset(seed=seeds)
            return [_row(batch, i) for i in range(self.num_envs)]
        if self.use_async_envs:
            # Send every reset before waiting, so the processes reset in parallel.
            for env, env_seed in zip(self._envs, seeds, strict=True):
                env.send("reset", env_seed)
            return [env.receive(refresh=True)[0] for env in self._envs]
        return [env.reset(seed=env_seed)[0] for env, env_seed in zip(self._envs, seeds, strict=True)]

    def _step_envs(self, actions: np.ndarray, running: list[int]) -> list[tuple[Any, ...]]:
        """Step the envs in `running`.

        Return `(i, obs, reward, terminated, truncated, success, metrics)` for each.
        """
        if self._batched:
            if not running:
                return []
            # One vector env steps every env; the rows of finished envs are dropped.
            batch, rewards, terminated, truncated, info = self._envs[0].step(actions)
            success = info.get("is_success", info.get("success", np.zeros(self.num_envs, dtype=bool)))
            metrics = info.get("metrics") or {}
            return [
                (i, _row(batch, i), rewards[i], terminated[i], truncated[i], success[i], _row(metrics, i))
                for i in running
            ]
        if self.use_async_envs:
            # Send every step before waiting, so the processes step in parallel.
            for i in running:
                self._envs[i].send("step", actions[i])
            results = [self._envs[i].receive() for i in running]
        else:
            results = [self._envs[i].step(actions[i]) for i in running]
        return [
            (
                i,
                obs,
                reward,
                terminated,
                truncated,
                info.get("is_success", info.get("success", False)),
                info.get("metrics") or {},
            )
            for i, (obs, reward, terminated, truncated, info) in zip(running, results, strict=True)
        ]

    def _snapshot(self) -> StepResult:
        """Return the batch's latest result, and keep it as `latest`."""
        obs = {key: np.stack([last.obs[key] for last in self._last]) for key in self._last[0].obs}
        # A metric an env has not reported in this episode reads as zero.
        names = dict.fromkeys(name for last in self._last for name in last.metrics)
        task = tuple(
            str(getattr(env, "task_description", getattr(env, "task", self.task_group))) for env in self._envs
        )
        self._latest = StepResult(
            obs=obs,
            # A vector env has one task description, for all of its envs.
            task=task * self.num_envs if self._batched else task,
            reward=np.array([last.reward for last in self._last], dtype=np.float32),
            terminated=np.array([last.terminated for last in self._last]),
            truncated=np.array([last.truncated for last in self._last]),
            is_success=np.array([last.is_success for last in self._last]),
            elapsed_steps=self._elapsed_steps.copy(),
            metrics={
                name: np.array([last.metrics.get(name, 0.0) for last in self._last], dtype=np.float32)
                for name in names
            },
        )
        return self._latest

    def _close_envs(self) -> None:
        """Close every env, even if one fails to close."""
        envs, self._envs = self._envs, []
        if self.use_async_envs:
            # Ask every process first, so they shut down in parallel instead of one after another.
            for env in envs:
                env.request_close()
        with contextlib.ExitStack() as stack:
            for env in envs:
                stack.callback(env.close)
