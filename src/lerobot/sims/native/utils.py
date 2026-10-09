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

from __future__ import annotations

from collections.abc import Callable, Sequence

import gymnasium as gym


def parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize ``camera_name`` into a non-empty list of strings.

    Accepts a comma-separated string (``"cam_a,cam_b"``) or a sequence of
    strings (tuples/lists). Whitespace is stripped; empty entries are
    dropped. Raises ``TypeError`` for unsupported input types and
    ``ValueError`` when the normalized list is empty.
    """
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list | tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


# Passed in `reset(options=...)` by `rollout()` to mark the start of a new rollout.
# FreezeAfterEpisodeEnd thaws only on this, so Gymnasium's argument-less autoreset
# cannot be mistaken for a genuine new episode.
NEW_ROLLOUT_OPTION = "lerobot_new_rollout"


class FreezeAfterEpisodeEnd(gym.Wrapper):
    """Stop doing simulator work once a sub-env's episode has ended.

    `rollout()` runs `while not np.all(done)` with `done` latched, so a sub-env that
    terminates early keeps being stepped -- physics and offscreen rendering included --
    until the slowest sub-env in the batch finishes. The batch runs for
    `max(episode_lengths)` iterations to complete work that only needs
    `mean(episode_lengths)`.

    This caches the terminal transition and replays it for any further `step()` or
    autoreset, so a finished sub-env costs nothing. The rollout already ignores those
    transitions.

    The freeze survives Gymnasium's autoreset deliberately. Under
    `AutoresetMode.NEXT_STEP` the vector env resets a terminated sub-env on the
    following step and runs it through an entire extra episode that the rollout
    discards, because `done` stays latched. Absorbing that reset is most of the saving.

    Only an explicit reset carrying `NEW_ROLLOUT_OPTION` thaws it, so the signal is
    explicit rather than inferred: Gymnasium's autoreset calls `reset()` with no
    arguments, but so would a caller passing `seeds=None`, and confusing the two would
    strand an env frozen for a whole rollout.

    `AutoresetMode.DISABLED` is not an alternative here — Gymnasium asserts that no
    terminated env is ever stepped in that mode, so the wrapper is never reached.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._frozen: tuple | None = None

    def reset(self, *, seed=None, options=None):
        if self._frozen is not None and not (options or {}).get(NEW_ROLLOUT_OPTION):
            # Gymnasium's autoreset for a sub-env the rollout has already finished with.
            # Replay the terminal observation instead of rebuilding the simulation.
            obs, _, _, _, info = self._frozen
            return obs, info
        self._frozen = None
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        if self._frozen is not None:
            return self._frozen
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            # Zero the reward on replay so a frozen sub-env cannot inflate a return if a
            # caller sums rewards over the padded tail.
            self._frozen = (obs, 0.0, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    @property
    def is_frozen(self) -> bool:
        return self._frozen is not None


def freeze_after_episode_end(env_fn: Callable[[], gym.Env]) -> Callable[[], gym.Env]:
    """Wrap an env factory so the built env freezes once its episode ends."""

    def _fn() -> gym.Env:
        return FreezeAfterEpisodeEnd(env_fn())

    return _fn


class _LazyAsyncVectorEnv:
    """Defers AsyncVectorEnv creation until first use.

    Creating all tasks' AsyncVectorEnvs upfront spawns N_tasks × n_envs worker
    processes, all of which allocate EGL/GPU resources immediately. Since tasks
    are evaluated sequentially, only one task's workers need to be alive at a
    time. This wrapper stores the factory functions and creates the real
    AsyncVectorEnv on first reset()/step()/call(), keeping peak process count = n_envs.
    """

    def __init__(
        self,
        env_fns: list[Callable],
        observation_space=None,
        action_space=None,
        metadata=None,
    ):
        self._env_fns = env_fns
        self._env: gym.vector.AsyncVectorEnv | None = None
        self.num_envs = len(env_fns)
        if observation_space is not None and action_space is not None and metadata is not None:
            self.observation_space = observation_space
            self.action_space = action_space
            self.metadata = metadata
        else:
            tmp = env_fns[0]()
            self.observation_space = tmp.observation_space
            self.action_space = tmp.action_space
            self.metadata = tmp.metadata
            tmp.close()
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    def _ensure(self) -> None:
        if self._env is None:
            self._env = gym.vector.AsyncVectorEnv(
                [freeze_after_episode_end(fn) for fn in self._env_fns],
                context="forkserver",
                shared_memory=True,
                autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
            )

    @property
    def unwrapped(self):
        return self

    def reset(self, **kwargs):
        self._ensure()
        return self._env.reset(**kwargs)

    def step(self, actions):
        self._ensure()
        return self._env.step(actions)

    def call(self, name, *args, **kwargs):
        self._ensure()
        return self._env.call(name, *args, **kwargs)

    def get_attr(self, name):
        self._ensure()
        return self._env.get_attr(name)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
