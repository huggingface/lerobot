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
"""`FreezeAfterEpisodeEnd` — no simulator work after an episode ends.

`rollout()` latches `done` and keeps stepping the batch until its slowest member
finishes, so a sub-env that terminated early is stepped, physically simulated and
rendered for transitions the rollout then discards.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from lerobot.envs.utils import (
    NEW_ROLLOUT_OPTION,
    FreezeAfterEpisodeEnd,
    freeze_after_episode_end,
)


class _CountingEnv(gym.Env):
    """Terminates after `term_at` steps and counts how often it is actually stepped."""

    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self, term_at: int = 3):
        self.term_at = term_at
        self.n_steps = 0
        self.n_resets = 0
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.n_resets += 1
        self._t = 0
        return np.zeros(2, dtype=np.float32), {"is_success": False}

    def step(self, action):
        self.n_steps += 1
        self._t += 1
        terminated = self._t >= self.term_at
        obs = np.full(2, float(self._t), dtype=np.float32)
        return obs, 1.0, terminated, False, {"is_success": terminated}


def test_no_stepping_after_termination():
    inner = _CountingEnv(term_at=3)
    env = FreezeAfterEpisodeEnd(inner)
    env.reset()

    for _ in range(10):
        env.step(env.action_space.sample())

    assert inner.n_steps == 3, "the wrapped env must not be stepped once its episode ended"
    assert env.is_frozen


def test_frozen_transition_replays_terminal_observation():
    inner = _CountingEnv(term_at=2)
    env = FreezeAfterEpisodeEnd(inner)
    env.reset()

    env.step(env.action_space.sample())
    obs_term, _, terminated, _, info_term = env.step(env.action_space.sample())
    assert terminated

    obs_frozen, reward_frozen, term_frozen, trunc_frozen, info_frozen = env.step(env.action_space.sample())
    np.testing.assert_array_equal(obs_frozen, obs_term)
    assert term_frozen is True
    assert trunc_frozen is False
    assert info_frozen == info_term
    # replayed transitions must not add return if a caller sums rewards over the tail
    assert reward_frozen == 0.0


def test_new_rollout_option_clears_the_freeze():
    inner = _CountingEnv(term_at=2)
    env = FreezeAfterEpisodeEnd(inner)
    env.reset(options={NEW_ROLLOUT_OPTION: True})
    for _ in range(5):
        env.step(env.action_space.sample())
    assert inner.n_steps == 2

    env.reset(options={NEW_ROLLOUT_OPTION: True})
    assert not env.is_frozen
    env.step(env.action_space.sample())
    assert inner.n_steps == 3


def test_autoreset_does_not_thaw_a_finished_env():
    """Gymnasium's autoreset calls reset() with no arguments; that must not rebuild."""
    inner = _CountingEnv(term_at=2)
    env = FreezeAfterEpisodeEnd(inner)
    env.reset(options={NEW_ROLLOUT_OPTION: True})
    env.step(env.action_space.sample())
    obs_term, _, _, _, _ = env.step(env.action_space.sample())
    resets_before = inner.n_resets

    obs, _ = env.reset()  # what the vector env issues on autoreset

    assert env.is_frozen
    assert inner.n_resets == resets_before, "autoreset must not touch the simulator"
    np.testing.assert_array_equal(obs, obs_term)


def test_a_bare_reset_still_works_before_termination():
    """Only a *frozen* env absorbs bare resets; otherwise reset() is normal."""
    inner = _CountingEnv(term_at=5)
    env = FreezeAfterEpisodeEnd(inner)
    env.reset()
    assert inner.n_resets == 1
    env.reset()
    assert inner.n_resets == 2


def test_factory_helper_wraps():
    made = freeze_after_episode_end(lambda: _CountingEnv(term_at=1))()
    assert isinstance(made, FreezeAfterEpisodeEnd)


def test_vector_env_stops_working_on_finished_subenvs():
    """End-to-end shape, under the real AutoresetMode.NEXT_STEP config.

    Gymnasium's AutoresetMode.DISABLED is not usable here: it asserts that a
    terminated sub-env is never stepped, so the wrapper would never be reached.
    """
    term_at = [2, 5, 9]
    envs = [_CountingEnv(term_at=t) for t in term_at]
    vec = gym.vector.SyncVectorEnv([freeze_after_episode_end(lambda e=e: e) for e in envs])
    vec.reset(options={NEW_ROLLOUT_OPTION: True})

    done = np.zeros(len(envs), dtype=bool)
    steps = 0
    while not np.all(done) and steps < 20:  # mirrors lerobot_eval.rollout
        _, _, term, trunc, _ = vec.step(np.zeros((len(envs), 1), dtype=np.float32))
        done = term | trunc | done
        steps += 1

    assert steps == max(term_at), "the batch still runs until its slowest member"
    # ...but each sub-env only did its own episode's work
    assert [e.n_steps for e in envs] == term_at
    # without the wrapper this would be len(term_at) * max(term_at) = 27
    assert sum(e.n_steps for e in envs) == sum(term_at)
    # and no sub-env was rebuilt by an autoreset it did not need
    assert [e.n_resets for e in envs] == [1, 1, 1]
