#!/usr/bin/env python

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
"""The initial-state sequence must not depend on when a slot terminates.

`LiberoEnv.reset()` selects `_init_states[init_state_id % len(...)]` and then advances
`init_state_id` by `_reset_stride`, so *any* reset it services consumes an index --
including the argument-less one Gymnasium issues under `AutoresetMode.NEXT_STEP` for a
sub-env that finished on the previous step. A slot that terminates early therefore walks
further along the initial-state list than one that does not, and which states an
evaluation visits becomes a function of policy behaviour rather than of configuration.

`FreezeAfterEpisodeEnd` absorbs that autoreset, so the only reset reaching the env is
`rollout()`'s explicit one carrying `NEW_ROLLOUT_OPTION`: exactly one index per slot per
rollout, whatever the policy did. These tests pin that accounting on the vector env
`create_libero_envs` builds for the sync path, without needing LIBERO's assets.

See https://github.com/huggingface/lerobot/issues/4152.
"""

from __future__ import annotations

from functools import partial

import gymnasium as gym
import numpy as np

from lerobot.envs.utils import NEW_ROLLOUT_OPTION, freeze_after_episode_end

STRIDE = 2
N_WAVES = 3
STEPS_PER_WAVE = 6
TERM_AT = 2


class _InitStateEnv(gym.Env):
    """Mirrors `LiberoEnv`'s initial-state bookkeeping and records every index used.

    `term_at=None` reproduces the slot that never terminates: `LiberoEnv.step()` returns
    `truncated=False` unconditionally and the LIBERO factories build the env without a
    `TimeLimit`, so the step budget is imposed only by `rollout()`'s loop.
    """

    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self, episode_index: int, stride: int, term_at: int | None):
        self.init_state_id = episode_index
        self._stride = stride
        self._term_at = term_at
        self._t = 0
        self.states_used: list[int] = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.states_used.append(self.init_state_id)
        self.init_state_id += self._stride
        self._t = 0
        return np.zeros(2, dtype=np.float32), {"is_success": False}

    def step(self, action):
        self._t += 1
        terminated = self._term_at is not None and self._t >= self._term_at
        obs = np.full(2, float(self._t), dtype=np.float32)
        return obs, 0.0, terminated, False, {"is_success": terminated}


def _build(term_ats: list[int | None], *, frozen: bool) -> gym.vector.SyncVectorEnv:
    """Build the vector env `create_libero_envs` builds, with and without the wrapper."""
    fns = [partial(_InitStateEnv, i, STRIDE, t) for i, t in enumerate(term_ats)]
    if frozen:
        fns = [freeze_after_episode_end(fn) for fn in fns]
    return gym.vector.SyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)


def _run_waves(envs: gym.vector.SyncVectorEnv) -> list[list[int]]:
    """Drive `envs` the way `rollout()` does and report the index each wave started on.

    One explicit `NEW_ROLLOUT_OPTION` reset per wave, then steps until every slot has
    finished at least once -- `done` is latched, so a slot that terminates early keeps
    being driven until the slowest one catches up. Only the index consumed by the
    explicit reset is reported: a correct fix is free to let incidental resets *inside*
    a rollout differ, so asserting on the full reset trace would reject one.
    """
    inner = [e.unwrapped for e in envs.envs]
    zero_action = np.zeros((envs.num_envs, 1), dtype=np.float32)
    per_wave: list[list[int]] = []

    for _ in range(N_WAVES):
        marks = [len(e.states_used) for e in inner]
        envs.reset(options={NEW_ROLLOUT_OPTION: True})
        per_wave.append([e.states_used[m] for e, m in zip(inner, marks, strict=True)])

        done = np.zeros(envs.num_envs, dtype=bool)
        for _ in range(STEPS_PER_WAVE):
            _, _, terminated, truncated, _ = envs.step(zero_action)
            done |= terminated | truncated
            if done.all():
                break

    return per_wave


def test_initial_state_sequence_is_independent_of_termination():
    """One rollout consumes exactly one index per slot, whenever the slot finished."""
    # Slot 0 terminates early every wave; slot 1 never terminates at all.
    envs = _build([TERM_AT, None], frozen=True)
    try:
        per_wave = _run_waves(envs)
    finally:
        envs.close()

    expected = [[slot + wave * STRIDE for slot in range(2)] for wave in range(N_WAVES)]
    assert per_wave == expected, (
        f"initial-state sequence moved with termination timing: {per_wave} != {expected}"
    )


def test_sequence_is_identical_whatever_the_slots_do():
    """The paired form: same config, different termination patterns, same sequence."""
    patterns = ([TERM_AT, None], [None, TERM_AT], [None, None], [TERM_AT, TERM_AT])
    sequences = []
    for term_ats in patterns:
        envs = _build(term_ats, frozen=True)
        try:
            sequences.append(_run_waves(envs))
        finally:
            envs.close()

    assert len({repr(s) for s in sequences}) == 1, (
        f"initial-state sequence differed across termination patterns: "
        f"{dict(zip([str(p) for p in patterns], sequences, strict=True))}"
    )


def test_the_guard_can_actually_fail():
    """Without the wrapper the sequence drifts -- the defect this file pins.

    Keeps the tests above honest: they would also pass against an env that never
    advanced `init_state_id` at all.
    """
    envs = _build([TERM_AT, None], frozen=False)
    try:
        per_wave = _run_waves(envs)
    finally:
        envs.close()

    terminating = [wave[0] for wave in per_wave]
    steady = [wave[1] for wave in per_wave]

    assert terminating != [0, 2, 4], (
        "expected the unwrapped sync env to drift; the autoreset may no longer reach reset()"
    )
    assert steady == [1, 3, 5], f"the slot that never terminates should be unaffected, got {steady}"
