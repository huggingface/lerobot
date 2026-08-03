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
"""Reproducibility of `LiberoEnv.reset(seed=...)`.

`init_state_id` selects which of the pre-generated initial states a rollout
starts from. It advances on every reset, so unless it is derived from the seed
the same seed yields a different scene depending on how many resets the process
performed earlier.
"""

import numpy as np
import pytest

libero = pytest.importorskip("libero.libero", reason="LIBERO is not installed")

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # noqa: E402
from lerobot.envs.factory import make_env  # noqa: E402

RESOLUTION = 128


@pytest.fixture(scope="module")
def env():
    cfg = LiberoEnvConfig(
        task="libero_spatial",
        task_ids=[0],
        observation_height=RESOLUTION,
        observation_width=RESOLUTION,
        episode_length=10,
    )
    envs = make_env(cfg, n_envs=1)
    e = envs["libero_spatial"][0]
    yield e
    e.close()


def _first_frame(env, seed):
    obs, _ = env.reset(seed=seed)
    return np.asarray(obs["pixels"]["image"])[0].copy()


def test_same_seed_gives_same_initial_state(env):
    """`reset(seed=s)` must be reproducible regardless of intervening resets."""
    first = _first_frame(env, 42)
    _first_frame(env, 7)  # unrelated reset in between
    second = _first_frame(env, 42)

    np.testing.assert_array_equal(
        first,
        second,
        err_msg=(
            "reset(seed=42) produced two different scenes; init_state_id advanced independently of the seed"
        ),
    )


def test_different_seeds_give_different_initial_states(env):
    """Deriving the initial state from the seed must not collapse seeds together."""
    assert not np.array_equal(_first_frame(env, 1), _first_frame(env, 2))


def test_unseeded_reset_still_advances(env):
    """Callers that rely on cycling through initial states are unaffected."""
    env.reset(seed=0)
    assert not np.array_equal(_first_frame(env, None), _first_frame(env, None))
