# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fair initial conditions in the paired simulator evaluation utility."""

import runpy
from pathlib import Path

import numpy as np
import pytest

helpers = runpy.run_path(str(Path(__file__).resolve().parents[3] / "examples/longwam/evaluate_libero.py"))
fingerprint = helpers["observation_digest"]
reset_fresh_environment = helpers["reset_fresh_environment"]


def test_fingerprint_is_order_independent_but_detects_sensor_changes():
    first = {"state": np.array([1.0, 2.0]), "pixels": {"camera": np.zeros((2, 2, 3), dtype=np.uint8)}}
    reordered = {"pixels": first["pixels"], "state": first["state"].copy()}
    assert fingerprint(first) == fingerprint(reordered)
    reordered["state"][0] += 1e-9
    assert fingerprint(first) != fingerprint(reordered)
    with pytest.raises(ValueError, match="object arrays"):
        fingerprint(np.array([object()], dtype=object))


def test_episode_reset_constructs_new_instances_after_seeding():
    class Environment:
        def __init__(self):
            self.fixture = np.random.rand(4)
            self.closed = False

        def reset(self, seed):
            return {"state": self.fixture.copy(), "episode": self.init_state_id}, {}

        def close(self):
            self.closed = True

    first, obs1 = reset_fresh_environment(Environment, 1, 3)
    first.fixture += 10  # A previous rollout must not affect the paired environment.
    second, obs2 = reset_fresh_environment(Environment, 1, 3)
    assert first is not second
    assert fingerprint(obs1) == fingerprint(obs2)
    first.close()
    second.close()


def test_failed_reset_closes_its_environment():
    class Environment:
        closed = False

        def reset(self, seed):
            raise RuntimeError("reset failed")

        def close(self):
            self.closed = True

    env = Environment()
    with pytest.raises(RuntimeError, match="reset failed"):
        reset_fresh_environment(lambda: env, 1, 0)
    assert env.closed
