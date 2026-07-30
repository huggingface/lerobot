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
"""Reset-path behaviour of `LiberoEnv`.

`reset()` is immediately followed by `set_init_state()` whenever init states are
in use, so the scene rebuild that LIBERO's default `hard_reset=True` performs is
discarded. These tests pin the resulting wiring: soft reset when init states
place the objects, hard reset when `reset()` is the only thing that does.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.envs import libero as libero_env


class _FakeTask:
    name = "fake_task"
    language = "do the thing"
    problem_folder = "fake_folder"
    bddl_file = "fake.bddl"


class _FakeTaskSuite:
    def get_task(self, task_id: int) -> _FakeTask:  # noqa: ARG002
        return _FakeTask()


def _build_env(monkeypatch: pytest.MonkeyPatch, *, init_states: bool) -> tuple[Any, Mock]:
    """Instantiate a `LiberoEnv` with the simulator stubbed out.

    Returns the env plus the `OffScreenRenderEnv` mock, so the constructor kwargs
    can be inspected without allocating a MuJoCo model or a GL context.
    """
    offscreen = Mock(name="OffScreenRenderEnv")
    monkeypatch.setattr(libero_env, "OffScreenRenderEnv", offscreen)
    monkeypatch.setattr(libero_env, "get_libero_path", lambda _key: "/tmp/libero")
    monkeypatch.setattr(
        libero_env, "get_task_init_states", lambda *_a, **_k: np.zeros((2, 8), dtype=np.float64)
    )

    env = libero_env.LiberoEnv(
        task_suite=_FakeTaskSuite(),
        task_id=0,
        task_suite_name="libero_spatial",
        init_states=init_states,
    )
    env._ensure_env()
    return env, offscreen


def test_init_states_use_a_soft_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """With init states, `set_init_state()` overwrites the state `reset()` produced.

    The model rebuild, GL-context recreation and observable re-wiring that
    `hard_reset=True` performs are therefore pure overhead (measured 1.0-1.5 s per
    episode across the four suites).

    This test pins the wiring only. End-to-end equivalence is NOT established: the
    two paths agree bit-for-bit with `num_steps_wait=0` but diverge with the default
    of 10, and that is unresolved.
    """
    _, offscreen = _build_env(monkeypatch, init_states=True)

    assert offscreen.call_count == 1
    assert offscreen.call_args.kwargs["hard_reset"] is False


def test_without_init_states_the_hard_reset_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without init states, `reset()` is the only thing placing the objects.

    Model-level randomisation happens in `_load_model()`, which a soft reset skips,
    so the hard reset has to stay.
    """
    _, offscreen = _build_env(monkeypatch, init_states=False)

    assert offscreen.call_count == 1
    assert offscreen.call_args.kwargs["hard_reset"] is True
