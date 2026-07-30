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
"""`LiberoEnv.step` must not reset on termination.

Gymnasium's vector envs default to `AutoresetMode.NEXT_STEP`, so a sub-env that
resets itself inside `step()` is reset twice per termination -- once by itself and
once by the vector env on the following step. Besides the wasted work, each reset
advances `init_state_id` by `_reset_stride`, so the self-reset silently skipped an
initial state per episode.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import gymnasium as gym
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


def _make_env(monkeypatch: pytest.MonkeyPatch, *, n_envs: int = 1) -> Any:
    """A LiberoEnv whose simulator is a mock that always reports success."""
    inner = Mock(name="OffScreenRenderEnv")
    inner.step.return_value = ({"agentview_image": np.zeros((256, 256, 3), dtype=np.uint8)}, 0.0, False, {})
    inner.check_success.return_value = True
    inner.robots = []

    monkeypatch.setattr(libero_env, "OffScreenRenderEnv", Mock(return_value=inner))
    monkeypatch.setattr(libero_env, "get_libero_path", lambda _key: "/tmp/libero")
    monkeypatch.setattr(
        libero_env, "get_task_init_states", lambda *_a, **_k: np.zeros((8, 4), dtype=np.float64)
    )

    env = libero_env.LiberoEnv(
        task_suite=_FakeTaskSuite(),
        task_id=0,
        task_suite_name="libero_spatial",
        n_envs=n_envs,
        obs_type="pixels",
        camera_name="agentview_image",
        camera_name_mapping={"agentview_image": "image"},
    )
    env._env = inner
    return env


def test_step_does_not_reset_on_termination(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _make_env(monkeypatch)
    reset_calls = Mock(wraps=env.reset)
    monkeypatch.setattr(env, "reset", reset_calls)

    _, _, terminated, _, _ = env.step(np.zeros(7, dtype=np.float32))

    assert terminated is True
    assert reset_calls.call_count == 0, (
        "step() must leave the reset to the caller; the vector env autoresets on the next step"
    )


def test_termination_does_not_skip_an_initial_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each reset advances `init_state_id` by `_reset_stride`; stepping must not."""
    env = _make_env(monkeypatch, n_envs=4)
    before = env.init_state_id

    env.step(np.zeros(7, dtype=np.float32))

    assert env.init_state_id == before


def test_gymnasium_vector_envs_autoreset_on_the_next_step() -> None:
    """Pins the assumption this fix relies on, so a Gymnasium bump can't silently break it."""
    assert gym.vector.AutoresetMode.NEXT_STEP.value == "NextStep"
