# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for `eval_policy_all` task parallelism.

These cover policy-state isolation only, so they need no environment and no GPU.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip("gymnasium", reason="gymnasium is required (install lerobot[evaluation])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from torch import nn  # noqa: E402

from lerobot.scripts import lerobot_eval  # noqa: E402


class _FakePolicy(nn.Module):
    """Minimal stand-in for a chunked policy: per-episode queue, reassigned by `reset`."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)  # so the policy owns parameters to share
        self._action_queue: list[int] = []

    def reset(self):
        self._action_queue = []


class _StubEnv:
    def close(self):
        pass


def _run_eval(policy, n_tasks, max_parallel_tasks):
    envs = {"group": {i: _StubEnv() for i in range(n_tasks)}}
    return lerobot_eval.eval_policy_all(
        envs,
        policy,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        n_episodes=1,
        max_parallel_tasks=max_parallel_tasks,
    )


def _metrics(success):
    return {"sum_rewards": 0.0, "max_rewards": 0.0, "successes": [success]}


def test_parallel_tasks_get_isolated_policy_state(monkeypatch):
    """Each worker must get its own policy view, or tasks clobber each other's action queue."""
    policy = _FakePolicy()
    seen_ids = []
    queue_intact = []

    def fake_run_one(task_group, task_id, env, *, policy, **kwargs):
        seen_ids.append(id(policy))
        policy.reset()
        policy._action_queue.append(task_id)
        # Long enough that a shared policy is reset by a sibling task before we look again.
        time.sleep(0.05)
        queue_intact.append(policy._action_queue == [task_id])
        return task_group, task_id, _metrics(queue_intact[-1])

    monkeypatch.setattr(lerobot_eval, "run_one", fake_run_one)
    _run_eval(policy, n_tasks=4, max_parallel_tasks=4)

    assert len(set(seen_ids)) == 4, "tasks shared a policy instance"
    assert id(policy) not in seen_ids, "the caller's policy was handed to a worker"
    assert all(queue_intact), "a task's action queue was reset by a concurrent task"


def test_policy_view_shares_parameters():
    """Views must reuse the original tensors so parallel eval costs no extra device memory."""
    policy = _FakePolicy()
    view = lerobot_eval._policy_view_for_task(policy)

    assert view is not policy
    assert view.layer.weight is policy.layer.weight
    assert view._action_queue is not policy._action_queue


def test_sequential_path_still_uses_the_original_policy(monkeypatch):
    """`max_parallel_tasks=1` is already safe; it must not pay for copies."""
    policy = _FakePolicy()
    seen = []

    def fake_run_one(task_group, task_id, env, *, policy, **kwargs):
        seen.append(policy)
        return task_group, task_id, _metrics(True)

    monkeypatch.setattr(lerobot_eval, "run_one", fake_run_one)
    _run_eval(policy, n_tasks=2, max_parallel_tasks=1)

    assert seen == [policy, policy]


def test_uncopyable_policy_raises_actionable_error(monkeypatch):
    """A policy that cannot be copied must fail loudly, not silently return garbage."""

    class _Uncopyable(_FakePolicy):
        def __deepcopy__(self, memo):
            raise TypeError("cannot pickle 'tokenizers.Tokenizer' object")

    def fake_run_one(task_group, task_id, env, *, policy, **kwargs):
        return task_group, task_id, _metrics(True)

    monkeypatch.setattr(lerobot_eval, "run_one", fake_run_one)
    with pytest.raises(RuntimeError, match="max_parallel_tasks=1"):
        _run_eval(_Uncopyable(), n_tasks=2, max_parallel_tasks=2)
