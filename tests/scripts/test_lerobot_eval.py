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

"""Regression tests for `eval_policy_all`'s parallel-task dispatch (issue #4327).

`eval_policy_all(max_parallel_tasks > 1)` used to hand every `ThreadPoolExecutor` worker the *same*
`policy` object. Every supported policy class keeps its rollout state (action queues populated by
`select_action`, reset by `reset()`) directly on the instance, so two tasks running concurrently would
clobber each other's queue mid-rollout with no error — every task silently returned near-0% success.
"""

import threading
import time
from typing import Any

import pytest
import torch
from torch import nn

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts import lerobot_eval as eval_module  # noqa: E402


class _FakeEnv:
    """Stand-in for a `gym.vector.VectorEnv`: only `.close()` is touched by `eval_policy_all`."""

    def __init__(self, token: str):
        self.token = token
        self.closed = False

    def close(self):
        self.closed = True


class _QueuePolicy(nn.Module):
    """Minimal reduction of the real bug: rollout state lives directly on the policy instance.

    `reset()` reassigns `self._queue` to a fresh object (as every real policy's `reset()` reassigns
    `self._queues`), and `select_action()` stamps and re-reads it after yielding the GIL. If two threads
    share one instance, one thread's `reset()`/`select_action()` can land between the other thread's
    stamp and re-read, so the re-read observes the wrong token.
    """

    def __init__(self):
        super().__init__()
        self._queue: dict[str, str | None] = {}

    def reset(self):
        self._queue = {"token": None}

    def select_action(self, batch: dict[str, str]) -> torch.Tensor:
        token = batch["token"]
        self._queue["token"] = token
        time.sleep(0.005)  # yield the GIL so a concurrent reset()/select_action() can interleave
        seen = self._queue["token"]
        if seen != token:
            raise AssertionError(f"cross-task interference: expected token {token!r}, saw {seen!r}")
        return torch.zeros(1)


def _fake_eval_one(env: _FakeEnv, *, policy: _QueuePolicy, n_episodes: int, **_ignored: Any) -> dict:
    for _ in range(n_episodes):
        policy.reset()
        policy.select_action({"token": env.token})
    return {
        "sum_rewards": [0.0] * n_episodes,
        "max_rewards": [0.0] * n_episodes,
        "successes": [True] * n_episodes,
    }


def test_eval_policy_all_parallel_tasks_do_not_share_policy_state(monkeypatch):
    """Each worker thread must get its own policy copy, not the shared instance (see #4327)."""
    monkeypatch.setattr(eval_module, "eval_one", _fake_eval_one)

    policy = _QueuePolicy()
    envs = {"group": {i: _FakeEnv(token=f"task-{i}") for i in range(8)}}

    result = eval_module.eval_policy_all(
        envs,
        policy,
        env_preprocessor=lambda x: x,
        env_postprocessor=lambda x: x,
        preprocessor=lambda x: x,
        postprocessor=lambda x: x,
        n_episodes=20,
        max_parallel_tasks=4,
    )

    assert result["overall"]["n_episodes"] == 8 * 20
    assert result["overall"]["pc_success"] == 100.0
    assert all(env.closed for group in envs.values() for env in group.values())


def test_eval_policy_all_threaded_reuses_one_copy_per_worker_thread(monkeypatch):
    """Memory should scale with `max_parallel_tasks`, not with the number of tasks."""
    monkeypatch.setattr(eval_module, "eval_one", _fake_eval_one)

    seen_instance_ids: set[int] = set()
    lock = threading.Lock()

    def _tracking_eval_one(env: _FakeEnv, *, policy: _QueuePolicy, n_episodes: int, **_ignored: Any) -> dict:
        with lock:
            seen_instance_ids.add(id(policy))
        return _fake_eval_one(env, policy=policy, n_episodes=n_episodes)

    monkeypatch.setattr(eval_module, "eval_one", _tracking_eval_one)

    policy = _QueuePolicy()
    envs = {"group": {i: _FakeEnv(token=f"task-{i}") for i in range(6)}}

    eval_module.eval_policy_all(
        envs,
        policy,
        env_preprocessor=lambda x: x,
        env_postprocessor=lambda x: x,
        preprocessor=lambda x: x,
        postprocessor=lambda x: x,
        n_episodes=1,
        max_parallel_tasks=2,
    )

    # 6 tasks funnel through at most 2 worker threads, so at most 2 distinct deep copies are made,
    # never one per task and never the original shared instance.
    assert 1 <= len(seen_instance_ids) <= 2
    assert id(policy) not in seen_instance_ids


def test_eval_policy_all_sequential_reuses_shared_policy(monkeypatch):
    """`max_parallel_tasks<=1` should keep using the original policy directly (no copying overhead)."""
    seen_instance_ids: set[int] = set()

    def _tracking_eval_one(env: _FakeEnv, *, policy: _QueuePolicy, n_episodes: int, **_ignored: Any) -> dict:
        seen_instance_ids.add(id(policy))
        return _fake_eval_one(env, policy=policy, n_episodes=n_episodes)

    monkeypatch.setattr(eval_module, "eval_one", _tracking_eval_one)

    policy = _QueuePolicy()
    envs = {"group": {i: _FakeEnv(token=f"task-{i}") for i in range(3)}}

    eval_module.eval_policy_all(
        envs,
        policy,
        env_preprocessor=lambda x: x,
        env_postprocessor=lambda x: x,
        preprocessor=lambda x: x,
        postprocessor=lambda x: x,
        n_episodes=1,
        max_parallel_tasks=1,
    )

    assert seen_instance_ids == {id(policy)}
