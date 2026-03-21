#!/usr/bin/env python

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

from lerobot.envs.lazy_vec_env import LazyVectorEnv
from lerobot.scripts import lerobot_eval


class _DummyTaskEnv:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _TrackedLazyEnv(LazyVectorEnv):
    def __init__(self, n_factory_fns: int = 1):
        super().__init__(lambda fns: None, [lambda: None for _ in range(n_factory_fns)])
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        super().close()


def _fake_metrics():
    return {
        "sum_rewards": [1.0],
        "max_rewards": [1.0],
        "successes": [True],
        "video_paths": [],
    }


def test_eval_policy_all_sequential_closes_envs(monkeypatch):
    def _fake_run_one(task_group, task_id, env, **kwargs):  # noqa: ARG001
        return task_group, task_id, _fake_metrics()

    monkeypatch.setattr(lerobot_eval, "run_one", _fake_run_one)
    env_a = _DummyTaskEnv()
    env_b = _DummyTaskEnv()
    envs = {"suite": {0: env_a, 1: env_b}}

    result = lerobot_eval.eval_policy_all(
        envs=envs,
        policy=None,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=1,
        max_parallel_tasks=1,
    )

    assert env_a.close_calls == 1
    assert env_b.close_calls == 1
    assert result["overall"]["n_episodes"] == 2


def test_eval_policy_all_threaded_fallback_closes_envs(monkeypatch):
    def _fake_run_one(task_group, task_id, env, **kwargs):  # noqa: ARG001
        return task_group, task_id, _fake_metrics()

    monkeypatch.setattr(lerobot_eval, "run_one", _fake_run_one)
    env_a = _DummyTaskEnv()
    env_b = _DummyTaskEnv()
    env_c = _DummyTaskEnv()
    envs = {"suite": {0: env_a, 1: env_b, 2: env_c}}

    result = lerobot_eval.eval_policy_all(
        envs=envs,
        policy=None,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=1,
        max_parallel_tasks=2,
    )

    assert env_a.close_calls == 1
    assert env_b.close_calls == 1
    assert env_c.close_calls == 1
    assert result["overall"]["n_episodes"] == 3


def test_eval_policy_all_uses_batched_lazy_mode(monkeypatch):
    def _run_one_should_not_be_called(*args, **kwargs):
        raise AssertionError("run_one should not run in batched lazy mode")

    chunk_sizes = []

    def _fake_eval_task_batch(chunk, **kwargs):  # noqa: ARG001
        chunk_sizes.append(len(chunk))
        return [(tg, tid, _fake_metrics()) for tg, tid, _ in chunk]

    monkeypatch.setattr(lerobot_eval, "run_one", _run_one_should_not_be_called)
    monkeypatch.setattr(lerobot_eval, "_eval_task_batch", _fake_eval_task_batch)

    envs = {
        "suite": {
            0: LazyVectorEnv(lambda fns: None, [lambda: None]),
            1: LazyVectorEnv(lambda fns: None, [lambda: None]),
            2: LazyVectorEnv(lambda fns: None, [lambda: None]),
        }
    }

    result = lerobot_eval.eval_policy_all(
        envs=envs,
        policy=None,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=1,
        max_parallel_tasks=2,
    )

    assert chunk_sizes == [2, 1]
    assert result["overall"]["n_episodes"] == 3


def test_eval_policy_all_disables_batched_lazy_when_n_episodes_not_one(monkeypatch):
    def _fake_run_one(task_group, task_id, env, **kwargs):  # noqa: ARG001
        return task_group, task_id, _fake_metrics()

    def _batch_should_not_run(*args, **kwargs):
        raise AssertionError("_eval_task_batch should not run when n_episodes != 1")

    monkeypatch.setattr(lerobot_eval, "run_one", _fake_run_one)
    monkeypatch.setattr(lerobot_eval, "_eval_task_batch", _batch_should_not_run)

    env_a = _TrackedLazyEnv()
    env_b = _TrackedLazyEnv()
    envs = {"suite": {0: env_a, 1: env_b}}

    result = lerobot_eval.eval_policy_all(
        envs=envs,
        policy=None,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=2,
        max_parallel_tasks=2,
    )

    assert env_a.close_calls == 1
    assert env_b.close_calls == 1
    assert result["overall"]["n_episodes"] == 2


def test_eval_policy_all_disables_batched_lazy_when_batch_size_above_one(monkeypatch):
    def _fake_run_one(task_group, task_id, env, **kwargs):  # noqa: ARG001
        return task_group, task_id, _fake_metrics()

    def _batch_should_not_run(*args, **kwargs):
        raise AssertionError("_eval_task_batch should not run when eval.batch_size > 1")

    monkeypatch.setattr(lerobot_eval, "run_one", _fake_run_one)
    monkeypatch.setattr(lerobot_eval, "_eval_task_batch", _batch_should_not_run)

    env_a = _TrackedLazyEnv(n_factory_fns=2)
    env_b = _TrackedLazyEnv(n_factory_fns=2)
    envs = {"suite": {0: env_a, 1: env_b}}

    result = lerobot_eval.eval_policy_all(
        envs=envs,
        policy=None,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=1,
        max_parallel_tasks=2,
    )

    assert env_a.close_calls == 1
    assert env_b.close_calls == 1
    assert result["overall"]["n_episodes"] == 2


def test_eval_policy_all_applies_instance_sharding(monkeypatch):
    called = []

    def _fake_run_one(task_group, task_id, env, **kwargs):  # noqa: ARG001
        called.append(task_id)
        return task_group, task_id, _fake_metrics()

    monkeypatch.setattr(lerobot_eval, "run_one", _fake_run_one)
    envs = {"suite": {0: _DummyTaskEnv(), 1: _DummyTaskEnv(), 2: _DummyTaskEnv(), 3: _DummyTaskEnv()}}

    result = lerobot_eval.eval_policy_all(
        envs=envs,
        policy=None,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=1,
        max_parallel_tasks=1,
        instance_count=2,
        instance_id=1,
    )

    assert called == [1, 3]
    assert result["overall"]["n_episodes"] == 2


def test_aggregate_eval_from_per_task_merges_groups_and_overall():
    per_task = [
        {
            "task_group": "a",
            "task_id": 0,
            "metrics": {"sum_rewards": [1.0], "max_rewards": [2.0], "successes": [True], "video_paths": ["v0"]},
        },
        {
            "task_group": "b",
            "task_id": 1,
            "metrics": {"sum_rewards": [3.0], "max_rewards": [4.0], "successes": [False], "video_paths": []},
        },
    ]

    merged = lerobot_eval._aggregate_eval_from_per_task(per_task, total_eval_s=10.0)

    assert merged["overall"]["n_episodes"] == 2
    assert merged["overall"]["avg_sum_reward"] == 2.0
    assert merged["overall"]["pc_success"] == 50.0
    assert merged["overall"]["eval_s"] == 10.0
    assert set(merged["per_group"]) == {"a", "b"}

