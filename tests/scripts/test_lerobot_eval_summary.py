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
"""`eval_policy_all` reports success counts and Wilson intervals at every aggregation level."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts import lerobot_eval  # noqa: E402

_SUCCESSES = {
    ("suite_a", 0): [True] * 9 + [False],
    ("suite_a", 1): [True] * 5 + [False] * 5,
    ("suite_b", 0): [False] * 10,
}


class _DummyEnv:
    def close(self):
        pass


def _fake_run_one(task_group, task_id, env, **kwargs):
    successes = _SUCCESSES[(task_group, task_id)]
    metrics = {
        "sum_rewards": [1.0] * len(successes),
        "max_rewards": [1.0] * len(successes),
        "successes": list(successes),
        "video_paths": [],
        "predicted_video_paths": [],
    }
    return task_group, task_id, metrics


@pytest.fixture
def info(monkeypatch):
    monkeypatch.setattr(lerobot_eval, "run_one", _fake_run_one)
    envs = {"suite_a": {0: _DummyEnv(), 1: _DummyEnv()}, "suite_b": {0: _DummyEnv()}}
    policy = MagicMock()
    policy.training = False
    return lerobot_eval.eval_policy_all(envs, policy, None, None, None, None, n_episodes=10)


def test_per_task_entries_carry_counts_and_interval(info):
    task = next(t for t in info["per_task"] if (t["task_group"], t["task_id"]) == ("suite_a", 0))
    assert task["n_episodes"] == 10
    assert task["n_success"] == 9
    assert task["pc_success"] == pytest.approx(90.0)
    assert task["pc_success_ci95"] == pytest.approx([59.58, 98.21], abs=0.05)


def test_per_group_aggregate_pools_its_tasks(info):
    group = info["per_group"]["suite_a"]
    assert group["n_episodes"] == 20
    assert group["n_success"] == 14
    assert group["pc_success"] == pytest.approx(70.0)
    assert group["pc_success_ci95"] == pytest.approx([48.10, 85.45], abs=0.05)


def test_overall_aggregate_pools_every_task(info):
    overall = info["overall"]
    assert overall["n_episodes"] == 30
    assert overall["n_success"] == 14
    assert overall["pc_success"] == pytest.approx(100 * 14 / 30)
    assert overall["pc_success_ci95"] == pytest.approx([30.23, 63.86], abs=0.05)


def test_existing_keys_are_untouched(info):
    overall = info["overall"]
    for key in ("avg_sum_reward", "avg_max_reward", "eval_s", "eval_ep_s", "video_paths"):
        assert key in overall
    assert set(info) == {"per_task", "per_group", "overall"}
