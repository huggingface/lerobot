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

"""Tests for the crash-safe progress logging and resume support of `lerobot-eval`."""

import json

import pytest

from lerobot.configs.default import EvalConfig
from lerobot.scripts import lerobot_eval
from lerobot.scripts.lerobot_eval import (
    MANIFEST_FILENAME,
    RESULTS_FILENAME,
    _load_result_records,
    eval_policy_all,
)


class _FakePolicy:
    """Minimal stand-in for a policy: `eval_policy_all` only toggles its train/eval mode."""

    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode


class _FakeEnv:
    def __init__(self):
        self.n_closed = 0

    def close(self):
        self.n_closed += 1


def _metrics(successes):
    return {
        "sum_rewards": [float(s) for s in successes],
        "max_rewards": [float(s) for s in successes],
        "successes": list(successes),
        "video_paths": [],
        "predicted_video_paths": [],
    }


def _make_envs(keys):
    """Build the nested {task_group: {task_id: env}} structure `eval_policy_all` expects."""
    envs = {}
    for group, task_id in keys:
        envs.setdefault(group, {})[task_id] = _FakeEnv()
    return envs


def _run(envs, tmp_path, monkeypatch, *, fail_keys=(), ran=None, **kwargs):
    """Call `eval_policy_all` with `run_one` stubbed out, recording which tasks actually ran."""

    def fake_run_one(task_group, task_id, env, **_):
        if ran is not None:
            ran.append((task_group, task_id))
        if (task_group, task_id) in fail_keys:
            raise RuntimeError(f"boom {task_group}/{task_id}")
        return task_group, task_id, _metrics([True, False])

    monkeypatch.setattr(lerobot_eval, "run_one", fake_run_one)

    return eval_policy_all(
        envs,
        _FakePolicy(),
        None,
        None,
        None,
        None,
        2,
        start_seed=1000,
        output_dir=tmp_path,
        **kwargs,
    )


def test_load_result_records_last_line_wins_and_skips_malformed(tmp_path):
    path = tmp_path / RESULTS_FILENAME
    path.write_text(
        json.dumps({"task_group": "libero_10", "task_id": 0, "status": "error"})
        + "\n"
        + json.dumps({"task_group": "libero_10", "task_id": 0, "status": "ok"})
        + "\n"
        + json.dumps({"task_group": "libero_10", "task_id": 1, "status": "ok"})
        + "\n"
        + '{"task_group": "libero_10", "task_i'  # torn final line, as after a hard crash
    )

    records = _load_result_records(path)

    assert set(records) == {("libero_10", 0), ("libero_10", 1)}
    assert records[("libero_10", 0)]["status"] == "ok"


def test_load_result_records_returns_empty_when_missing(tmp_path):
    assert _load_result_records(tmp_path / RESULTS_FILENAME) == {}


@pytest.mark.parametrize("max_parallel_tasks", [1, 2])
def test_writes_manifest_and_one_result_line_per_task(tmp_path, monkeypatch, max_parallel_tasks):
    keys = [("libero_10", 0), ("libero_10", 1), ("libero_90", 0)]
    ran = []

    _run(
        _make_envs(keys),
        tmp_path,
        monkeypatch,
        ran=ran,
        max_parallel_tasks=max_parallel_tasks,
        run_meta={"policy_path": "some/policy"},
    )

    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    assert manifest["n_tasks"] == 3
    assert manifest["n_episodes"] == 2
    assert manifest["start_seed"] == 1000
    assert manifest["policy_path"] == "some/policy"
    assert {(t["task_group"], t["task_id"]) for t in manifest["tasks"]} == set(keys)

    records = _load_result_records(tmp_path / RESULTS_FILENAME)
    assert set(records) == set(keys)
    assert all(r["status"] == "ok" for r in records.values())
    assert all(r["n_episodes"] == 2 and r["pc_success"] == 50.0 for r in records.values())
    assert all(r["wall_s"] is not None for r in records.values())
    assert sorted(ran) == sorted(keys)


@pytest.mark.parametrize("max_parallel_tasks", [1, 2])
def test_failing_task_is_recorded_without_aborting_the_run(tmp_path, monkeypatch, max_parallel_tasks):
    keys = [("libero_10", 0), ("libero_10", 1)]

    info = _run(
        _make_envs(keys),
        tmp_path,
        monkeypatch,
        fail_keys={("libero_10", 0)},
        max_parallel_tasks=max_parallel_tasks,
    )

    records = _load_result_records(tmp_path / RESULTS_FILENAME)
    assert records[("libero_10", 0)]["status"] == "error"
    assert "boom" in records[("libero_10", 0)]["error"]
    assert records[("libero_10", 1)]["status"] == "ok"
    # The surviving task still contributes to the aggregates.
    assert info["overall"]["n_episodes"] == 2


def test_resume_skips_completed_tasks_and_reloads_their_metrics(tmp_path, monkeypatch):
    keys = [("libero_10", 0), ("libero_10", 1)]
    _run(_make_envs(keys), tmp_path, monkeypatch, fail_keys={("libero_10", 1)})

    ran = []
    info = _run(_make_envs(keys), tmp_path, monkeypatch, ran=ran, resume=True)

    # Only the previously-failed task is re-run; the successful one is replayed from disk.
    assert ran == [("libero_10", 1)]
    assert info["overall"]["n_episodes"] == 4  # 2 replayed + 2 freshly run
    assert info["overall"]["pc_success"] == 50.0
    assert _load_result_records(tmp_path / RESULTS_FILENAME)[("libero_10", 1)]["status"] == "ok"


def test_resume_closes_envs_of_skipped_tasks(tmp_path, monkeypatch):
    keys = [("libero_10", 0), ("libero_10", 1)]
    _run(_make_envs(keys), tmp_path, monkeypatch)

    envs = _make_envs(keys)
    _run(envs, tmp_path, monkeypatch, resume=True)

    assert all(env.n_closed == 1 for group in envs.values() for env in group.values())


def test_resume_with_retry_failed_false_keeps_the_failure(tmp_path, monkeypatch):
    keys = [("libero_10", 0), ("libero_10", 1)]
    _run(_make_envs(keys), tmp_path, monkeypatch, fail_keys={("libero_10", 1)})

    ran = []
    info = _run(_make_envs(keys), tmp_path, monkeypatch, ran=ran, resume=True, retry_failed=False)

    assert ran == []
    assert info["overall"]["n_episodes"] == 2  # only the task that succeeded first time round
    assert _load_result_records(tmp_path / RESULTS_FILENAME)[("libero_10", 1)]["status"] == "error"


def test_resume_keeps_original_manifest_and_warns_on_mismatch(tmp_path, monkeypatch, caplog):
    _run(_make_envs([("libero_10", 0), ("libero_10", 1)]), tmp_path, monkeypatch)
    created_ts = json.loads((tmp_path / MANIFEST_FILENAME).read_text())["created_ts"]

    with caplog.at_level("WARNING"):
        _run(_make_envs([("libero_10", 0)]), tmp_path, monkeypatch, resume=True)

    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    assert manifest["n_tasks"] == 2  # original total preserved, so progress stays meaningful
    assert manifest["created_ts"] == created_ts
    assert "n_tasks: 2 -> 1" in caplog.text


def test_warns_when_appending_into_an_existing_run_without_resume(tmp_path, monkeypatch, caplog):
    _run(_make_envs([("libero_10", 0)]), tmp_path, monkeypatch)

    with caplog.at_level("WARNING"):
        _run(_make_envs([("libero_10", 0)]), tmp_path, monkeypatch)

    assert "--eval.resume=true" in caplog.text


def test_no_persistence_without_output_dir(tmp_path, monkeypatch):
    _run(_make_envs([("libero_10", 0)]), None, monkeypatch)

    assert not (tmp_path / RESULTS_FILENAME).exists()
    assert not (tmp_path / MANIFEST_FILENAME).exists()


def test_eval_config_rejects_resume_with_recording():
    with pytest.raises(ValueError, match="eval.resume is not supported"):
        EvalConfig(n_episodes=2, resume=True, recording=True)

    EvalConfig(n_episodes=2, resume=True)  # resume alone is fine
