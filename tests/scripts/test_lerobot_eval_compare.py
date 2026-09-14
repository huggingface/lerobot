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
"""`lerobot-eval-compare`: two eval_info.json files in, per-task verdicts out, exit 1 on a regression."""

import json

import pytest

from lerobot.scripts import lerobot_eval_compare


def _eval_info(tasks: dict[tuple[str, int], list[bool]]) -> dict:
    """Build the `eval_policy_all` output shape from {(task_group, task_id): successes}."""
    per_task = []
    for (group, task_id), successes in tasks.items():
        per_task.append(
            {
                "task_group": group,
                "task_id": task_id,
                "metrics": {
                    "sum_rewards": [1.0] * len(successes),
                    "max_rewards": [1.0] * len(successes),
                    "successes": list(successes),
                    "video_paths": [],
                    "predicted_video_paths": [],
                },
            }
        )
    all_successes = [s for v in tasks.values() for s in v]
    n = len(all_successes)
    return {
        "per_task": per_task,
        "per_group": {},
        "overall": {"pc_success": 100.0 * sum(all_successes) / n, "n_episodes": n},
    }


def _write(tmp_path, name, info):
    p = tmp_path / name
    p.write_text(json.dumps(info))
    return str(p)


def _bits(k, n):
    return [True] * k + [False] * (n - k)


def test_regression_exits_one_and_names_the_task(tmp_path, capsys):
    base = _write(
        tmp_path,
        "base.json",
        _eval_info({("suite", 0): _bits(95, 100), ("suite", 1): _bits(80, 100)}),
    )
    cand = _write(
        tmp_path,
        "cand.json",
        _eval_info({("suite", 0): _bits(60, 100), ("suite", 1): _bits(82, 100)}),
    )
    code = lerobot_eval_compare.main([base, cand])
    out = capsys.readouterr().out
    assert code == 1
    assert "REGRESSED" in out
    assert "suite/0" in out


def test_no_regression_exits_zero(tmp_path, capsys):
    base = _write(tmp_path, "base.json", _eval_info({("suite", 0): _bits(85, 100)}))
    cand = _write(tmp_path, "cand.json", _eval_info({("suite", 0): _bits(83, 100)}))
    assert lerobot_eval_compare.main([base, cand]) == 0
    assert "HELD" in capsys.readouterr().out


def test_json_output_carries_every_task_and_the_summary(tmp_path, capsys):
    base = _write(tmp_path, "base.json", _eval_info({("s", 0): _bits(92, 100), ("s", 1): _bits(37, 100)}))
    cand = _write(tmp_path, "cand.json", _eval_info({("s", 0): _bits(82, 100), ("s", 1): _bits(77, 100)}))
    code = lerobot_eval_compare.main([base, cand, "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    by_task = {t["task"]: t for t in report["tasks"]}
    assert by_task["s/0"]["verdict"] == "SUSPECT"
    assert by_task["s/1"]["verdict"] == "IMPROVED"
    assert report["summary"]["n_regressed"] == 0
    assert report["summary"]["n_suspect"] == 1
    assert report["summary"]["overall"]["baseline_pc_success"] == pytest.approx(64.5)
    assert report["summary"]["overall"]["candidate_pc_success"] == pytest.approx(79.5)
    assert report["summary"]["median_resolvable_drop_pp"] > 5.0


def test_tasks_missing_on_one_side_are_reported_not_dropped_silently(tmp_path, capsys):
    base = _write(tmp_path, "base.json", _eval_info({("s", 0): _bits(9, 10), ("s", 1): _bits(9, 10)}))
    cand = _write(tmp_path, "cand.json", _eval_info({("s", 0): _bits(9, 10)}))
    code = lerobot_eval_compare.main([base, cand, "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["summary"]["only_in_baseline"] == ["s/1"]
    assert report["summary"]["only_in_candidate"] == []


def test_min_drop_changes_the_call(tmp_path, capsys):
    base = _write(tmp_path, "base.json", _eval_info({("s", 0): _bits(95, 100)}))
    cand = _write(tmp_path, "cand.json", _eval_info({("s", 0): _bits(80, 100)}))
    assert lerobot_eval_compare.main([base, cand, "--min-drop", "5"]) == 1
    capsys.readouterr()
    assert lerobot_eval_compare.main([base, cand, "--min-drop", "20"]) == 0


def test_underpowered_tasks_are_marked_and_do_not_fail(tmp_path, capsys):
    base = _write(tmp_path, "base.json", _eval_info({("s", 0): _bits(5, 5)}))
    cand = _write(tmp_path, "cand.json", _eval_info({("s", 0): _bits(1, 5)}))
    assert lerobot_eval_compare.main([base, cand]) == 0
    assert "UNDERPOWERED" in capsys.readouterr().out


def test_unreadable_input_exits_two(tmp_path, capsys):
    base = _write(tmp_path, "base.json", _eval_info({("s", 0): _bits(9, 10)}))
    assert lerobot_eval_compare.main([base, str(tmp_path / "missing.json")]) == 2
    (tmp_path / "bad.json").write_text("{}")
    assert lerobot_eval_compare.main([base, str(tmp_path / "bad.json")]) == 2


def test_single_env_eval_info_is_treated_as_one_task(tmp_path, capsys):
    single = {"per_episode": [{"success": s} for s in _bits(9, 10)], "aggregated": {"pc_success": 90.0}}
    base = _write(tmp_path, "base.json", single)
    cand = _write(tmp_path, "cand.json", single)
    code = lerobot_eval_compare.main([base, cand, "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert [t["task"] for t in report["tasks"]] == ["all"]
