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

"""Tests for `lerobot-watch-eval`, the read-only progress reporter for eval runs."""

import json

import pytest

from lerobot.scripts.lerobot_watch_eval import (
    MANIFEST_FILENAME,
    RESULTS_FILENAME,
    _fmt_dur,
    main,
    summarize,
)


def _write_run(run_dir, records, manifest=None):
    """Write one run directory. A `str` record is written verbatim, to simulate a torn line."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    (run_dir / RESULTS_FILENAME).write_text("".join(line + "\n" for line in lines))
    if manifest is not None:
        (run_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    return run_dir


def _record(group, task_id, successes, **kwargs):
    return {
        "task_group": group,
        "task_id": task_id,
        "status": "ok",
        "metrics": {"successes": successes},
        "n_episodes": len(successes),
        **kwargs,
    }


def test_summarize_reports_progress_and_per_suite_success(tmp_path):
    run_dir = _write_run(
        tmp_path / "run",
        [
            _record("libero_10", 0, [True, True], wall_s=10.0),
            _record("libero_10", 1, [True, False], wall_s=10.0),
            _record("libero_90", 0, [False, False], wall_s=10.0),
        ],
        manifest={"n_tasks": 4, "policy_path": "some/policy"},
    )

    out = summarize(run_dir)

    assert "policy:   some/policy" in out
    assert "progress: 3/4 tasks (75.0%)   ok=3  err=0" in out
    assert "overall:  pc_success 50.0%  (6 episodes)" in out
    assert "libero_10" in out and "75.0%" in out
    assert "libero_90" in out and "0.0%" in out
    # One task left at ~10s each.
    assert "~10s left for 1 tasks" in out


def test_summarize_lists_errors_and_excludes_them_from_success_rate(tmp_path):
    run_dir = _write_run(
        tmp_path / "run",
        [
            _record("libero_10", 0, [True, True]),
            {
                "task_group": "libero_10",
                "task_id": 1,
                "status": "error",
                "metrics": None,
                "error": "RuntimeError('boom')",
            },
        ],
        manifest={"n_tasks": 2},
    )

    out = summarize(run_dir)

    assert "ok=1  err=1" in out
    assert "overall:  pc_success 100.0%  (2 episodes)" in out
    assert "libero_10/1: RuntimeError('boom')" in out


def test_summarize_without_manifest_reports_unknown_total(tmp_path):
    run_dir = _write_run(tmp_path / "run", [_record("libero_10", 0, [True])])

    out = summarize(run_dir)

    assert "no manifest; total unknown" in out
    assert "pace:" not in out


def test_summarize_uses_latest_line_per_task(tmp_path):
    """A task retried on resume must count once, with its final outcome."""
    run_dir = _write_run(
        tmp_path / "run",
        [
            {"task_group": "libero_10", "task_id": 0, "status": "error", "metrics": None},
            _record("libero_10", 0, [True, True]),
            '{"task_group": "libero_10", "task_i',  # torn line is ignored
        ],
        manifest={"n_tasks": 1},
    )

    out = summarize(run_dir)

    assert "progress: 1/1 tasks (100.0%)   ok=1  err=0" in out


@pytest.mark.parametrize(
    ("seconds", "expected"), [(45, "45s"), (90, "1m30s"), (3600, "1h00m"), (7860, "2h11m")]
)
def test_fmt_dur(seconds, expected):
    assert _fmt_dur(seconds) == expected


def test_main_walks_nested_run_directories(tmp_path, monkeypatch, capsys):
    _write_run(tmp_path / "2026-08-20" / "run_a", [_record("libero_10", 0, [True])])
    _write_run(tmp_path / "2026-08-20" / "run_b", [_record("libero_90", 0, [False])])

    monkeypatch.setattr("sys.argv", ["lerobot-watch-eval", str(tmp_path)])
    main()

    out = capsys.readouterr().out
    assert "Found 2 run(s)" in out
    assert "run_a" in out and "run_b" in out


def test_main_errors_when_no_runs_found(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["lerobot-watch-eval", str(tmp_path)])
    with pytest.raises(SystemExit, match="No runs"):
        main()
