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

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_missing_metrics_fail(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/ci/parse_eval_metrics.py"),
            "--artifacts-dir",
            str(tmp_path),
            "--env",
            "toy",
            "--task",
            "reach",
            "--policy",
            "test",
        ],
        capture_output=True,
    )
    assert result.returncode == 1


@pytest.mark.parametrize(
    "metrics,expected",
    [
        ({"pc_success": 0, "n_episodes": 1}, 0),
        ({"pc_success": float("inf"), "n_episodes": 1}, 1),
        ({"pc_success": 101, "n_episodes": 1}, 1),
        ({"pc_success": 0, "n_episodes": 0}, 1),
        ({"pc_success": 0, "n_episodes": 1.5}, 1),
    ],
)
def test_metrics_validity(tmp_path, metrics, expected):
    (tmp_path / "eval_info.json").write_text(json.dumps({"overall": metrics}))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/ci/parse_eval_metrics.py"),
            "--artifacts-dir",
            str(tmp_path),
            "--env",
            "toy",
            "--task",
            "reach",
            "--policy",
            "test",
        ],
        capture_output=True,
    )
    assert result.returncode == expected


def test_matrix_has_eight_benchmarks_and_training_coverage():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/ci/benchmark_matrix.py"), "--root", str(ROOT)],
        capture_output=True,
        text=True,
        check=True,
    )
    entries = json.loads(result.stdout)["include"]
    assert len(entries) == 8
    assert [e["name"] for e in entries if e["train_smoke"]] == ["libero"]
    assert all(len(e["sim_tag"]) == 24 for e in entries)
    assert all(len(e["revision"]) == 40 for e in entries)
    assert next(e for e in entries if e["name"] == "libero_plus")["task"] == "libero_spatial"
