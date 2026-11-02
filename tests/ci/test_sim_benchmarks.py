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

"""Simulator benchmark CI: the matrix, the metrics it records, and the LIBERO policy-server example."""

import json
import subprocess
import sys
from pathlib import Path

import draccus
import pytest
import yaml

from lerobot.remote_inference.configs import ServerConfig
from tests.utils import resolve_without_simulator

ROOT = Path(__file__).resolve().parents[2]


def _parse_metrics(artifacts: Path, *options: str) -> subprocess.CompletedProcess:
    """Run parse_eval_metrics.py on `artifacts` for a toy benchmark."""
    command = [
        sys.executable,
        str(ROOT / "scripts/ci/parse_eval_metrics.py"),
        "--artifacts-dir",
        str(artifacts),
    ]
    return subprocess.run(
        [*command, "--env", "toy", "--task", "reach", "--policy", "test", *options], capture_output=True
    )


def test_missing_metrics_fail(tmp_path):
    assert _parse_metrics(tmp_path).returncode == 1


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
    assert _parse_metrics(tmp_path).returncode == expected


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
    assert [e["name"] for e in entries if e["parity"]] == ["libero", "metaworld"]
    assert all(len(e["sim_tag"]) == 24 for e in entries)
    assert all(len(e["revision"]) == 40 for e in entries)
    assert next(e for e in entries if e["name"] == "libero_plus")["task"] == "libero_spatial"


def test_libero_policy_server_example_matches_simulator_config(monkeypatch):
    policy = draccus.parse(ServerConfig, config_path=str(ROOT / "docker/sim/policy_libero.yaml"), args=[])
    settings = yaml.safe_load((ROOT / "docker/sim/configs/libero.yaml").read_text())["sim"]
    sim = resolve_without_simulator(settings, monkeypatch)
    assert policy.semantics == sim.semantics
    assert policy.robot_type == sim.robot_type
    assert tuple(policy.action_feature.names) == sim.action_names
    state = next(f for f in policy.features if f.name == "observation.state")
    assert tuple(state.names) == sim.state_names


@pytest.mark.parametrize("max_episode_steps", [None, 30])
def test_metrics_record_what_the_score_depends_on(tmp_path, max_episode_steps):
    (tmp_path / "eval_info.json").write_text(json.dumps({"overall": {"pc_success": 50, "n_episodes": 2}}))
    pins = ["--policy-revision", "abc123", "--sim-tag", "f00d", "--commit", "deadbeef"]
    capped = [] if max_episode_steps is None else ["--max-episode-steps", str(max_episode_steps)]
    assert _parse_metrics(tmp_path, *pins, *capped).returncode == 0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert (metrics["policy_revision"], metrics["sim_tag"], metrics["commit"]) == (
        "abc123",
        "f00d",
        "deadbeef",
    )
    # A capped evaluation is a pipeline check, never a policy score.
    assert metrics["smoke"] is (max_episode_steps is not None)
    assert metrics["max_episode_steps"] == max_episode_steps
