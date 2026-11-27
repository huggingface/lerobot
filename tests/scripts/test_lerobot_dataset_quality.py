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

"""Tests for lerobot.scripts.lerobot_dataset_quality."""

import json
import math

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_quality import (
    DEFAULT_MOTIONLESS_FRACTION,
    EpisodeMetrics,
    _format_table,
    _json_safe,
    analyze_actions,
    build_delete_command,
    compute_episode_metrics,
    flag_outliers,
    load_episode_actions,
    rank_flagged,
)
from tests.fixtures.constants import DEFAULT_FPS, DUMMY_REPO_ID

pytestmark = pytest.mark.dataset


def _ramp(num_frames: int, dim: int = 2, end: float = 1.0) -> np.ndarray:
    """Linear trajectory from the origin to ``end`` in every dimension."""
    return np.linspace(0.0, end, num=num_frames, dtype=np.float64)[:, None].repeat(dim, axis=1)


def _make_metrics(**overrides) -> EpisodeMetrics:
    defaults = {
        "num_frames": 10,
        "duration_s": 10 / DEFAULT_FPS,
        "median_jerk": 0.0,
        "p95_jerk": 0.0,
        "peak_velocity": 0.1,
        "motionless_fraction": 0.0,
        "final_pose_distance": 0.1,
    }
    return EpisodeMetrics(**{**defaults, **overrides})


# ── Per-episode metrics ──────────────────────────────────────────────


class TestComputeEpisodeMetrics:
    def test_constant_trajectory(self):
        actions = np.zeros((10, 2))
        metrics = compute_episode_metrics(actions, DEFAULT_FPS, DEFAULT_MOTIONLESS_FRACTION, np.zeros(2))

        assert metrics.num_frames == 10
        assert metrics.duration_s == pytest.approx(10 / DEFAULT_FPS)
        assert metrics.peak_velocity == 0.0
        assert metrics.motionless_fraction == 1.0
        assert metrics.median_jerk == pytest.approx(0.0, abs=1e-12)
        assert metrics.p95_jerk == pytest.approx(0.0, abs=1e-12)

    def test_smooth_ramp_has_zero_jerk(self):
        actions = _ramp(10)
        metrics = compute_episode_metrics(actions, DEFAULT_FPS, DEFAULT_MOTIONLESS_FRACTION, np.zeros(2))

        # A linear ramp has constant velocity, hence zero jerk.
        assert metrics.median_jerk == pytest.approx(0.0, abs=1e-12)
        assert metrics.p95_jerk == pytest.approx(0.0, abs=1e-12)
        assert metrics.peak_velocity == pytest.approx(np.linalg.norm(np.array([1.0, 1.0]) / 9))

    def test_short_episode_jerk_is_nan(self):
        actions = _ramp(3)
        metrics = compute_episode_metrics(actions, DEFAULT_FPS, DEFAULT_MOTIONLESS_FRACTION, np.zeros(2))

        assert math.isnan(metrics.median_jerk)
        assert math.isnan(metrics.p95_jerk)
        # Velocity is still defined for 3 frames (2 transitions).
        assert metrics.peak_velocity == pytest.approx(np.linalg.norm(np.array([1.0, 1.0]) / 2))

    def test_motionless_fraction_counts_hesitations(self):
        # Fast motion with a single full stop in the middle.
        actions = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [5.0, 5.0], [6.0, 6.0], [10.0, 10.0]])
        metrics = compute_episode_metrics(
            actions, DEFAULT_FPS, motionless_threshold=0.5, mean_final_pose=np.zeros(2)
        )

        # 1 of the 5 transitions is motionless.
        assert metrics.motionless_fraction == pytest.approx(1 / 5)

    def test_final_pose_distance(self):
        actions = _ramp(10)
        metrics = compute_episode_metrics(
            actions, DEFAULT_FPS, DEFAULT_MOTIONLESS_FRACTION, np.array([0.3, 0.4])
        )

        assert metrics.final_pose_distance == pytest.approx(np.linalg.norm(np.array([0.7, 0.6])))


class TestAnalyzeActions:
    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="No episodes"):
            analyze_actions({}, DEFAULT_FPS)

    def test_single_episode_final_pose_distance_is_zero(self):
        metrics = analyze_actions({0: _ramp(10)}, DEFAULT_FPS)

        assert metrics[0].final_pose_distance == 0.0

    def test_all_static_episodes_are_motionless(self):
        actions = {ep: np.full((10, 2), float(ep)) for ep in range(3)}
        metrics = analyze_actions(actions, DEFAULT_FPS)

        for ep in range(3):
            assert metrics[ep].motionless_fraction == 1.0
            assert metrics[ep].peak_velocity == 0.0

    def test_outlier_episode_receives_high_peak_velocity(self):
        actions = {ep: _ramp(10) for ep in range(5)}
        actions[3][5] += np.array([10.0, 10.0])  # single sharp jerk
        metrics = analyze_actions(actions, DEFAULT_FPS)

        assert metrics[3].peak_velocity > 5 * metrics[0].peak_velocity


# ── IQR flagging ─────────────────────────────────────────────────────


class TestFlagOutliers:
    def test_no_flags_on_symmetric_values(self):
        values = {ep: float(ep) for ep in range(10)}

        assert flag_outliers(values, k=1.5) == set()

    def test_flags_extreme_value(self):
        values = {ep: float(ep) for ep in range(10)}
        values[9] = 100.0

        assert flag_outliers(values, k=1.5) == {9}

    def test_larger_k_flags_fewer(self):
        values = {ep: float(ep) for ep in range(10)}
        values[9] = 30.0

        assert flag_outliers(values, k=1.5) == {9}
        assert flag_outliers(values, k=10.0) == set()

    def test_zero_iqr_flags_nothing(self):
        values = dict.fromkeys(range(10), 1.0)

        assert flag_outliers(values, k=1.5) == set()

    @pytest.mark.parametrize("outlier", [-100.0, 100.0])
    def test_zero_iqr_with_distinct_tail_still_flags_outlier(self, outlier):
        # Q1 == Q3 does not imply all values are equal: a single distinct episode
        # among identical ones must still be flagged.
        values = dict.fromkeys(range(8), 1.0)
        values[8] = outlier
        values[9] = math.nan

        assert flag_outliers(values, k=1.5) == {8}

    def test_nan_values_are_ignored(self):
        values = {0: 0.0, 1: 1.0, 2: math.nan, 3: 2.0, 4: 100.0}

        assert flag_outliers(values, k=1.5) == {4}

    def test_fewer_than_two_finite_values_flags_nothing(self):
        assert flag_outliers({0: 1.0, 1: math.nan}, k=1.5) == set()


# ── Ranking and output helpers ────────────────────────────────────────


class TestRankFlagged:
    def test_orders_by_flag_count_then_episode(self):
        metrics = {ep: _make_metrics() for ep in range(4)}
        flagged = {"duration_s": {0, 2}, "peak_velocity": {2, 1}}

        ranked = rank_flagged(metrics, flagged)

        assert ranked == [
            (2, ["duration_s", "peak_velocity"]),
            (0, ["duration_s"]),
            (1, ["peak_velocity"]),
        ]

    def test_empty_when_nothing_flagged(self):
        assert rank_flagged({0: _make_metrics()}, {"duration_s": set()}) == []


class TestBuildDeleteCommand:
    def test_command_without_root(self):
        command = build_delete_command("lerobot/pusht", [0, 5, 2], None)

        assert command == (
            "lerobot-edit-dataset --repo_id lerobot/pusht "
            '--operation.type delete_episodes --operation.episode_indices "[0, 5, 2]"'
        )

    def test_command_with_root(self):
        command = build_delete_command("lerobot/pusht", [3], "/data/pusht")

        assert command.endswith('--operation.episode_indices "[3]" --root /data/pusht')


class TestFormatTable:
    def test_table_lists_flagged_episodes(self):
        metrics = {ep: _make_metrics() for ep in range(4)}
        ranked = [(2, ["duration_s", "peak_velocity"]), (0, ["duration_s"])]

        table = _format_table("lerobot/pusht", DEFAULT_FPS, 1.5, metrics, ranked)

        assert "4 total, 2 flagged" in table
        assert "flagged: duration_s, peak_velocity" in table

    def test_table_reports_no_outliers(self):
        table = _format_table("lerobot/pusht", DEFAULT_FPS, 1.5, {0: _make_metrics()}, [])

        assert "No outlier episodes detected." in table


# ── End-to-end through the LeRobotDataset surface ─────────────────────


@pytest.fixture(scope="module")
def recorded_dataset(tmp_path_factory):
    """A small recorded dataset: three smooth ramps and one spiky episode."""
    features = {"action": {"dtype": "float32", "shape": (2,), "names": None}}
    root = tmp_path_factory.mktemp("dataset_quality") / "test_dataset"
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, root=root, features=features, use_videos=False
    )
    for _ in range(3):
        for value in np.linspace(0.0, 1.0, 10):
            dataset.add_frame({"action": np.array([value, value], dtype=np.float32), "task": "test"})
        dataset.save_episode()
    # Episode 3: same duration and same final pose, but with a violent correction spike.
    for i, value in enumerate(np.linspace(0.0, 1.0, 10)):
        action = np.array([value, value], dtype=np.float32)
        if i == 5:
            action = np.array([30.0, -30.0], dtype=np.float32)
        dataset.add_frame({"action": action, "task": "test"})
    dataset.save_episode()
    return DUMMY_REPO_ID, root


class TestEndToEnd:
    def test_load_episode_actions(self, recorded_dataset):
        repo_id, root = recorded_dataset
        dataset = LeRobotDataset(repo_id, root=root)

        actions = load_episode_actions(dataset)

        assert set(actions) == {0, 1, 2, 3}
        for ep in range(4):
            assert actions[ep].shape == (10, 2)

    def test_outlier_episode_is_flagged(self, recorded_dataset):
        repo_id, root = recorded_dataset
        dataset = LeRobotDataset(repo_id, root=root)
        metrics = analyze_actions(load_episode_actions(dataset), dataset.meta.fps)

        peak_velocity_flags = flag_outliers({ep: m.peak_velocity for ep, m in metrics.items()}, k=1.5)

        assert 3 in peak_velocity_flags
        assert {0, 1, 2} & peak_velocity_flags == set()

    def test_json_safe_produces_strict_json(self):
        payload = {"median_jerk": _json_safe(math.nan), "peak_velocity": _json_safe(0.5)}

        assert json.loads(json.dumps(payload)) == {"median_jerk": None, "peak_velocity": 0.5}
