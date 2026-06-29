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
"""Tests for lerobot_analyze_dataset quality analyzers."""

import json

import numpy as np
import pytest

from lerobot.scripts.lerobot_analyze_dataset import (
    analyze_consistency,
    analyze_coverage,
    analyze_dead_frames,
    analyze_smoothness,
    analyze_temporal,
    compute_composite_score,
)

# ---------------------------------------------------------------------------
# Dead Frame Analyzer
# ---------------------------------------------------------------------------


class TestDeadFrameAnalyzer:
    def test_dead_frame_detection(self):
        """Actions with known dead segments should be detected."""
        actions = np.zeros((100, 2))
        # Frames 0-49: dead (no change)
        # Frames 50-99: moving (linspace starts at 0, so delta at index 49 is also 0)
        actions[50:, 0] = np.linspace(0, 10, 50)
        actions[50:, 1] = np.linspace(0, 5, 50)

        result = analyze_dead_frames(actions, threshold=1e-4)

        # Deltas 0..48 are zero (49 frames within the dead block).
        # Delta 49 (frame 49->50) is also zero since linspace starts at 0.
        # Deltas 50..98 are non-zero (49 moving deltas).
        assert result["dead_frame_count"] == 50
        assert result["dead_frame_ratio"] == pytest.approx(50 / 99, abs=0.01)
        assert result["longest_dead_streak"] == 50
        assert result["score"] < 0.6

    def test_dead_frame_all_moving(self):
        """When all actions change significantly, there should be 0 dead frames."""
        t = np.linspace(0, 10, 100)
        actions = np.stack([np.sin(t), np.cos(t)], axis=1)

        result = analyze_dead_frames(actions, threshold=1e-6)

        assert result["dead_frame_count"] == 0
        assert result["dead_frame_ratio"] == 0.0
        assert result["longest_dead_streak"] == 0
        assert result["score"] == 1.0

    def test_dead_frame_short_episode(self):
        """Single-frame episode should return clean result."""
        actions = np.array([[1.0, 2.0]])
        result = analyze_dead_frames(actions, threshold=1e-4)
        assert result["dead_frame_count"] == 0
        assert result["score"] == 1.0


# ---------------------------------------------------------------------------
# Smoothness Analyzer
# ---------------------------------------------------------------------------


class TestSmoothnessAnalyzer:
    def test_constant_velocity(self):
        """Linear trajectory has uniform step sizes -> CV near 0 -> high score."""
        t = np.linspace(0, 1, 100).reshape(-1, 1)
        actions = np.hstack([t * 5.0, t * 3.0])  # Linear in both dims

        result = analyze_smoothness(actions, fps=50)

        assert result["delta_cv"] < 0.01
        assert result["smoothness_score"] > 0.99

    def test_jerky_motion(self):
        """Random actions should have high delta CV and lower smoothness."""
        rng = np.random.RandomState(42)
        actions = rng.randn(100, 2)

        result = analyze_smoothness(actions, fps=50)

        assert result["delta_cv"] > 0.3
        assert result["smoothness_score"] < 0.9

    def test_short_episode(self):
        """Two-frame episode should still work (one delta)."""
        actions = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = analyze_smoothness(actions, fps=50)
        # Single delta -> std=0 -> cv=0 -> score=1
        assert result["smoothness_score"] == pytest.approx(1.0, abs=0.01)

    def test_single_frame(self):
        """Single frame -> no deltas -> defaults."""
        actions = np.array([[1.0, 2.0]])
        result = analyze_smoothness(actions, fps=50)
        assert result["smoothness_score"] == 1.0
        assert result["delta_cv"] == 0.0


# ---------------------------------------------------------------------------
# Consistency Analyzer
# ---------------------------------------------------------------------------


class TestConsistencyAnalyzer:
    def test_identical_episodes(self):
        """All identical episodes => no outliers, high consistency."""
        base = np.linspace(0, 1, 50).reshape(-1, 1) * np.ones((1, 2))
        episodes = [base.copy() for _ in range(10)]

        result = analyze_consistency(episodes, outlier_std=2.0)

        assert result["outlier_episodes"] == []
        assert result["consistency_score"] == pytest.approx(1.0, abs=0.01)
        assert result["mean_pairwise_distance"] < 1e-6

    def test_one_outlier(self):
        """One episode very different from the rest should be flagged."""
        base = np.linspace(0, 1, 50).reshape(-1, 1) * np.ones((1, 2))
        episodes = [base.copy() for _ in range(10)]
        # Make episode 5 completely different
        episodes[5] = np.random.RandomState(42).randn(50, 2) * 100

        result = analyze_consistency(episodes, outlier_std=2.0)

        assert 5 in result["outlier_episodes"]
        assert result["consistency_score"] < 1.0

    def test_single_episode(self):
        """Single episode: consistency is trivially perfect."""
        result = analyze_consistency([np.ones((10, 2))], outlier_std=2.0)
        assert result["consistency_score"] == 1.0
        assert result["outlier_episodes"] == []

    def test_different_lengths_resampled(self):
        """Episodes of different lengths should be handled via resampling."""
        episodes = [
            np.linspace(0, 1, 30).reshape(-1, 1) * np.ones((1, 2)),
            np.linspace(0, 1, 50).reshape(-1, 1) * np.ones((1, 2)),
            np.linspace(0, 1, 70).reshape(-1, 1) * np.ones((1, 2)),
        ]
        result = analyze_consistency(episodes, outlier_std=2.0)
        # All represent the same linear trajectory, should not be outliers
        assert result["outlier_episodes"] == []


# ---------------------------------------------------------------------------
# Coverage Analyzer
# ---------------------------------------------------------------------------


class TestCoverageAnalyzer:
    def test_coverage_basic(self):
        """Known state distribution should have correct ranges and std."""
        rng = np.random.RandomState(42)
        # Uniform-ish distribution in 3 dims
        states = [rng.uniform(-1, 1, (100, 3)) for _ in range(5)]

        result = analyze_coverage(states)

        assert len(result["state_dim_ranges"]) == 3
        assert len(result["per_dim_std"]) == 3
        assert len(result["per_dim_utilization"]) == 3
        assert 0.0 <= result["coverage_score"] <= 1.0

        # Ranges should be close to [-1, 1] with 500 samples
        for lo, hi in result["state_dim_ranges"]:
            assert lo < -0.8
            assert hi > 0.8

    def test_coverage_narrow(self):
        """Very narrow distribution should have lower coverage."""
        states = [np.ones((100, 2)) + np.random.RandomState(42).randn(100, 2) * 0.001]
        result = analyze_coverage(states)
        # Score should still be valid
        assert 0.0 <= result["coverage_score"] <= 1.0


# ---------------------------------------------------------------------------
# Temporal Analyzer
# ---------------------------------------------------------------------------


class TestTemporalAnalyzer:
    def test_uniform_lengths(self):
        """All episodes same length => no short/long flags."""
        lengths = [100] * 20

        result = analyze_temporal(lengths, fps=50, min_episode_length_ratio=0.3)

        assert result["short_episodes"] == []
        assert result["long_episodes"] == []
        assert result["mean_length"] == 100
        assert result["median_length"] == 100
        assert result["temporal_score"] > 0.9

    def test_one_short_episode(self):
        """One much shorter episode should be flagged."""
        lengths = [100] * 19 + [5]  # Episode 19 is very short

        result = analyze_temporal(lengths, fps=50, min_episode_length_ratio=0.3)

        assert 19 in result["short_episodes"]
        assert result["min_length"] == 5

    def test_one_long_episode(self):
        """One much longer episode should be flagged."""
        lengths = [100] * 19 + [1000]  # Episode 19 is very long

        result = analyze_temporal(lengths, fps=50, min_episode_length_ratio=0.3)

        assert 19 in result["long_episodes"]
        assert result["max_length"] == 1000

    def test_durations_computed(self):
        """Duration in seconds should be length / fps."""
        result = analyze_temporal([50, 100], fps=50, min_episode_length_ratio=0.3)
        assert result["durations_s"] == pytest.approx([1.0, 2.0])


# ---------------------------------------------------------------------------
# Composite Score
# ---------------------------------------------------------------------------


class TestCompositeScore:
    def test_weights(self):
        """Provide known per-dimension scores, verify weighted average."""
        scores = {
            "dead_frames": 1.0,
            "smoothness": 1.0,
            "consistency": 1.0,
            "coverage": 1.0,
            "temporal": 1.0,
        }
        assert compute_composite_score(scores) == pytest.approx(1.0)

    def test_partial_dimensions(self):
        """Score should work with only some dimensions present."""
        scores = {"dead_frames": 0.8, "temporal": 0.6}
        # Weights: dead_frames=0.25, temporal=0.20
        expected = (0.25 * 0.8 + 0.20 * 0.6) / (0.25 + 0.20)
        assert compute_composite_score(scores) == pytest.approx(expected, abs=0.001)

    def test_zero_scores(self):
        """All zeros should give zero."""
        scores = {"dead_frames": 0.0, "smoothness": 0.0, "consistency": 0.0}
        assert compute_composite_score(scores) == pytest.approx(0.0)

    def test_empty_scores(self):
        """No dimensions => 0."""
        assert compute_composite_score({}) == 0.0


# ---------------------------------------------------------------------------
# JSON Output (Integration-like with synthetic data)
# ---------------------------------------------------------------------------


class TestJSONOutput:
    def test_json_structure(self, tmp_path):
        """Run analyzers on synthetic data and verify JSON report structure."""
        from lerobot.scripts.lerobot_analyze_dataset import build_json_report

        report = build_json_report(
            repo_id="test/synthetic",
            total_episodes=3,
            total_frames=300,
            fps=50,
            overall_score=0.75,
            dimension_scores={"dead_frames": 0.9, "smoothness": 0.7},
            episode_results=[
                {"episode_index": 0, "length": 100, "composite_score": 0.8, "flags": []},
                {"episode_index": 1, "length": 100, "composite_score": 0.7, "flags": ["short"]},
                {"episode_index": 2, "length": 100, "composite_score": 0.75, "flags": []},
            ],
            recommendations=["Consider removing episodes: [1]"],
        )

        # Verify top-level structure
        assert report["repo_id"] == "test/synthetic"
        assert report["total_episodes"] == 3
        assert report["total_frames"] == 300
        assert report["fps"] == 50
        assert report["overall_score"] == 0.75
        assert "dead_frames" in report["dimension_scores"]
        assert len(report["episodes"]) == 3
        assert len(report["recommendations"]) == 1

        # Verify JSON serializable
        json_path = tmp_path / "report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["repo_id"] == "test/synthetic"
