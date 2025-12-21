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

import pytest

pytest.importorskip("transformers")

from lerobot.data_processing.sarm_annotations.subtask_annotation import (
    Subtask,
    SubtaskAnnotation,
    Timestamp,
    compute_temporal_proportions,
)


def make_annotation(subtasks: list[tuple[str, int, int]]) -> SubtaskAnnotation:
    """Helper to create SubtaskAnnotation from list of (name, start_sec, end_sec)."""
    return SubtaskAnnotation(
        subtasks=[
            Subtask(
                name=name,
                timestamps=Timestamp(
                    start=f"{start // 60:02d}:{start % 60:02d}", end=f"{end // 60:02d}:{end % 60:02d}"
                ),
            )
            for name, start, end in subtasks
        ]
    )


class TestComputeTemporalProportions:
    """Tests for compute_temporal_proportions (SARM Paper Formula 1).

    Formula: ᾱ_k = (1/M) × Σ_i (L_{i,k} / T_i)

    Key insight: This averages the PROPORTION of each subtask within each trajectory,
    giving equal weight to all trajectories regardless of absolute length.
    """

    def test_basic_two_trajectories_equal_proportions(self):
        """Test with two trajectories that have equal proportions."""
        # Both trajectories: subtask1 = 50%, subtask2 = 50%
        # Traj 1: T=100s, subtask1=50s, subtask2=50s
        # Traj 2: T=200s, subtask1=100s, subtask2=100s
        annotations = {
            0: make_annotation([("subtask1", 0, 50), ("subtask2", 50, 100)]),
            1: make_annotation([("subtask1", 0, 100), ("subtask2", 100, 200)]),
        }

        result = compute_temporal_proportions(annotations)

        # Both should be 0.5
        assert abs(result["subtask1"] - 0.5) < 1e-6
        assert abs(result["subtask2"] - 0.5) < 1e-6

    def test_paper_example_different_from_avg_durations(self):
        """Test that compute_temporal_proportions differs from naive average duration approach.

        This is the key test showing the difference between:
        - Paper formula: average of (L_i,k / T_i)
        - Naive approach: mean(L_i,k) / sum(mean(L_i,j))
        """
        # Episode 1: T=100s, subtask1=80s, subtask2=20s (proportions: 0.8, 0.2)
        # Episode 2: T=200s, subtask1=40s, subtask2=160s (proportions: 0.2, 0.8)
        annotations = {
            0: make_annotation([("subtask1", 0, 80), ("subtask2", 80, 100)]),
            1: make_annotation([("subtask1", 0, 40), ("subtask2", 40, 200)]),
        }

        result = compute_temporal_proportions(annotations)

        # Paper formula:
        # ᾱ_1 = (1/2) × (80/100 + 40/200) = (1/2) × (0.8 + 0.2) = 0.5
        # ᾱ_2 = (1/2) × (20/100 + 160/200) = (1/2) × (0.2 + 0.8) = 0.5
        assert abs(result["subtask1"] - 0.5) < 1e-6
        assert abs(result["subtask2"] - 0.5) < 1e-6

    def test_single_trajectory(self):
        """Test with a single trajectory."""
        # T=100s, reach=30s, grasp=20s, lift=50s
        annotations = {
            0: make_annotation([("reach", 0, 30), ("grasp", 30, 50), ("lift", 50, 100)]),
        }

        result = compute_temporal_proportions(annotations)

        assert abs(result["reach"] - 0.3) < 1e-6
        assert abs(result["grasp"] - 0.2) < 1e-6
        assert abs(result["lift"] - 0.5) < 1e-6

    def test_sum_to_one(self):
        """Test that proportions always sum to 1."""
        # Three episodes with varying proportions
        annotations = {
            0: make_annotation([("a", 0, 10), ("b", 10, 50), ("c", 50, 100)]),  # 0.1, 0.4, 0.5
            1: make_annotation([("a", 0, 20), ("b", 20, 70), ("c", 70, 100)]),  # 0.2, 0.5, 0.3
            2: make_annotation([("a", 0, 30), ("b", 30, 90), ("c", 90, 100)]),  # 0.3, 0.6, 0.1
        }

        result = compute_temporal_proportions(annotations)

        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6

    def test_empty_annotations_returns_empty(self):
        """Test that empty annotations returns empty dict."""
        result = compute_temporal_proportions({})
        assert result == {}

    def test_uniform_proportions(self):
        """Test with uniform proportions across subtasks."""
        # Each subtask takes 25% of each episode
        annotations = {
            0: make_annotation([("a", 0, 25), ("b", 25, 50), ("c", 50, 75), ("d", 75, 100)]),
            1: make_annotation([("a", 0, 50), ("b", 50, 100), ("c", 100, 150), ("d", 150, 200)]),
        }

        result = compute_temporal_proportions(annotations)

        for name in ["a", "b", "c", "d"]:
            assert abs(result[name] - 0.25) < 1e-6
