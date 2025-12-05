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

"""
Tests for SARM utility functions.

Tests the implementation of SARM paper formulas:
- Formula (1): compute_temporal_proportions - dataset-level temporal proportions
- Formula (2): compute_tau, compute_cumulative_progress - progress labels
- Rewind augmentation (A.4)
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch

from lerobot.policies.sarm.sarm_utils import SubtaskAnnotation, Subtask, Timestamp
from lerobot.policies.sarm.sarm_utils import (
    compute_temporal_proportions,
    compute_tau,
    compute_cumulative_progress_batch,
)

def make_annotation(subtasks: list[tuple[str, int, int]]) -> SubtaskAnnotation:
    """Helper to create SubtaskAnnotation from list of (name, start_sec, end_sec)."""
    return SubtaskAnnotation(
        subtasks=[
            Subtask(
                name=name,
                timestamps=Timestamp(
                    start=f"{start // 60:02d}:{start % 60:02d}",
                    end=f"{end // 60:02d}:{end % 60:02d}"
                )
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
            0: make_annotation([('subtask1', 0, 50), ('subtask2', 50, 100)]),
            1: make_annotation([('subtask1', 0, 100), ('subtask2', 100, 200)]),
        }
        
        result = compute_temporal_proportions(annotations)
        
        # Both should be 0.5
        assert abs(result['subtask1'] - 0.5) < 1e-6
        assert abs(result['subtask2'] - 0.5) < 1e-6
    
    def test_paper_example_different_from_avg_durations(self):
        """Test that compute_temporal_proportions differs from naive average duration approach.
        
        This is the key test showing the difference between:
        - Paper formula: average of (L_i,k / T_i)
        - Naive approach: mean(L_i,k) / sum(mean(L_i,j))
        """
        # Episode 1: T=100s, subtask1=80s, subtask2=20s (proportions: 0.8, 0.2)
        # Episode 2: T=200s, subtask1=40s, subtask2=160s (proportions: 0.2, 0.8)
        annotations = {
            0: make_annotation([('subtask1', 0, 80), ('subtask2', 80, 100)]),
            1: make_annotation([('subtask1', 0, 40), ('subtask2', 40, 200)]),
        }
        
        result = compute_temporal_proportions(annotations)
        
        # Paper formula: 
        # ᾱ_1 = (1/2) × (80/100 + 40/200) = (1/2) × (0.8 + 0.2) = 0.5
        # ᾱ_2 = (1/2) × (20/100 + 160/200) = (1/2) × (0.2 + 0.8) = 0.5
        assert abs(result['subtask1'] - 0.5) < 1e-6
        assert abs(result['subtask2'] - 0.5) < 1e-6
    
    def test_single_trajectory(self):
        """Test with a single trajectory."""
        # T=100s, reach=30s, grasp=20s, lift=50s
        annotations = {
            0: make_annotation([('reach', 0, 30), ('grasp', 30, 50), ('lift', 50, 100)]),
        }
        
        result = compute_temporal_proportions(annotations)
        
        assert abs(result['reach'] - 0.3) < 1e-6
        assert abs(result['grasp'] - 0.2) < 1e-6
        assert abs(result['lift'] - 0.5) < 1e-6
    
    def test_sum_to_one(self):
        """Test that proportions always sum to 1."""
        # Three episodes with varying proportions
        annotations = {
            0: make_annotation([('a', 0, 10), ('b', 10, 50), ('c', 50, 100)]),  # 0.1, 0.4, 0.5
            1: make_annotation([('a', 0, 20), ('b', 20, 70), ('c', 70, 100)]),  # 0.2, 0.5, 0.3
            2: make_annotation([('a', 0, 30), ('b', 30, 90), ('c', 90, 100)]),  # 0.3, 0.6, 0.1
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
            0: make_annotation([('a', 0, 25), ('b', 25, 50), ('c', 50, 75), ('d', 75, 100)]),
            1: make_annotation([('a', 0, 50), ('b', 50, 100), ('c', 100, 150), ('d', 150, 200)]),
        }
        
        result = compute_temporal_proportions(annotations)
        
        for name in ['a', 'b', 'c', 'd']:
            assert abs(result[name] - 0.25) < 1e-6


class TestComputeTau:
    """Tests for compute_tau (within-subtask progress).
    
    Formula: τ_t = (t - s_k) / (e_k - s_k) ∈ [0, 1]
    """
    
    def test_at_start(self):
        """τ should be 0 at subtask start."""
        tau = compute_tau(current_frame=10, subtask_start=10, subtask_end=50)
        assert tau == 0.0
    
    def test_at_end(self):
        """τ should be 1 at subtask end."""
        tau = compute_tau(current_frame=50, subtask_start=10, subtask_end=50)
        assert tau == 1.0
    
    def test_at_middle(self):
        """τ should be 0.5 at subtask midpoint."""
        tau = compute_tau(current_frame=30, subtask_start=10, subtask_end=50)
        assert abs(tau - 0.5) < 1e-6
    
    def test_quarter_progress(self):
        """Test τ at 25% through subtask."""
        tau = compute_tau(current_frame=20, subtask_start=0, subtask_end=80)
        assert abs(tau - 0.25) < 1e-6
    
    def test_zero_duration_subtask(self):
        """τ should be 1.0 for zero-duration subtask."""
        tau = compute_tau(current_frame=10, subtask_start=10, subtask_end=10)
        assert tau == 1.0
    
    def test_clamps_below_zero(self):
        """τ should be clamped to 0 if frame is before subtask."""
        tau = compute_tau(current_frame=5, subtask_start=10, subtask_end=50)
        assert tau == 0.0
    
    def test_clamps_above_one(self):
        """τ should be clamped to 1 if frame is after subtask."""
        tau = compute_tau(current_frame=60, subtask_start=10, subtask_end=50)
        assert tau == 1.0
    
    def test_float_inputs(self):
        """Test with float frame indices (from interpolation)."""
        tau = compute_tau(current_frame=25.5, subtask_start=10.0, subtask_end=50.0)
        expected = (25.5 - 10.0) / (50.0 - 10.0)
        assert abs(tau - expected) < 1e-6


class TestComputeCumulativeProgressBatchScalar:
    """Tests for compute_cumulative_progress_batch with scalar inputs (normalized progress y_t).
    
    Formula: y_t = P_{k-1} + ᾱ_k × τ_t ∈ [0, 1]
    """
    
    def test_first_subtask_start(self):
        """y should be 0 at start of first subtask."""
        proportions = [0.3, 0.5, 0.2]
        y = compute_cumulative_progress_batch(tau=0.0, stage_indices=0, alpha=proportions)
        assert y == 0.0
    
    def test_first_subtask_end(self):
        """y should equal ᾱ_1 at end of first subtask."""
        proportions = [0.3, 0.5, 0.2]
        y = compute_cumulative_progress_batch(tau=1.0, stage_indices=0, alpha=proportions)
        assert abs(y - 0.3) < 1e-6
    
    def test_second_subtask_start(self):
        """y should equal P_1 at start of second subtask."""
        proportions = [0.3, 0.5, 0.2]
        y = compute_cumulative_progress_batch(tau=0.0, stage_indices=1, alpha=proportions)
        assert abs(y - 0.3) < 1e-6
    
    def test_second_subtask_end(self):
        """y should equal P_2 at end of second subtask."""
        proportions = [0.3, 0.5, 0.2]
        y = compute_cumulative_progress_batch(tau=1.0, stage_indices=1, alpha=proportions)
        assert abs(y - 0.8) < 1e-6  # 0.3 + 0.5
    
    def test_third_subtask_end(self):
        """y should be 1.0 at end of last subtask."""
        proportions = [0.3, 0.5, 0.2]
        y = compute_cumulative_progress_batch(tau=1.0, stage_indices=2, alpha=proportions)
        assert abs(y - 1.0) < 1e-6
    
    def test_midpoint_of_subtask(self):
        """Test progress at midpoint of a subtask."""
        proportions = [0.4, 0.6]
        # At τ=0.5 in subtask 1: y = P_0 + ᾱ_1 × 0.5 = 0 + 0.4 × 0.5 = 0.2
        y = compute_cumulative_progress_batch(tau=0.5, stage_indices=0, alpha=proportions)
        assert abs(y - 0.2) < 1e-6
        
        # At τ=0.5 in subtask 2: y = P_1 + ᾱ_2 × 0.5 = 0.4 + 0.6 × 0.5 = 0.7
        y = compute_cumulative_progress_batch(tau=0.5, stage_indices=1, alpha=proportions)
        assert abs(y - 0.7) < 1e-6
    
    def test_uniform_proportions(self):
        """Test with uniform proportions."""
        proportions = [0.25, 0.25, 0.25, 0.25]
        
        # At end of each subtask, progress should be 0.25, 0.5, 0.75, 1.0
        for i in range(4):
            y = compute_cumulative_progress_batch(tau=1.0, stage_indices=i, alpha=proportions)
            expected = (i + 1) * 0.25
            assert abs(y - expected) < 1e-6


class TestComputeCumulativeProgressBatchTensor:
    """Tests for compute_cumulative_progress_batch with tensor inputs (GPU batch version)."""
    
    def test_tensor_matches_scalar_version(self):
        """Test that tensor version matches scalar version."""
        proportions = [0.3, 0.5, 0.2]
        alpha = torch.tensor(proportions, dtype=torch.float32)
        cumulative = torch.zeros(len(proportions) + 1, dtype=torch.float32)
        cumulative[1:] = torch.cumsum(alpha, dim=0)
        
        test_cases = [
            (0.0, 0),  # start of subtask 0
            (1.0, 0),  # end of subtask 0
            (0.0, 1),  # start of subtask 1
            (0.5, 1),  # middle of subtask 1
            (1.0, 2),  # end of subtask 2
        ]
        
        for tau_val, stage_idx in test_cases:
            # Scalar version
            expected = compute_cumulative_progress_batch(tau_val, stage_idx, proportions)
            
            # Tensor version (single element)
            tau = torch.tensor([[[tau_val]]])  # (1, 1, 1)
            stages = torch.tensor([[stage_idx]])  # (1, 1)
            result = compute_cumulative_progress_batch(tau, stages, alpha, cumulative)
            
            assert abs(result[0, 0, 0].item() - expected) < 1e-6
    
    def test_batch_processing(self):
        """Test batch processing with multiple samples."""
        proportions = [0.4, 0.6]
        alpha = torch.tensor(proportions, dtype=torch.float32)
        cumulative = torch.zeros(3, dtype=torch.float32)
        cumulative[1:] = torch.cumsum(alpha, dim=0)
        
        # Batch of 2 samples, sequence length 3
        tau = torch.tensor([
            [[0.0], [0.5], [1.0]],  # sample 1
            [[0.0], [0.5], [1.0]],  # sample 2
        ])
        stages = torch.tensor([
            [0, 0, 0],  # sample 1: all in subtask 0
            [1, 1, 1],  # sample 2: all in subtask 1
        ])
        
        result = compute_cumulative_progress_batch(tau, stages, alpha, cumulative)
        
        # Sample 1: subtask 0 with tau 0, 0.5, 1.0 -> y = 0, 0.2, 0.4
        assert abs(result[0, 0, 0].item() - 0.0) < 1e-6
        assert abs(result[0, 1, 0].item() - 0.2) < 1e-6
        assert abs(result[0, 2, 0].item() - 0.4) < 1e-6
        
        # Sample 2: subtask 1 with tau 0, 0.5, 1.0 -> y = 0.4, 0.7, 1.0
        assert abs(result[1, 0, 0].item() - 0.4) < 1e-6
        assert abs(result[1, 1, 0].item() - 0.7) < 1e-6
        assert abs(result[1, 2, 0].item() - 1.0) < 1e-6
    
    def test_auto_compute_cumulative_prior(self):
        """Test that cumulative_prior is auto-computed when not provided."""
        proportions = [0.3, 0.5, 0.2]
        alpha = torch.tensor(proportions, dtype=torch.float32)
        
        tau = torch.tensor([[[0.5]]])
        stages = torch.tensor([[1]])
        
        # Without cumulative_prior (should auto-compute)
        result = compute_cumulative_progress_batch(tau, stages, alpha)
        
        # Expected: P_0 + alpha_1 * 0.5 = 0.3 + 0.5 * 0.5 = 0.55
        assert abs(result[0, 0, 0].item() - 0.55) < 1e-6


class TestEndToEndProgressLabeling:
    """End-to-end tests for progress label computation."""
    
    def test_consistent_semantic_meaning(self):
        """Test that same subtask completion maps to same progress across trajectories.
        
        This is the key semantic property: "end of subtask 1" should always 
        mean the same progress value regardless of trajectory speed.
        """
        proportions = [0.3, 0.5, 0.2]
        
        # Fast trajectory: subtask 1 ends at frame 30 (of 100)
        tau_fast = compute_tau(30, 0, 30)  # = 1.0
        y_fast = compute_cumulative_progress_batch(tau_fast, 0, proportions)
        
        # Slow trajectory: subtask 1 ends at frame 90 (of 300)
        tau_slow = compute_tau(90, 0, 90)  # = 1.0
        y_slow = compute_cumulative_progress_batch(tau_slow, 0, proportions)
        
        # Both should map to same progress (0.3 = end of subtask 1)
        assert abs(y_fast - y_slow) < 1e-6
        assert abs(y_fast - 0.3) < 1e-6
    
    def test_monotonic_within_subtask(self):
        """Test that progress is monotonically increasing within a subtask."""
        proportions = [0.4, 0.6]
        
        prev_y = -1
        for tau in np.linspace(0, 1, 11):
            y = compute_cumulative_progress_batch(tau, 0, proportions)
            assert y > prev_y or (tau == 0 and y == 0)
            prev_y = y
    
    def test_continuous_across_subtasks(self):
        """Test that progress is continuous at subtask boundaries."""
        proportions = [0.3, 0.5, 0.2]
        
        # End of subtask 0 (tau=1.0)
        y_end_0 = compute_cumulative_progress_batch(1.0, 0, proportions)
        
        # Start of subtask 1 (tau=0.0)
        y_start_1 = compute_cumulative_progress_batch(0.0, 1, proportions)
        
        # Should be equal (P_1 = 0.3)
        assert abs(y_end_0 - y_start_1) < 1e-6
        
        # End of subtask 1
        y_end_1 = compute_cumulative_progress_batch(1.0, 1, proportions)
        
        # Start of subtask 2
        y_start_2 = compute_cumulative_progress_batch(0.0, 2, proportions)
        
        # Should be equal (P_2 = 0.8)
        assert abs(y_end_1 - y_start_2) < 1e-6


class TestApplyTemporalAugmentation:
    """Tests for _apply_temporal_augmentation (SARM Paper A.4 rewind augmentation).
    
    Rewind augmentation simulates going backwards from a stopping point.
    Example: [1,2,3,4,5,6] with n=2 → [1,2,3,4,3,2]
    (progress to 4, then rewind: 4→3→2)
    """
    
    @staticmethod
    def apply_temporal_augmentation(
        video: torch.Tensor,
        progress: torch.Tensor,
        state: torch.Tensor | None,
        num_reverse: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Standalone implementation of rewind augmentation for testing.
        
        This mirrors the logic in SarmPolicy._apply_temporal_augmentation
        but with deterministic num_reverse for testing.
        """
        seq_len = video.shape[0]
        
        # Cut point: keep frames up to here (exclusive of last num_reverse)
        cut_idx = seq_len - num_reverse
        
        # Rewind: go backwards from (cut_idx - 1) for num_reverse steps
        rewind_start = cut_idx - num_reverse - 1
        rewind_end = cut_idx - 1
        
        keep_video = video[:cut_idx]
        rewind_video = video[rewind_start:rewind_end].flip(0)
        video = torch.cat([keep_video, rewind_video], dim=0)
        
        keep_progress = progress[:cut_idx]
        rewind_progress = progress[rewind_start:rewind_end].flip(0)
        progress = torch.cat([keep_progress, rewind_progress], dim=0)
        
        if state is not None:
            keep_state = state[:cut_idx]
            rewind_state = state[rewind_start:rewind_end].flip(0)
            state = torch.cat([keep_state, rewind_state], dim=0)
        
        return video, progress, state
    
    def test_basic_example_n1(self):
        """Test with n=1: [1,2,3,4,5,6] → [1,2,3,4,5,4]."""
        video = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
        progress = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
        
        result_video, result_progress, _ = self.apply_temporal_augmentation(
            video, progress, state=None, num_reverse=1
        )
        
        # cut_idx = 6 - 1 = 5, keep [1,2,3,4,5]
        # rewind_start = 5 - 1 - 1 = 3, rewind_end = 5 - 1 = 4
        # video[3:4] = [4], flip = [4]
        expected_video = torch.tensor([1, 2, 3, 4, 5, 4], dtype=torch.float32)
        expected_progress = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.4], dtype=torch.float32)
        
        assert torch.allclose(result_video, expected_video)
        assert torch.allclose(result_progress, expected_progress)
    
    def test_basic_example_n4(self):
        """Test with n=4 on longer sequence: [1,2,3,4,5,6,7,8,9,10] → [1,2,3,4,5,6,5,4,3,2]."""
        video = torch.arange(1, 11, dtype=torch.float32)  # [1,2,3,4,5,6,7,8,9,10]
        progress = torch.arange(0.1, 1.1, 0.1, dtype=torch.float32)
        
        result_video, result_progress, _ = self.apply_temporal_augmentation(
            video, progress, state=None, num_reverse=4
        )
        
        # cut_idx = 10 - 4 = 6, keep [1,2,3,4,5,6]
        # rewind_start = 6 - 4 - 1 = 1, rewind_end = 6 - 1 = 5
        # video[1:5] = [2,3,4,5], flip = [5,4,3,2]
        expected_video = torch.tensor([1, 2, 3, 4, 5, 6, 5, 4, 3, 2], dtype=torch.float32)
        
        assert torch.allclose(result_video, expected_video)
    
    def test_with_state(self):
        """Test that state is also augmented correctly."""
        video = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
        progress = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
        state = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=torch.float32)
        
        result_video, result_progress, result_state = self.apply_temporal_augmentation(
            video, progress, state=state, num_reverse=2
        )
        
        expected_state = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [3, 3], [2, 2]], dtype=torch.float32)
        
        assert result_state is not None
        assert torch.allclose(result_state, expected_state)

    def test_rewind_simulates_going_backwards(self):
        """Test that the rewind part goes backwards through previously seen frames.
        
        The key property: after the cut point, we see frames from before the cut point
        in reverse order (simulating rewinding).
        """
        video = torch.arange(3, 11, dtype=torch.float32)  # [3,4,5,6,7,8,9,10]
        progress = torch.arange(0.1, 0.9, 0.1, dtype=torch.float32)
        
        result_video, _, _ = self.apply_temporal_augmentation(
            video, progress, state=None, num_reverse=3
        )
        
        # cut_idx = 8 - 3 = 5, keep [3,4,5,6,7]
        # rewind_start = 5 - 3 - 1 = 1, rewind_end = 5 - 1 = 4
        # video[1:4] = [4,5,6], flip = [6,5,4]
        
        # First part is forward progress
        assert torch.allclose(result_video[:5], torch.tensor([3, 4, 5, 6, 7], dtype=torch.float32))
        
        # Second part is backwards (rewind)
        assert torch.allclose(result_video[5:], torch.tensor([6, 5, 4], dtype=torch.float32))

