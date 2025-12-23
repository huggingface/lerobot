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

import numpy as np
import pytest
import torch

from lerobot.policies.sarm.sarm_utils import (
    apply_rewind_augmentation,
    compute_absolute_indices,
    compute_tau,
    find_stage_and_tau,
    normalize_stage_tau,
    temporal_proportions_to_breakpoints,
)


class TestProgressLabelsWithModes:
    """End-to-end tests for progress label generation in different modes."""

    def test_sparse_mode_single_stage(self):
        """Sparse mode with single stage should give linear progress."""
        episode_length = 300
        global_names = ["task"]
        proportions = {"task": 1.0}

        # Test at various frames
        for frame in [0, 100, 200, 299]:
            stage, tau = find_stage_and_tau(
                frame, episode_length, None, None, None, global_names, proportions
            )

            expected_tau = frame / (episode_length - 1)
            assert stage == 0
            assert abs(tau - expected_tau) < 1e-5

    def test_sparse_mode_multi_stage(self):
        """Sparse mode with multiple stages."""
        global_names = ["reach", "grasp", "lift", "place"]
        proportions = {"reach": 0.2, "grasp": 0.2, "lift": 0.3, "place": 0.3}

        subtask_names = ["reach", "grasp", "lift", "place"]
        subtask_starts = [0, 60, 120, 210]
        subtask_ends = [59, 119, 209, 299]

        # Check stages are correctly identified
        stage_at_30, _ = find_stage_and_tau(
            30, 300, subtask_names, subtask_starts, subtask_ends, global_names, proportions
        )
        assert stage_at_30 == 0

        stage_at_90, _ = find_stage_and_tau(
            90, 300, subtask_names, subtask_starts, subtask_ends, global_names, proportions
        )
        assert stage_at_90 == 1

        stage_at_150, _ = find_stage_and_tau(
            150, 300, subtask_names, subtask_starts, subtask_ends, global_names, proportions
        )
        assert stage_at_150 == 2

    def test_dense_mode_more_stages(self):
        """Dense mode should work with more fine-grained stages."""
        global_names = ["a", "b", "c", "d", "e", "f", "g", "h"]
        proportions = dict.fromkeys(global_names, 1 / 8)

        subtask_names = global_names
        subtask_starts = [i * 50 for i in range(8)]
        subtask_ends = [(i + 1) * 50 - 1 for i in range(8)]

        # Each stage should occupy 50 frames
        for stage_idx in range(8):
            mid_frame = stage_idx * 50 + 25
            stage, _ = find_stage_and_tau(
                mid_frame, 400, subtask_names, subtask_starts, subtask_ends, global_names, proportions
            )
            assert stage == stage_idx


class TestComputeAbsoluteIndices:
    """Tests for compute_absolute_indices (bidirectional sampling)."""

    def test_no_clamping_when_in_middle(self):
        """When frame is in middle of episode, no clamping should occur."""
        frame_idx = 300
        ep_start = 0
        ep_end = 1000
        n_obs_steps = 8
        frame_gap = 30

        indices, out_of_bounds = compute_absolute_indices(frame_idx, ep_start, ep_end, n_obs_steps, frame_gap)

        # All should be valid (no out of bounds)
        assert out_of_bounds.sum() == 0

        # Check bidirectional indices: [-120, -90, -60, -30, 0, 30, 60, 90, 120] from center
        half_steps = n_obs_steps // 2
        expected = (
            [frame_idx - frame_gap * i for i in range(half_steps, 0, -1)]
            + [frame_idx]
            + [frame_idx + frame_gap * i for i in range(1, half_steps + 1)]
        )
        assert indices.tolist() == expected

        # Center frame (index 4) should be frame_idx
        assert indices[half_steps] == frame_idx

    def test_clamping_at_episode_start(self):
        """Early frames should be clamped to episode start."""
        frame_idx = 50  # Not enough history for full past window
        ep_start = 0
        ep_end = 1000
        n_obs_steps = 8
        frame_gap = 30

        indices, out_of_bounds = compute_absolute_indices(frame_idx, ep_start, ep_end, n_obs_steps, frame_gap)

        # Some past frames should be clamped (out_of_bounds = 1)
        assert out_of_bounds.sum() > 0

        # All indices should be >= ep_start
        assert (indices >= ep_start).all()

        # Center index should be frame_idx
        half_steps = n_obs_steps // 2
        assert indices[half_steps] == frame_idx

    def test_clamping_at_episode_end(self):
        """Late frames should be clamped to episode end."""
        frame_idx = 950  # Not enough future for full window
        ep_start = 0
        ep_end = 1000
        n_obs_steps = 8
        frame_gap = 30

        indices, out_of_bounds = compute_absolute_indices(frame_idx, ep_start, ep_end, n_obs_steps, frame_gap)

        # Some future frames should be clamped
        assert out_of_bounds.sum() > 0

        # All indices should be < ep_end
        assert (indices < ep_end).all()

        # Center index should be frame_idx
        half_steps = n_obs_steps // 2
        assert indices[half_steps] == frame_idx

    def test_sequence_is_monotonic(self):
        """Frame indices should be monotonically increasing."""
        for frame_idx in [50, 100, 300, 950]:
            indices, _ = compute_absolute_indices(frame_idx, 0, 1000, 8, 30)

            # Check monotonic (non-decreasing due to clamping)
            diffs = indices[1:] - indices[:-1]
            assert (diffs >= 0).all(), f"Non-monotonic at frame {frame_idx}"


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


class TestFindStageAndTau:
    """Tests for find_stage_and_tau logic.

    This function is the core of progress label computation. It determines
    which stage a frame belongs to and the within-stage progress (tau).
    """

    def test_single_stage_mode_linear_progress(self):
        """Single-stage mode should give linear progress from 0 to 1."""
        episode_length = 100

        # Frame 0 -> tau = 0
        stage, tau = find_stage_and_tau(0, episode_length, None, None, None, ["task"], {"task": 1.0})
        assert stage == 0
        assert abs(tau - 0.0) < 1e-6

        # Frame 50 -> tau = 0.505 (50/99)
        stage, tau = find_stage_and_tau(50, episode_length, None, None, None, ["task"], {"task": 1.0})
        assert stage == 0
        assert abs(tau - 50 / 99) < 1e-6

        # Frame 99 -> tau = 1.0
        stage, tau = find_stage_and_tau(99, episode_length, None, None, None, ["task"], {"task": 1.0})
        assert stage == 0
        assert abs(tau - 1.0) < 1e-6

    def test_multi_stage_within_subtask(self):
        """Test finding stage when frame is within a subtask."""
        global_names = ["reach", "grasp", "lift"]
        proportions = {"reach": 0.3, "grasp": 0.2, "lift": 0.5}

        subtask_names = ["reach", "grasp", "lift"]
        subtask_starts = [0, 30, 50]
        subtask_ends = [29, 49, 99]

        # Frame 15 in "reach" stage (index 0)
        stage, tau = find_stage_and_tau(
            15, 100, subtask_names, subtask_starts, subtask_ends, global_names, proportions
        )
        assert stage == 0
        assert abs(tau - 15 / 29) < 1e-6

        # Frame 40 in "grasp" stage (index 1)
        stage, tau = find_stage_and_tau(
            40, 100, subtask_names, subtask_starts, subtask_ends, global_names, proportions
        )
        assert stage == 1
        # tau = (40 - 30) / (49 - 30) = 10/19
        assert abs(tau - 10 / 19) < 1e-6

        # Frame 75 in "lift" stage (index 2)
        stage, tau = find_stage_and_tau(
            75, 100, subtask_names, subtask_starts, subtask_ends, global_names, proportions
        )
        assert stage == 2
        # tau = (75 - 50) / (99 - 50) = 25/49
        assert abs(tau - 25 / 49) < 1e-6

    def test_frame_at_subtask_boundaries(self):
        """Test frames exactly at subtask boundaries."""
        global_names = ["a", "b"]
        proportions = {"a": 0.5, "b": 0.5}

        subtask_names = ["a", "b"]
        subtask_starts = [0, 50]
        subtask_ends = [49, 99]

        # Frame at start of first subtask
        stage, tau = find_stage_and_tau(
            0, 100, subtask_names, subtask_starts, subtask_ends, global_names, proportions
        )
        assert stage == 0
        assert tau == 0.0

        # Frame at end of first subtask
        stage, tau = find_stage_and_tau(
            49, 100, subtask_names, subtask_starts, subtask_ends, global_names, proportions
        )
        assert stage == 0
        assert tau == 1.0

        # Frame at start of second subtask
        stage, tau = find_stage_and_tau(
            50, 100, subtask_names, subtask_starts, subtask_ends, global_names, proportions
        )
        assert stage == 1
        assert tau == 0.0

    def test_frame_after_last_subtask(self):
        """Frames after last subtask should return last stage with high tau."""
        global_names = ["a", "b"]
        proportions = {"a": 0.5, "b": 0.5}

        subtask_names = ["a", "b"]
        subtask_starts = [0, 30]
        subtask_ends = [29, 59]

        # Frame 80 is after last subtask
        stage, tau = find_stage_and_tau(
            80, 100, subtask_names, subtask_starts, subtask_ends, global_names, proportions
        )
        assert stage == 1  # Last stage
        assert tau == 0.999  # Nearly complete


class TestEndToEndProgressLabeling:
    """End-to-end tests for progress label computation using normalize_stage_tau."""

    def test_consistent_semantic_meaning(self):
        """Test that same subtask completion maps to same progress across trajectories.

        This is the key semantic property: "end of subtask 1" should always
        mean the same progress value regardless of trajectory speed.
        """
        proportions = [0.3, 0.5, 0.2]

        # Fast trajectory: subtask 1 ends at frame 30 (of 100)
        tau_fast = compute_tau(30, 0, 30)  # = 1.0
        y_fast = normalize_stage_tau(0 + tau_fast, temporal_proportions=proportions)

        # Slow trajectory: subtask 1 ends at frame 90 (of 300)
        tau_slow = compute_tau(90, 0, 90)  # = 1.0
        y_slow = normalize_stage_tau(0 + tau_slow, temporal_proportions=proportions)

        # Both should map to same progress (0.3 = end of subtask 1)
        assert abs(y_fast - y_slow) < 1e-6
        assert abs(y_fast - 0.3) < 1e-6

    def test_monotonic_within_subtask(self):
        """Test that progress is monotonically increasing within a subtask."""
        proportions = [0.4, 0.6]

        prev_y = -1
        for tau in np.linspace(0, 1, 11):
            y = normalize_stage_tau(0 + tau, temporal_proportions=proportions)
            assert y > prev_y or (tau == 0 and y == 0)
            prev_y = y

    def test_continuous_across_subtasks(self):
        """Test that progress is continuous at subtask boundaries."""
        proportions = [0.3, 0.5, 0.2]

        # End of subtask 0 (stage=0, tau=1.0) -> stage.tau = 1.0
        y_end_0 = normalize_stage_tau(0 + 1.0, temporal_proportions=proportions)

        # Start of subtask 1 (stage=1, tau=0.0) -> stage.tau = 1.0
        y_start_1 = normalize_stage_tau(1 + 0.0, temporal_proportions=proportions)

        # Should be equal (P_1 = 0.3)
        assert abs(y_end_0 - y_start_1) < 1e-6

        # End of subtask 1 (stage=1, tau=1.0) -> stage.tau = 2.0
        y_end_1 = normalize_stage_tau(1 + 1.0, temporal_proportions=proportions)

        # Start of subtask 2 (stage=2, tau=0.0) -> stage.tau = 2.0
        y_start_2 = normalize_stage_tau(2 + 0.0, temporal_proportions=proportions)

        # Should be equal (P_2 = 0.8)
        assert abs(y_end_1 - y_start_2) < 1e-6


class TestTemporalProportionsToBreakpoints:
    """Tests for temporal_proportions_to_breakpoints.

    Converts temporal proportions to cumulative breakpoints for normalization.
    Example: [0.3, 0.5, 0.2] -> [0.0, 0.3, 0.8, 1.0]
    """

    def test_basic_conversion(self):
        """Test basic conversion from proportions to breakpoints."""
        proportions = [0.3, 0.5, 0.2]
        breakpoints = temporal_proportions_to_breakpoints(proportions)

        assert breakpoints is not None
        assert len(breakpoints) == 4
        assert breakpoints[0] == 0.0
        assert abs(breakpoints[1] - 0.3) < 1e-6
        assert abs(breakpoints[2] - 0.8) < 1e-6
        assert breakpoints[3] == 1.0

    def test_dict_input(self):
        """Test with dict input."""
        proportions = {"a": 0.25, "b": 0.25, "c": 0.5}
        breakpoints = temporal_proportions_to_breakpoints(proportions)

        assert breakpoints is not None
        assert len(breakpoints) == 4
        assert breakpoints[0] == 0.0
        assert breakpoints[-1] == 1.0

    def test_dict_with_subtask_names_order(self):
        """Test that subtask_names determines order for dict input."""
        proportions = {"c": 0.5, "a": 0.2, "b": 0.3}  # Dict order
        subtask_names = ["a", "b", "c"]  # Different order

        breakpoints = temporal_proportions_to_breakpoints(proportions, subtask_names)

        # Breakpoints should follow subtask_names order: a=0.2, b=0.3, c=0.5
        assert abs(breakpoints[1] - 0.2) < 1e-6  # a
        assert abs(breakpoints[2] - 0.5) < 1e-6  # a + b = 0.5
        assert breakpoints[3] == 1.0  # a + b + c = 1.0

    def test_uniform_proportions(self):
        """Test with uniform proportions."""
        proportions = [0.25, 0.25, 0.25, 0.25]
        breakpoints = temporal_proportions_to_breakpoints(proportions)

        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        for i, (bp, exp) in enumerate(zip(breakpoints, expected, strict=True)):
            assert abs(bp - exp) < 1e-6, f"Breakpoint {i} mismatch"

    def test_none_input(self):
        """Test that None input returns None."""
        result = temporal_proportions_to_breakpoints(None)
        assert result is None

    def test_normalization(self):
        """Test that non-normalized proportions are normalized."""
        # Proportions sum to 2.0, not 1.0
        proportions = [0.6, 1.0, 0.4]
        breakpoints = temporal_proportions_to_breakpoints(proportions)

        # Should be normalized: [0.3, 0.5, 0.2] -> [0, 0.3, 0.8, 1.0]
        assert breakpoints[-1] == 1.0
        assert abs(breakpoints[1] - 0.3) < 1e-6


class TestNormalizeStageTau:
    """Tests for normalize_stage_tau.

    Normalizes stage+tau values to [0, 1] using breakpoints.
    """

    def test_linear_fallback(self):
        """Test linear normalization when only num_stages is provided."""
        # 4 stages, linear: [0, 0.25, 0.5, 0.75, 1.0]

        # Stage 0 start
        assert normalize_stage_tau(0.0, num_stages=4) == 0.0

        # Stage 0 end / Stage 1 start
        assert abs(normalize_stage_tau(1.0, num_stages=4) - 0.25) < 1e-6

        # Stage 1 middle
        assert abs(normalize_stage_tau(1.5, num_stages=4) - 0.375) < 1e-6

        # Stage 3 end
        assert normalize_stage_tau(4.0, num_stages=4) == 1.0

    def test_with_custom_breakpoints(self):
        """Test with custom breakpoints."""
        # Non-linear breakpoints
        breakpoints = [0.0, 0.1, 0.5, 1.0]  # 3 stages

        # Stage 0: maps [0, 1) to [0.0, 0.1)
        assert abs(normalize_stage_tau(0.5, breakpoints=breakpoints) - 0.05) < 1e-6

        # Stage 1: maps [1, 2) to [0.1, 0.5)
        assert abs(normalize_stage_tau(1.5, breakpoints=breakpoints) - 0.3) < 1e-6

        # Stage 2: maps [2, 3) to [0.5, 1.0)
        assert abs(normalize_stage_tau(2.5, breakpoints=breakpoints) - 0.75) < 1e-6

    def test_with_temporal_proportions(self):
        """Test with temporal proportions (auto-computed breakpoints)."""
        proportions = {"a": 0.2, "b": 0.3, "c": 0.5}
        subtask_names = ["a", "b", "c"]

        # Stage 0 end should map to 0.2
        result = normalize_stage_tau(1.0, temporal_proportions=proportions, subtask_names=subtask_names)
        assert abs(result - 0.2) < 1e-6

        # Stage 1 end should map to 0.5
        result = normalize_stage_tau(2.0, temporal_proportions=proportions, subtask_names=subtask_names)
        assert abs(result - 0.5) < 1e-6

    def test_tensor_input(self):
        """Test with tensor input."""
        x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        breakpoints = [0.0, 0.3, 0.8, 1.0]  # 3 stages

        result = normalize_stage_tau(x, breakpoints=breakpoints)

        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert abs(result[0].item() - 0.0) < 1e-6
        assert abs(result[2].item() - 0.3) < 1e-6  # End of stage 0
        assert abs(result[4].item() - 0.8) < 1e-6  # End of stage 1

    def test_clamping(self):
        """Test that output is clamped to [0, 1]."""
        # Below 0
        assert normalize_stage_tau(-0.5, num_stages=4) == 0.0

        # Above num_stages
        assert normalize_stage_tau(5.0, num_stages=4) == 1.0

    def test_batch_tensor(self):
        """Test with batched tensor."""
        x = torch.tensor([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]])  # (2, 3)

        result = normalize_stage_tau(x, num_stages=3)

        assert result.shape == (2, 3)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_requires_one_of_inputs(self):
        """Test that at least one input method is required."""
        with pytest.raises(ValueError):
            normalize_stage_tau(1.0)


class TestRewindAugmentation:
    """Tests for rewind augmentation logic with bidirectional observation sampling.

    Rewind appends frames before the earliest observation frame, going backwards.
    With bidirectional sampling centered at frame_idx:
    - Earliest obs frame = frame_idx - half_steps * frame_gap
    - Rewind goes backwards from that point
    """

    def test_rewind_indices_go_backwards_from_earliest_obs(self):
        """Rewind indices should go backwards from earliest observation frame."""
        frame_idx = 300  # Center of bidirectional window
        ep_start = 0
        n_obs_steps = 4  # half_steps = 2
        frame_gap = 30

        # Earliest obs frame = 300 - 2*30 = 240
        # Rewind goes backwards: 210, 180
        rewind_step, rewind_indices = apply_rewind_augmentation(
            frame_idx,
            ep_start,
            n_obs_steps=n_obs_steps,
            max_rewind_steps=2,
            frame_gap=frame_gap,
            rewind_step=2,
        )

        assert rewind_step == 2
        assert len(rewind_indices) == 2
        # First rewind frame is closest to obs window, second is further back
        assert rewind_indices[0] == 210  # 240 - 30
        assert rewind_indices[1] == 180  # 240 - 60
        assert rewind_indices[0] > rewind_indices[1], "Rewind should be descending"

    def test_rewind_goes_backward_through_history(self):
        """Rewind frames should go backward before the observation window."""
        frame_idx = 450  # Center of bidirectional window
        ep_start = 0
        n_obs_steps = 8  # half_steps = 4
        frame_gap = 30

        # Earliest obs frame = 450 - 4*30 = 330
        # Rewind from 330: [300, 270, 240]
        rewind_step, rewind_indices = apply_rewind_augmentation(
            frame_idx,
            ep_start,
            n_obs_steps=n_obs_steps,
            max_rewind_steps=4,
            frame_gap=frame_gap,
            rewind_step=3,
        )

        assert rewind_step == 3
        expected = [300, 270, 240]  # Going backwards from 330
        assert rewind_indices == expected

    def test_no_rewind_when_obs_window_at_episode_start(self):
        """No rewind when observation window reaches episode start."""
        frame_idx = 120  # Center of window
        ep_start = 0
        n_obs_steps = 8  # half_steps = 4
        frame_gap = 30

        # Earliest obs frame = 120 - 4*30 = 0 (at episode start)
        rewind_step, rewind_indices = apply_rewind_augmentation(
            frame_idx, ep_start, n_obs_steps=n_obs_steps, max_rewind_steps=4, frame_gap=frame_gap
        )

        # No room for rewind
        assert rewind_step == 0
        assert rewind_indices == []

    def test_rewind_targets_are_decreasing(self):
        """Progress targets for rewind frames should be decreasing."""
        # Simulate progress values
        obs_progress = [0.1, 0.2, 0.3, 0.4, 0.5]  # Forward progress

        # Rewind reverses progress
        rewind_indices = [4, 3, 2]  # Go backwards through indices
        rewind_progress = [obs_progress[i] for i in rewind_indices]

        # Should be decreasing
        for i in range(len(rewind_progress) - 1):
            assert rewind_progress[i] > rewind_progress[i + 1]
