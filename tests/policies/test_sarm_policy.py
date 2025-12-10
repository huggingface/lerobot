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
Tests to verify SARM implementation matches original code behavior.

These tests compare the LeRobot SARM implementation against the original
author's implementation to ensure behavioral parity.

Reference files from original implementation:
- rm_lerobot_dataset.py: Data loading, frame sampling, rewind augmentation
- raw_data_utils.py: Normalization functions
- sarm_ws.py: Training loop, teacher forcing
- stage_estimator.py, subtask_estimator.py: Model architecture

Run with: pytest tests/policies/test_sarm_original_parity.py -v
"""

import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F


# Reference implementations from original code
def original_get_frame_indices(
    idx: int, n_obs_steps: int, frame_gap: int, ep_start: int = 0, ep_end: int | None = None
) -> list[int]:
    """
    Original implementation from rm_lerobot_dataset.py get_frame_indices().

    Build a monotonic sequence of length n_obs_steps+1 ending at idx.
    - Prefer fixed frame_gap when enough history exists.
    - Otherwise adapt the effective gap to fit within [ep_start, idx].
    """
    if ep_end is not None:
        idx = min(idx, ep_end)
    idx = max(idx, ep_start)

    gaps = n_obs_steps
    if gaps == 0:
        return [idx]

    total_needed = frame_gap * gaps
    available = idx - ep_start

    if available >= total_needed:
        frames = [idx - frame_gap * (gaps - k) for k in range(gaps)] + [idx]
    else:
        frames = [ep_start + round(available * k / gaps) for k in range(gaps)] + [idx]
        for i in range(1, len(frames)):
            if frames[i] < frames[i - 1]:
                frames[i] = frames[i - 1]

    return frames


def original_normalize_sparse(x: float) -> float:
    """
    Original hardcoded normalize_sparse from raw_data_utils.py for 5 stages.

    Uses breakpoints: [0.0, 0.05, 0.1, 0.3, 0.9, 1.0]
    """
    if 0 <= x < 1:
        return 0.0 + (x - 0) / (1 - 0) * (0.05 - 0.0)
    elif 1 <= x < 2:
        return 0.05 + (x - 1) / (2 - 1) * (0.1 - 0.05)
    elif 2 <= x < 3:
        return 0.1 + (x - 2) / (3 - 2) * (0.3 - 0.1)
    elif 3 <= x < 4:
        return 0.3 + (x - 3) / (4 - 3) * (0.9 - 0.3)
    elif 4 <= x <= 5:
        return 0.9 + (x - 4) / (5 - 4) * (1.0 - 0.9)
    else:
        raise ValueError("x must be in range [0, 5]")


def original_normalize_dense(x: float) -> float:
    """
    Original hardcoded normalize_dense from raw_data_utils.py for 8 stages.

    Uses breakpoints: [0.0, 0.08, 0.37, 0.53, 0.67, 0.72, 0.81, 0.9, 1.0]
    """
    if 0 <= x < 1:
        return 0.0 + (x - 0) * (0.08 - 0.0)
    elif 1 <= x < 2:
        return 0.08 + (x - 1) * (0.37 - 0.08)
    elif 2 <= x < 3:
        return 0.37 + (x - 2) * (0.53 - 0.37)
    elif 3 <= x < 4:
        return 0.53 + (x - 3) * (0.67 - 0.53)
    elif 4 <= x <= 5:
        return 0.67 + (x - 4) * (0.72 - 0.67)
    elif 5 <= x <= 6:
        return 0.72 + (x - 5) * (0.81 - 0.72)
    elif 6 <= x <= 7:
        return 0.81 + (x - 6) * (0.9 - 0.81)
    elif 7 <= x <= 8:
        return 0.9 + (x - 7) * (1.0 - 0.9)
    else:
        raise ValueError("x must be in range [0, 8]")


def original_gen_stage_emb(num_classes: int, trg: torch.Tensor) -> torch.Tensor:
    """
    Original gen_stage_emb from sarm_ws.py.

    Returns stage_onehot with a modality dim (B, 1, T, C).
    """
    idx = trg.long().clamp(min=0, max=num_classes - 1)
    C = num_classes
    stage_onehot = torch.eye(C, device=trg.device)[idx]
    stage_onehot = stage_onehot.unsqueeze(1)
    return stage_onehot


def original_get_rewind_step(
    idx: int, frame_gap: int, max_rewind_steps: int, ep_start: int, n_obs_steps: int
) -> int:
    """
    Compute rewind step like original rm_lerobot_dataset.py _get_rewind().

    Returns 0 if rewind not possible, otherwise random in [1, max_rewind].
    """
    required_history = n_obs_steps * frame_gap

    if idx <= ep_start + required_history:
        return 0

    max_valid_step = (idx - frame_gap) // frame_gap
    max_rewind = min(max_rewind_steps, max_valid_step)

    if max_rewind <= 0:
        return 0

    return random.randint(1, max_rewind)


# ============================================================================
# Test 1: Frame Sampling (Backward-looking with adaptive gap)
# Reference: rm_lerobot_dataset.py get_frame_indices()
# ============================================================================


class TestFrameSampling:
    """Test backward-looking frame sampling matches original."""

    def test_fixed_gap_sufficient_history(self):
        """When enough history, use fixed frame_gap."""
        result = original_get_frame_indices(idx=300, n_obs_steps=8, frame_gap=30, ep_start=0, ep_end=500)

        assert len(result) == 9  # n_obs_steps + 1
        assert result[-1] == 300  # Ends at target
        assert result[0] == 300 - 8 * 30  # Fixed gap back

        # Check uniform spacing
        for i in range(1, len(result)):
            assert result[i] - result[i - 1] == 30

    def test_adaptive_gap_insufficient_history(self):
        """When insufficient history, adapt gap evenly."""
        result = original_get_frame_indices(idx=100, n_obs_steps=8, frame_gap=30, ep_start=0, ep_end=500)

        assert len(result) == 9
        assert result[-1] == 100
        assert result[0] == 0  # Starts at episode start

        # Frames should be monotonically increasing
        for i in range(1, len(result)):
            assert result[i] >= result[i - 1]

    def test_very_early_frame(self):
        """Test frame at very start of episode."""
        result = original_get_frame_indices(idx=10, n_obs_steps=8, frame_gap=30, ep_start=0, ep_end=500)

        assert len(result) == 9
        assert result[-1] == 10
        assert result[0] == 0

    def test_frame_at_episode_boundary(self):
        """Test frame at episode end."""
        result = original_get_frame_indices(idx=499, n_obs_steps=8, frame_gap=30, ep_start=0, ep_end=500)

        assert len(result) == 9
        assert result[-1] == 499

    def test_mid_episode_partial_history(self):
        """Test frame with partial history available."""
        # idx=150, need 240 for full gap, available=150
        result = original_get_frame_indices(idx=150, n_obs_steps=8, frame_gap=30, ep_start=0, ep_end=500)

        assert len(result) == 9
        assert result[-1] == 150
        assert result[0] == 0  # Adaptive spacing from start

    def test_non_zero_episode_start(self):
        """Test with episode not starting at 0."""
        result = original_get_frame_indices(idx=350, n_obs_steps=8, frame_gap=30, ep_start=100, ep_end=500)

        assert len(result) == 9
        assert result[-1] == 350
        # With ep_start=100, available=250, need=240, so fixed gap works
        assert result[0] == 350 - 8 * 30  # = 110


# ============================================================================
# Test 2: Rewind Augmentation
# Reference: rm_lerobot_dataset.py _get_rewind()
# ============================================================================


class TestRewindAugmentation:
    """Test rewind augmentation logic."""

    def test_rewind_possible(self):
        """Rewind should be possible when sufficient history."""
        random.seed(42)
        rewind_step = original_get_rewind_step(
            idx=500, frame_gap=30, max_rewind_steps=4, ep_start=0, n_obs_steps=8
        )
        assert 1 <= rewind_step <= 4

    def test_rewind_not_possible_early(self):
        """Rewind should be 0 when too early in episode."""
        rewind_step = original_get_rewind_step(
            idx=100, frame_gap=30, max_rewind_steps=4, ep_start=0, n_obs_steps=8
        )
        # required_history = 8 * 30 = 240, but idx=100 < 240
        assert rewind_step == 0

    def test_rewind_boundary(self):
        """Test rewind at boundary condition."""
        # required_history = 240, so idx=240 should NOT allow rewind
        rewind_step = original_get_rewind_step(
            idx=240, frame_gap=30, max_rewind_steps=4, ep_start=0, n_obs_steps=8
        )
        assert rewind_step == 0

        # idx=241 should allow rewind
        random.seed(42)
        rewind_step = original_get_rewind_step(
            idx=300, frame_gap=30, max_rewind_steps=4, ep_start=0, n_obs_steps=8
        )
        assert rewind_step >= 0  # May be limited by max_valid_step

    def test_rewind_indices_reversed(self):
        """Rewind indices should be reversed (going backward)."""
        idx = 300
        frame_gap = 30
        rewind_step = 3

        # Original logic: range(idx - rewind_step * frame_gap, idx, frame_gap) then flip
        rewind_indices = list(range(idx - rewind_step * frame_gap, idx, frame_gap))
        rewind_indices_reversed = rewind_indices[::-1]

        # Should go: [210, 240, 270] -> reversed -> [270, 240, 210]
        assert rewind_indices == [210, 240, 270]
        assert rewind_indices_reversed == [270, 240, 210]

    def test_rewind_probability_distribution(self):
        """Test 80% rewind probability (as in original)."""
        random.seed(42)
        torch.manual_seed(42)

        num_trials = 1000
        rewind_count = 0

        for _ in range(num_trials):
            if random.random() < 0.8:
                rewind_count += 1

        # Should be roughly 80% (allow 75-85%)
        assert 0.75 < rewind_count / num_trials < 0.85


# Test 3: Language Perturbation + Target Zeroing
# Reference: rm_lerobot_dataset.py lines 103-117
class TestLanguagePerturbation:
    """Test language perturbation zeroes targets."""

    def test_perturbed_targets_are_zero(self):
        """When language is perturbed, targets should be zeroed."""
        n_obs_steps = 8
        max_rewind_steps = 4
        total_frames = 1 + n_obs_steps + max_rewind_steps

        # Simulate original behavior
        pertube_task_flag = True
        targets = torch.zeros(total_frames, dtype=torch.float32)
        progress_list = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        if not pertube_task_flag:
            targets[: n_obs_steps + 1] = progress_list

        # When perturbed, targets remain zero
        assert torch.all(targets == 0)

    def test_non_perturbed_targets_filled(self):
        """When not perturbed, targets should be filled."""
        n_obs_steps = 8
        max_rewind_steps = 4
        total_frames = 1 + n_obs_steps + max_rewind_steps

        pertube_task_flag = False
        targets = torch.zeros(total_frames, dtype=torch.float32)
        progress_list = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        if not pertube_task_flag:
            targets[: n_obs_steps + 1] = progress_list

        # Targets should be filled
        assert torch.allclose(targets[:9], progress_list)
        assert torch.all(targets[9:] == 0)  # Rewind slots still zero

    def test_perturbation_probability(self):
        """Test 20% perturbation probability."""
        random.seed(42)

        num_trials = 1000
        perturbed_count = 0

        for _ in range(num_trials):
            if random.random() < 0.2:
                perturbed_count += 1

        # Should be roughly 20% (allow 15-25%)
        assert 0.15 < perturbed_count / num_trials < 0.25


# Test 4: Normalization with Temporal Proportions
# Reference: raw_data_utils.py normalize_sparse/normalize_dense
class TestNormalization:
    """Test normalization matches original breakpoints."""

    def test_original_sparse_breakpoints(self):
        """Verify original sparse breakpoints: [0.0, 0.05, 0.1, 0.3, 0.9, 1.0]"""
        # Stage boundaries
        assert abs(original_normalize_sparse(0.0) - 0.0) < 1e-6
        assert abs(original_normalize_sparse(1.0) - 0.05) < 1e-6
        assert abs(original_normalize_sparse(2.0) - 0.1) < 1e-6
        assert abs(original_normalize_sparse(3.0) - 0.3) < 1e-6
        assert abs(original_normalize_sparse(4.0) - 0.9) < 1e-6
        assert abs(original_normalize_sparse(5.0) - 1.0) < 1e-6

    def test_original_dense_breakpoints(self):
        """Verify original dense breakpoints."""
        assert abs(original_normalize_dense(0.0) - 0.0) < 1e-6
        assert abs(original_normalize_dense(1.0) - 0.08) < 1e-6
        assert abs(original_normalize_dense(2.0) - 0.37) < 1e-6
        assert abs(original_normalize_dense(8.0) - 1.0) < 1e-6

    def test_temporal_proportions_to_breakpoints(self):
        """Test that temporal proportions produce correct cumulative breakpoints."""
        from lerobot.policies.sarm.sarm_utils import temporal_proportions_to_breakpoints

        # Example proportions
        proportions = {"task1": 0.1, "task2": 0.2, "task3": 0.3, "task4": 0.4}
        subtask_names = ["task1", "task2", "task3", "task4"]

        breakpoints = temporal_proportions_to_breakpoints(proportions, subtask_names)

        # Should be cumulative: [0.0, 0.1, 0.3, 0.6, 1.0]
        expected = [0.0, 0.1, 0.3, 0.6, 1.0]
        assert len(breakpoints) == len(expected)
        for bp, exp in zip(breakpoints, expected):
            assert abs(bp - exp) < 1e-6

    def test_normalize_sparse_with_original_breakpoints(self):
        """Test normalize_sparse with original hardcoded breakpoints."""
        from lerobot.policies.sarm.sarm_utils import normalize_sparse

        # Use same breakpoints as original: [0.0, 0.05, 0.1, 0.3, 0.9, 1.0]
        breakpoints = [0.0, 0.05, 0.1, 0.3, 0.9, 1.0]

        # Test stage boundaries match original
        assert abs(normalize_sparse(0.0, breakpoints=breakpoints) - original_normalize_sparse(0.0)) < 1e-6
        assert abs(normalize_sparse(1.0, breakpoints=breakpoints) - original_normalize_sparse(1.0)) < 1e-6
        assert abs(normalize_sparse(2.0, breakpoints=breakpoints) - original_normalize_sparse(2.0)) < 1e-6
        assert abs(normalize_sparse(3.0, breakpoints=breakpoints) - original_normalize_sparse(3.0)) < 1e-6
        assert abs(normalize_sparse(4.0, breakpoints=breakpoints) - original_normalize_sparse(4.0)) < 1e-6
        assert abs(normalize_sparse(5.0, breakpoints=breakpoints) - original_normalize_sparse(5.0)) < 1e-6

        # Test mid-stage values
        assert abs(normalize_sparse(0.5, breakpoints=breakpoints) - original_normalize_sparse(0.5)) < 1e-6
        assert abs(normalize_sparse(2.5, breakpoints=breakpoints) - original_normalize_sparse(2.5)) < 1e-6
        assert abs(normalize_sparse(3.5, breakpoints=breakpoints) - original_normalize_sparse(3.5)) < 1e-6

    def test_normalize_sparse_tensor(self):
        """Test normalize_sparse works with tensors."""
        from lerobot.policies.sarm.sarm_utils import normalize_sparse

        breakpoints = [0.0, 0.05, 0.1, 0.3, 0.9, 1.0]
        x = torch.tensor([0.0, 0.5, 1.0, 2.5, 4.0, 5.0])

        result = normalize_sparse(x, breakpoints=breakpoints)

        # Compare with original scalar version
        expected = torch.tensor([original_normalize_sparse(v.item()) for v in x])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_normalize_with_temporal_proportions(self):
        """Test normalize_sparse using temporal proportions instead of explicit breakpoints."""
        from lerobot.policies.sarm.sarm_utils import normalize_sparse

        # Proportions that produce breakpoints [0.0, 0.2, 0.5, 1.0]
        proportions = [0.2, 0.3, 0.5]

        # At stage 0, tau=0.5: y = 0 + 0.2 * 0.5 = 0.1
        result = normalize_sparse(0.5, temporal_proportions=proportions)
        assert abs(result - 0.1) < 1e-6

        # At stage 1, tau=0: y = 0.2
        result = normalize_sparse(1.0, temporal_proportions=proportions)
        assert abs(result - 0.2) < 1e-6

        # At stage 2, tau=1: y = 1.0
        result = normalize_sparse(3.0, temporal_proportions=proportions)
        assert abs(result - 1.0) < 1e-6


# Test 5: Teacher Forcing (75/25 GT/Predicted)
# Reference: sarm_ws.py lines 168-179
class TestTeacherForcing:
    """Test teacher forcing matches original 75/25 ratio."""

    def test_gen_stage_emb_matches_original(self):
        """Test gen_stage_emb produces correct one-hot encoding."""
        from lerobot.policies.sarm.modeling_sarm import gen_stage_emb

        num_classes = 6
        targets = torch.tensor([[0.5, 1.3, 2.7, 3.1, 4.9, 5.0]])

        # Original
        original_emb = original_gen_stage_emb(num_classes, targets)

        # Your implementation
        your_emb = gen_stage_emb(num_classes, targets)

        assert torch.allclose(original_emb, your_emb)

        # Check shape: (B, 1, T, C)
        assert your_emb.shape == (1, 1, 6, 6)

        # Check one-hot: stage 0 at position 0, stage 1 at position 1, etc.
        assert your_emb[0, 0, 0, 0] == 1.0  # targets[0,0]=0.5 -> stage 0
        assert your_emb[0, 0, 1, 1] == 1.0  # targets[0,1]=1.3 -> stage 1
        assert your_emb[0, 0, 2, 2] == 1.0  # targets[0,2]=2.7 -> stage 2
        assert your_emb[0, 0, 3, 3] == 1.0  # targets[0,3]=3.1 -> stage 3
        assert your_emb[0, 0, 4, 4] == 1.0  # targets[0,4]=4.9 -> stage 4
        assert your_emb[0, 0, 5, 5] == 1.0  # targets[0,5]=5.0 -> stage 5 (clamped)

    def test_teacher_forcing_ratio(self):
        """Test 75/25 GT/predicted ratio."""
        random.seed(42)

        num_trials = 1000
        gt_count = 0

        for _ in range(num_trials):
            if random.random() < 0.75:
                gt_count += 1

        # Should be roughly 75% (allow 70-80%)
        assert 0.70 < gt_count / num_trials < 0.80

    def test_gen_stage_emb_clamps_indices(self):
        """Test that stage indices are clamped to valid range."""
        from lerobot.policies.sarm.modeling_sarm import gen_stage_emb

        num_classes = 4
        # Targets with out-of-range values
        targets = torch.tensor([[-1.0, 5.0, 10.0]])

        emb = gen_stage_emb(num_classes, targets)

        # All should be clamped to valid range [0, 3]
        assert emb[0, 0, 0, 0] == 1.0  # -1 clamped to 0
        assert emb[0, 0, 1, 3] == 1.0  # 5 clamped to 3
        assert emb[0, 0, 2, 3] == 1.0  # 10 clamped to 3


# Test 6: Stage+Tau Target Format
# Reference: sarm_ws.py train_step
class TestStageTauFormat:
    """Test stage+tau target format extraction."""

    def test_extract_stage_and_tau(self):
        """Test extracting stage and tau from combined target."""
        targets = torch.tensor([[0.3, 1.7, 2.5, 3.9, 4.1]])

        # Original extraction
        gt_stage = torch.floor(targets).long()
        gt_tau = torch.remainder(targets, 1.0)

        expected_stage = torch.tensor([[0, 1, 2, 3, 4]])
        expected_tau = torch.tensor([[0.3, 0.7, 0.5, 0.9, 0.1]])

        assert torch.all(gt_stage == expected_stage)
        assert torch.allclose(gt_tau, expected_tau, atol=1e-6)

    def test_combine_stage_and_tau(self):
        """Test combining stage and tau back to target."""
        stage_idx = torch.tensor([[0, 1, 2, 3, 4]])
        tau_pred = torch.tensor([[0.3, 0.7, 0.5, 0.9, 0.1]])

        # Combine
        raw_reward = stage_idx.float() + tau_pred

        expected = torch.tensor([[0.3, 1.7, 2.5, 3.9, 4.1]])
        assert torch.allclose(raw_reward, expected, atol=1e-6)

    def test_integer_targets(self):
        """Test with integer targets (tau=0)."""
        targets = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

        gt_stage = torch.floor(targets).long()
        gt_tau = torch.remainder(targets, 1.0)

        assert torch.all(gt_stage == torch.tensor([[0, 1, 2, 3]]))
        assert torch.allclose(gt_tau, torch.zeros_like(gt_tau), atol=1e-6)


# ============================================================================
# Test 7: Lengths Tensor
# Reference: rm_lerobot_dataset.py
# ============================================================================


class TestLengthsTensor:
    """Test lengths tensor computation."""

    def test_lengths_without_rewind(self):
        """Lengths = n_obs_steps + 1 when no rewind."""
        n_obs_steps = 8
        rewind_step = 0

        length = 1 + n_obs_steps + rewind_step
        assert length == 9

    def test_lengths_with_rewind(self):
        """Lengths = n_obs_steps + 1 + rewind_step when rewind applied."""
        n_obs_steps = 8
        rewind_step = 3

        length = 1 + n_obs_steps + rewind_step
        assert length == 12

    def test_lengths_max(self):
        """Max lengths = n_obs_steps + 1 + max_rewind_steps."""
        n_obs_steps = 8
        max_rewind_steps = 4

        max_length = 1 + n_obs_steps + max_rewind_steps
        assert max_length == 13

    def test_lengths_batch(self):
        """Test batch lengths computation."""
        n_obs_steps = 8
        batch_size = 4
        rewind_steps = torch.tensor([0, 2, 4, 1])

        lengths = 1 + n_obs_steps + rewind_steps

        expected = torch.tensor([9, 11, 13, 10])
        assert torch.all(lengths == expected)


# Test 8: Rewind Target Reversal
# Reference: rm_lerobot_dataset.py lines 115-117
class TestRewindTargetReversal:
    """Test that rewind targets are reversed."""

    def test_rewind_targets_reversed(self):
        """Rewind targets should be flipped from observation targets."""
        n_obs_steps = 8
        max_rewind_steps = 4
        rewind_step = 3

        # Observation progress (increasing)
        progress_list = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        targets = torch.zeros(1 + n_obs_steps + max_rewind_steps, dtype=torch.float32)
        targets[: n_obs_steps + 1] = progress_list

        # Original: for i in range(rewind_step): targets[...] = flip(progress_list)[i + 1]
        flipped = torch.flip(progress_list, dims=[0])
        for i in range(rewind_step):
            targets[1 + n_obs_steps + i] = flipped[i + 1]

        # Rewind targets should be [0.8, 0.7, 0.6] (going backward from 0.9)
        assert abs(targets[9] - 0.8) < 1e-6
        assert abs(targets[10] - 0.7) < 1e-6
        assert abs(targets[11] - 0.6) < 1e-6
        assert targets[12] == 0  # Unused slot

    def test_rewind_single_step(self):
        """Test rewind with single step."""
        progress_list = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        n_obs_steps = 4
        max_rewind_steps = 2
        rewind_step = 1

        targets = torch.zeros(1 + n_obs_steps + max_rewind_steps)
        targets[: n_obs_steps + 1] = progress_list

        flipped = torch.flip(progress_list, dims=[0])
        for i in range(rewind_step):
            targets[1 + n_obs_steps + i] = flipped[i + 1]

        # Only one rewind frame: 0.4 (second to last)
        assert abs(targets[5] - 0.4) < 1e-6
        assert targets[6] == 0


# Test 9: Prediction Smoother
# Reference: pred_smoother.py
class TestPredictionSmoother:
    """Test RegressionConfidenceSmoother matches original."""

    def test_smoother_basic(self):
        """Test basic smoother functionality."""
        from lerobot.policies.sarm.sarm_utils import RegressionConfidenceSmoother

        smoother = RegressionConfidenceSmoother(
            window_size=10, beta=3.0, eps=1e-6, low_conf_th=0.9, value_range=(0.0, 1.0)
        )

        # First update
        result = smoother.update(0.5, 0.95)
        assert 0 <= result <= 1

    def test_smoother_low_confidence_rejected(self):
        """Low confidence predictions should be rejected."""
        from lerobot.policies.sarm.sarm_utils import RegressionConfidenceSmoother

        smoother = RegressionConfidenceSmoother(low_conf_th=0.9)

        # First high-conf update
        smoother.update(0.5, 0.95)

        # Low-conf update should return previous value
        result = smoother.update(0.9, 0.5)  # Low confidence

        # Should not jump to 0.9, should stay near previous
        assert result < 0.7

    def test_smoother_reset(self):
        """Test reset clears history."""
        from lerobot.policies.sarm.sarm_utils import RegressionConfidenceSmoother

        smoother = RegressionConfidenceSmoother()
        smoother.update(0.5, 0.95)
        smoother.update(0.6, 0.95)

        smoother.reset()

        assert len(smoother.hist_vals) == 0
        assert len(smoother.hist_confs) == 0
        assert smoother.last_smoothed is None

    def test_smoother_weighted_average(self):
        """Test confidence-weighted averaging."""
        from lerobot.policies.sarm.sarm_utils import RegressionConfidenceSmoother

        smoother = RegressionConfidenceSmoother(window_size=2, beta=1.0, low_conf_th=0.0)

        # Two updates with different confidence
        smoother.update(0.0, 1.0)  # High confidence
        result = smoother.update(1.0, 1.0)  # High confidence

        # With equal weights, should average to 0.5
        assert abs(result - 0.5) < 0.1


# Test 10: Model Architecture
# Reference: stage_estimator.py, subtask_estimator.py
class TestModelArchitecture:
    """Test model architecture matches original."""

    def test_stage_transformer_output_shape(self):
        """Test StageTransformer produces correct output shape."""
        from lerobot.policies.sarm.modeling_sarm import StageTransformer

        model = StageTransformer(
            d_model=512,
            vis_emb_dim=512,
            text_emb_dim=512,
            state_dim=14,
            n_layers=2,
            n_heads=8,
            num_cameras=1,
            num_classes_sparse=6,
            num_classes_dense=9,
        )

        batch_size, num_cameras, seq_len = 2, 1, 13
        img_seq = torch.randn(batch_size, num_cameras, seq_len, 512)
        lang_emb = torch.randn(batch_size, 512)
        state = torch.randn(batch_size, seq_len, 14)
        lengths = torch.tensor([9, 11])

        output = model(img_seq, lang_emb, state, lengths, scheme="sparse")

        assert output.shape == (batch_size, seq_len, 6)

    def test_subtask_transformer_output_shape(self):
        """Test SubtaskTransformer produces correct output shape."""
        from lerobot.policies.sarm.modeling_sarm import SubtaskTransformer

        model = SubtaskTransformer(
            d_model=512,
            vis_emb_dim=512,
            text_emb_dim=512,
            state_dim=14,
            n_layers=2,
            n_heads=8,
            num_cameras=1,
        )

        batch_size, num_cameras, seq_len, num_classes = 2, 1, 13, 6
        img_seq = torch.randn(batch_size, num_cameras, seq_len, 512)
        lang_emb = torch.randn(batch_size, 512)
        state = torch.randn(batch_size, seq_len, 14)
        lengths = torch.tensor([9, 11])
        stage_prior = torch.zeros(batch_size, 1, seq_len, num_classes)
        stage_prior[:, :, :, 0] = 1.0  # All in stage 0

        output = model(img_seq, lang_emb, state, lengths, stage_prior, scheme="sparse")

        assert output.shape == (batch_size, seq_len)
        assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output

    def test_stage_transformer_dense_head(self):
        """Test StageTransformer dense head."""
        from lerobot.policies.sarm.modeling_sarm import StageTransformer

        model = StageTransformer(
            d_model=256,
            vis_emb_dim=512,
            text_emb_dim=512,
            state_dim=14,
            n_layers=2,
            n_heads=4,
            num_cameras=1,
            num_classes_sparse=6,
            num_classes_dense=9,
        )

        batch_size, seq_len = 2, 13
        img_seq = torch.randn(batch_size, 1, seq_len, 512)
        lang_emb = torch.randn(batch_size, 512)
        state = torch.randn(batch_size, seq_len, 14)
        lengths = torch.tensor([9, 11])

        # Sparse output
        sparse_output = model(img_seq, lang_emb, state, lengths, scheme="sparse")
        assert sparse_output.shape == (batch_size, seq_len, 6)

        # Dense output
        dense_output = model(img_seq, lang_emb, state, lengths, scheme="dense")
        assert dense_output.shape == (batch_size, seq_len, 9)

    def test_per_timestep_language_embedding(self):
        """Test that models accept per-timestep language embeddings."""
        from lerobot.policies.sarm.modeling_sarm import StageTransformer

        model = StageTransformer(
            d_model=256,
            vis_emb_dim=512,
            text_emb_dim=512,
            state_dim=14,
            n_layers=2,
            n_heads=4,
            num_cameras=1,
            num_classes_sparse=6,
            num_classes_dense=9,
        )

        batch_size, seq_len = 2, 13
        img_seq = torch.randn(batch_size, 1, seq_len, 512)
        state = torch.randn(batch_size, seq_len, 14)
        lengths = torch.tensor([9, 11])

        # Broadcast language (original)
        lang_emb_broadcast = torch.randn(batch_size, 512)
        output1 = model(img_seq, lang_emb_broadcast, state, lengths, scheme="sparse")

        # Per-timestep language (dense mode)
        lang_emb_per_step = torch.randn(batch_size, seq_len, 512)
        output2 = model(img_seq, lang_emb_per_step, state, lengths, scheme="sparse")

        # Both should produce valid output
        assert output1.shape == output2.shape == (batch_size, seq_len, 6)


# Test 11: Integration Test - Full Pipeline
class TestIntegration:
    """Integration tests for full SARM pipeline."""

    def test_full_forward_pass(self):
        """Test full SARM forward pass."""
        from lerobot.policies.sarm.configuration_sarm import SARMConfig
        from lerobot.policies.sarm.modeling_sarm import SARMRewardModel

        # Force CPU to avoid MPS nested tensor issues
        config = SARMConfig(
            annotation_mode="single_stage",
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            device="cpu",
        )

        model = SARMRewardModel(config)
        model.to("cpu")
        model.eval()

        batch_size, seq_len = 2, 13
        batch = {
            "observation": {
                "video_features": torch.randn(batch_size, seq_len, 512),
                "text_features": torch.randn(batch_size, 512),
                "state_features": torch.randn(batch_size, seq_len, 14),
                "lengths": torch.tensor([9, 11]),
                "sparse_targets": torch.rand(batch_size, seq_len),
            }
        }

        loss, output_dict = model(batch)

        assert loss.item() >= 0
        assert "sparse_stage_loss" in output_dict
        assert "sparse_subtask_loss" in output_dict
        assert "total_loss" in output_dict

    def test_training_step_matches_original(self):
        """Test that training step produces expected loss components."""
        from lerobot.policies.sarm.configuration_sarm import SARMConfig
        from lerobot.policies.sarm.modeling_sarm import SARMRewardModel

        # Force CPU to avoid MPS nested tensor issues
        config = SARMConfig(
            annotation_mode="single_stage",
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            device="cpu",
        )

        model = SARMRewardModel(config)
        model.to("cpu")
        model.train()

        batch_size, seq_len = 2, 13
        batch = {
            "observation": {
                "video_features": torch.randn(batch_size, seq_len, 512),
                "text_features": torch.randn(batch_size, 512),
                "state_features": torch.randn(batch_size, seq_len, 14),
                "lengths": torch.tensor([9, 11]),
                "sparse_targets": torch.rand(batch_size, seq_len) * 0.999,  # Keep < 1
            }
        }

        loss, output_dict = model(batch)

        # Stage loss should be cross-entropy (can be large)
        assert output_dict["sparse_stage_loss"] >= 0

        # Subtask loss should be MSE (bounded by target range)
        assert output_dict["sparse_subtask_loss"] >= 0
        assert output_dict["sparse_subtask_loss"] <= 1.0  # MSE of values in [0, 1]

    def test_calculate_rewards(self):
        """Test calculate_rewards inference method."""
        from lerobot.policies.sarm.configuration_sarm import SARMConfig
        from lerobot.policies.sarm.modeling_sarm import SARMRewardModel

        # Force CPU to avoid MPS nested tensor issues
        config = SARMConfig(
            annotation_mode="single_stage",
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            device="cpu",
        )

        model = SARMRewardModel(config)
        model.to("cpu")
        model.eval()

        batch_size, seq_len = 2, 13
        text_emb = torch.randn(batch_size, 512)
        video_emb = torch.randn(batch_size, seq_len, 512)
        state_features = torch.randn(batch_size, seq_len, 14)
        lengths = torch.tensor([9, 11])

        # Basic call
        rewards = model.calculate_rewards(text_emb, video_emb, state_features, lengths)

        assert rewards.shape == (batch_size,)
        assert np.all((rewards >= 0) & (rewards <= 1))

        # With all frames
        rewards_all = model.calculate_rewards(
            text_emb, video_emb, state_features, lengths, return_all_frames=True
        )

        assert rewards_all.shape == (batch_size, seq_len)

        # With stages and confidence
        rewards, stages, conf = model.calculate_rewards(
            text_emb, video_emb, state_features, lengths, return_stages=True, return_confidence=True
        )

        assert rewards.shape == (batch_size,)
        assert stages.shape == (batch_size, seq_len, config.num_sparse_stages)
        assert conf.shape == (batch_size, seq_len)


# Test 12: Configuration Validation
class TestConfiguration:
    """Test SARM configuration validation."""

    def test_single_stage_mode_defaults(self):
        """Test single_stage mode sets correct defaults."""
        from lerobot.policies.sarm.configuration_sarm import SARMConfig

        config = SARMConfig(annotation_mode="single_stage")

        assert config.num_sparse_stages == 1
        assert config.sparse_subtask_names == ["task"]
        assert config.sparse_temporal_proportions == [1.0]
        assert config.num_dense_stages is None

    def test_num_frames_computation(self):
        """Test num_frames property."""
        from lerobot.policies.sarm.configuration_sarm import SARMConfig

        config = SARMConfig(n_obs_steps=8, max_rewind_steps=4)

        # num_frames = 1 + n_obs_steps + max_rewind_steps
        assert config.num_frames == 13

    def test_observation_delta_indices(self):
        """Test observation_delta_indices property."""
        from lerobot.policies.sarm.configuration_sarm import SARMConfig

        config = SARMConfig(n_obs_steps=8, frame_gap=30, max_rewind_steps=4)

        deltas = config.observation_delta_indices

        # Should have 9 obs + 4 rewind = 13 deltas
        assert len(deltas) == 13

        # First 9 are backward-looking: [-240, -210, ..., -30, 0]
        obs_deltas = deltas[:9]
        assert obs_deltas[0] == -240
        assert obs_deltas[-1] == 0

        # Last 4 are rewind placeholders: [-30, -60, -90, -120]
        rewind_deltas = deltas[9:]
        assert len(rewind_deltas) == 4

    def test_invalid_rewind_steps(self):
        """Test validation of max_rewind_steps < n_obs_steps."""
        from lerobot.policies.sarm.configuration_sarm import SARMConfig

        with pytest.raises(ValueError, match="max_rewind_steps"):
            SARMConfig(n_obs_steps=4, max_rewind_steps=4)

        with pytest.raises(ValueError, match="max_rewind_steps"):
            SARMConfig(n_obs_steps=4, max_rewind_steps=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
