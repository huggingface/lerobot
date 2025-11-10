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

"""Tests for RTC modeling module (RTCProcessor)."""

import pytest
import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor

# ====================== Fixtures ======================


@pytest.fixture
def rtc_config_debug_enabled():
    """Create RTC config with debug enabled."""
    return RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
        max_guidance_weight=10.0,
        execution_horizon=10,
        debug=True,
        debug_maxlen=100,
    )


@pytest.fixture
def rtc_config_debug_disabled():
    """Create RTC config with debug disabled."""
    return RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
        max_guidance_weight=10.0,
        execution_horizon=10,
        debug=False,
    )


@pytest.fixture
def rtc_processor_debug_enabled(rtc_config_debug_enabled):
    """Create RTCProcessor with debug enabled."""
    return RTCProcessor(rtc_config_debug_enabled)


@pytest.fixture
def rtc_processor_debug_disabled(rtc_config_debug_disabled):
    """Create RTCProcessor with debug disabled."""
    return RTCProcessor(rtc_config_debug_disabled)


@pytest.fixture
def sample_x_t():
    """Create sample x_t tensor (batch, time, action_dim)."""
    return torch.randn(1, 50, 6)


@pytest.fixture
def sample_prev_chunk():
    """Create sample previous chunk tensor."""
    return torch.randn(1, 50, 6)


# ====================== Initialization Tests ======================


def test_rtc_processor_initialization_with_debug(rtc_config_debug_enabled):
    """Test RTCProcessor initializes with debug tracker."""
    processor = RTCProcessor(rtc_config_debug_enabled)
    assert processor.rtc_config == rtc_config_debug_enabled
    assert processor.tracker is not None
    assert processor.tracker.enabled is True


def test_rtc_processor_initialization_without_debug(rtc_config_debug_disabled):
    """Test RTCProcessor initializes without debug tracker."""
    processor = RTCProcessor(rtc_config_debug_disabled)
    assert processor.rtc_config == rtc_config_debug_disabled
    assert processor.tracker is None


# ====================== Tracker Proxy Methods Tests ======================


def test_track_when_tracker_enabled(rtc_processor_debug_enabled, sample_x_t):
    """Test track() forwards to tracker when enabled."""
    rtc_processor_debug_enabled.track(
        time=torch.tensor(0.5),
        x_t=sample_x_t,
        v_t=sample_x_t,
        guidance_weight=2.0,
    )

    # Should have tracked one step
    steps = rtc_processor_debug_enabled.get_all_debug_steps()
    assert len(steps) == 1
    assert steps[0].time == 0.5


def test_track_when_tracker_disabled(rtc_processor_debug_disabled, sample_x_t):
    """Test track() does nothing when tracker disabled."""
    # Should not raise error
    rtc_processor_debug_disabled.track(
        time=torch.tensor(0.5),
        x_t=sample_x_t,
        v_t=sample_x_t,
    )

    # Should return empty list
    steps = rtc_processor_debug_disabled.get_all_debug_steps()
    assert len(steps) == 0


def test_get_all_debug_steps_when_enabled(rtc_processor_debug_enabled, sample_x_t):
    """Test get_all_debug_steps() returns tracked steps."""
    rtc_processor_debug_enabled.track(time=torch.tensor(0.5), x_t=sample_x_t)
    rtc_processor_debug_enabled.track(time=torch.tensor(0.4), x_t=sample_x_t)

    steps = rtc_processor_debug_enabled.get_all_debug_steps()
    assert len(steps) == 2


def test_get_all_debug_steps_when_disabled(rtc_processor_debug_disabled):
    """Test get_all_debug_steps() returns empty list when disabled."""
    steps = rtc_processor_debug_disabled.get_all_debug_steps()
    assert steps == []
    assert isinstance(steps, list)


def test_is_debug_enabled_when_tracker_exists(rtc_processor_debug_enabled):
    """Test is_debug_enabled() returns True when tracker enabled."""
    assert rtc_processor_debug_enabled.is_debug_enabled() is True


def test_is_debug_enabled_when_tracker_disabled(rtc_processor_debug_disabled):
    """Test is_debug_enabled() returns False when tracker disabled."""
    assert rtc_processor_debug_disabled.is_debug_enabled() is False


def test_reset_tracker_when_enabled(rtc_processor_debug_enabled, sample_x_t):
    """Test reset_tracker() clears tracked steps."""
    rtc_processor_debug_enabled.track(time=torch.tensor(0.5), x_t=sample_x_t)
    rtc_processor_debug_enabled.track(time=torch.tensor(0.4), x_t=sample_x_t)
    assert len(rtc_processor_debug_enabled.get_all_debug_steps()) == 2

    rtc_processor_debug_enabled.reset_tracker()
    assert len(rtc_processor_debug_enabled.get_all_debug_steps()) == 0


def test_reset_tracker_when_disabled(rtc_processor_debug_disabled):
    """Test reset_tracker() doesn't error when tracker disabled."""
    rtc_processor_debug_disabled.reset_tracker()  # Should not raise


# ====================== get_prefix_weights Tests ======================


def test_get_prefix_weights_zeros_schedule():
    """Test get_prefix_weights with ZEROS schedule."""
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.ZEROS)
    processor = RTCProcessor(config)

    weights = processor.get_prefix_weights(start=5, end=10, total=20)

    # First 5 should be 1.0, rest should be 0.0
    assert weights.shape == (20,)
    assert torch.all(weights[:5] == 1.0)
    assert torch.all(weights[5:] == 0.0)


def test_get_prefix_weights_ones_schedule():
    """Test get_prefix_weights with ONES schedule."""
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.ONES)
    processor = RTCProcessor(config)

    weights = processor.get_prefix_weights(start=5, end=15, total=20)

    # First 15 should be 1.0, rest should be 0.0
    assert weights.shape == (20,)
    assert torch.all(weights[:15] == 1.0)
    assert torch.all(weights[15:] == 0.0)


def test_get_prefix_weights_linear_schedule():
    """Test get_prefix_weights with LINEAR schedule."""
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.LINEAR)
    processor = RTCProcessor(config)

    weights = processor.get_prefix_weights(start=5, end=14, total=25)

    # Should have shape (20,)
    assert weights.shape == (25,)

    # First 5 should be 1.0 (leading ones)
    assert torch.all(weights[:5] == 1.0)

    # Middle section (5:15) should be linearly decreasing from 1 to 0
    middle_weights = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    assert torch.allclose(weights[5:14], middle_weights)

    # Last 5 should be 0.0 (trailing zeros)
    assert torch.all(weights[14:] == 0.0)


def test_get_prefix_weights_exp_schedule():
    """Test get_prefix_weights with EXP schedule."""
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.EXP)
    processor = RTCProcessor(config)

    weights = processor.get_prefix_weights(start=5, end=14, total=25)

    # Should have shape (20,)
    assert weights.shape == (25,)

    # First 5 should be 1.0 (leading ones)
    assert torch.all(weights[:5] == 1.0)

    # Middle section should be exponentially weighted
    middle_weights = torch.tensor([0.7645, 0.5706, 0.4130, 0.2871, 0.1888, 0.1145, 0.0611, 0.0258, 0.0061])
    assert torch.allclose(weights[5:14], middle_weights, atol=1e-4)

    # Last 5 should be 0.0 (trailing zeros)
    assert torch.all(weights[14:] == 0.0)


def test_get_prefix_weights_with_start_equals_end():
    """Test get_prefix_weights when start equals end."""
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.LINEAR)
    processor = RTCProcessor(config)

    weights = processor.get_prefix_weights(start=10, end=10, total=20)

    # Should have ones up to start, then zeros
    assert torch.all(weights[:10] == 1.0)
    assert torch.all(weights[10:] == 0.0)


def test_get_prefix_weights_with_start_greater_than_end():
    """Test get_prefix_weights when start > end (gets clamped)."""
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.LINEAR)
    processor = RTCProcessor(config)

    # start > end should use min(start, end) = end
    weights = processor.get_prefix_weights(start=15, end=10, total=20)

    # Should have ones up to end (10), then zeros
    assert torch.all(weights[:10] == 1.0)
    assert torch.all(weights[10:] == 0.0)


# ====================== Helper Method Tests ======================


def test_linweights_with_end_equals_start():
    """Test _linweights when end equals start."""
    config = RTCConfig()
    processor = RTCProcessor(config)

    weights = processor._linweights(start=10, end=10, total=20)

    # Should return empty tensor
    assert len(weights) == 0


def test_linweights_with_end_less_than_start():
    """Test _linweights when end < start."""
    config = RTCConfig()
    processor = RTCProcessor(config)

    weights = processor._linweights(start=15, end=10, total=20)

    # Should return empty tensor
    assert len(weights) == 0


def test_add_trailing_zeros_normal():
    """Test _add_trailing_zeros adds zeros correctly."""
    config = RTCConfig()
    processor = RTCProcessor(config)

    weights = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])
    result = processor._add_trailing_zeros(weights, total=10, end=5)

    # Should add 5 zeros (total - end = 10 - 5 = 5)
    assert len(result) == 10
    assert torch.all(result[:5] == weights)
    assert torch.all(result[5:] == 0.0)


def test_add_trailing_zeros_no_zeros_needed():
    """Test _add_trailing_zeros when no zeros needed."""
    config = RTCConfig()
    processor = RTCProcessor(config)

    weights = torch.tensor([1.0, 0.8, 0.6])
    result = processor._add_trailing_zeros(weights, total=3, end=5)

    # zeros_len = 3 - 5 = -2 <= 0, so no zeros added
    assert torch.equal(result, weights)


def test_add_leading_ones_normal():
    """Test _add_leading_ones adds ones correctly."""
    config = RTCConfig()
    processor = RTCProcessor(config)

    weights = torch.tensor([0.8, 0.6, 0.4, 0.2, 0.0])
    result = processor._add_leading_ones(weights, start=3, total=10)

    # Should add 3 ones at the start
    assert len(result) == 8
    assert torch.all(result[:3] == 1.0)
    assert torch.all(result[3:] == weights)


def test_add_leading_ones_no_ones_needed():
    """Test _add_leading_ones when no ones needed."""
    config = RTCConfig()
    processor = RTCProcessor(config)

    weights = torch.tensor([0.8, 0.6, 0.4])
    result = processor._add_leading_ones(weights, start=0, total=10)

    # ones_len = 0, so no ones added
    assert torch.equal(result, weights)


def test_get_prefix_weights_with_start_equals_total():
    """Test get_prefix_weights when start equals total."""
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.LINEAR)
    processor = RTCProcessor(config)

    weights = processor.get_prefix_weights(start=10, end=10, total=20)

    # Should have ones up to start, then zeros
    assert len(weights) == 20
    assert torch.all(weights[:10] == 1.0)
    assert torch.all(weights[10:] == 0.0)


def test_get_prefix_weights_with_total_less_than_start():
    """Test get_prefix_weights when total less than start."""
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.LINEAR)
    processor = RTCProcessor(config)

    weights = processor.get_prefix_weights(start=10, end=10, total=5)

    # Should have ones up to start, then zeros
    assert len(weights) == 5
    assert torch.all(weights == 1.0)


# ====================== denoise_step Tests ======================


def test_denoise_step_without_prev_chunk(rtc_processor_debug_disabled):
    """Test denoise_step without previous chunk (no guidance)."""
    x_t = torch.randn(1, 50, 6)

    # Mock denoiser that returns fixed velocity
    def mock_denoiser(x):
        return torch.ones_like(x) * 0.5

    result = rtc_processor_debug_disabled.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=None,
        inference_delay=5,
        time=torch.tensor(0.5),
        original_denoise_step_partial=mock_denoiser,
    )

    # Should return v_t unchanged (no guidance)
    expected = mock_denoiser(x_t)
    assert torch.allclose(result, expected)


def test_denoise_step_with_prev_chunk(rtc_processor_debug_disabled):
    """Test denoise_step with previous chunk applies guidance."""
    x_t = torch.randn(1, 50, 6)
    prev_chunk = torch.randn(1, 50, 6)

    def mock_denoiser(x):
        return torch.ones_like(x) * 0.5

    result = rtc_processor_debug_disabled.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk,
        inference_delay=5,
        time=torch.tensor(0.5),
        original_denoise_step_partial=mock_denoiser,
    )

    # Result should be different from base v_t (guidance applied)
    base_v_t = mock_denoiser(x_t)
    assert not torch.allclose(result, base_v_t)

    # Result should have same shape
    assert result.shape == x_t.shape


def test_denoise_step_adds_batch_dimension():
    """Test denoise_step handles 2D input by adding batch dimension."""
    config = RTCConfig(execution_horizon=10, max_guidance_weight=5.0)
    processor = RTCProcessor(config)

    # 2D input (no batch dimension)
    x_t = torch.randn(50, 6)
    prev_chunk = torch.randn(50, 6)

    def mock_denoiser(x):
        return torch.ones_like(x) * 0.5

    result = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk,
        inference_delay=5,
        time=torch.tensor(0.5),
        original_denoise_step_partial=mock_denoiser,
    )

    # Output should be 2D (batch dimension removed)
    assert result.ndim == 2
    assert result.shape == (50, 6)


def test_denoise_step_pads_shorter_prev_chunk():
    """Test denoise_step pads previous chunk if shorter than x_t."""
    config = RTCConfig(execution_horizon=10, max_guidance_weight=5.0)
    processor = RTCProcessor(config)

    x_t = torch.randn(1, 50, 6)
    prev_chunk = torch.randn(1, 30, 6)  # Shorter than x_t

    def mock_denoiser(x):
        return torch.ones_like(x) * 0.5

    result = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk,
        inference_delay=5,
        time=torch.tensor(0.5),
        original_denoise_step_partial=mock_denoiser,
    )

    # Should complete successfully (padding happens internally)
    assert result.shape == x_t.shape


def test_denoise_step_pads_fewer_action_dims():
    """Test denoise_step pads if prev_chunk has fewer action dimensions."""
    config = RTCConfig(execution_horizon=10, max_guidance_weight=5.0)
    processor = RTCProcessor(config)

    x_t = torch.randn(1, 50, 6)
    prev_chunk = torch.randn(1, 50, 4)  # Fewer action dims

    def mock_denoiser(x):
        return torch.ones_like(x) * 0.5

    result = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk,
        inference_delay=5,
        time=torch.tensor(0.5),
        original_denoise_step_partial=mock_denoiser,
    )

    # Should complete successfully (padding happens internally)
    assert result.shape == x_t.shape


def test_denoise_step_uses_custom_execution_horizon():
    """Test denoise_step uses custom execution_horizon parameter."""
    config = RTCConfig(execution_horizon=10)
    processor = RTCProcessor(config)

    x_t = torch.randn(1, 50, 6)
    prev_chunk = torch.randn(1, 50, 6)

    def mock_denoiser(x):
        return torch.ones_like(x) * 0.5

    # Use custom execution_horizon
    result = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk,
        inference_delay=5,
        time=torch.tensor(0.5),
        original_denoise_step_partial=mock_denoiser,
        execution_horizon=20,  # Override config value
    )

    assert result.shape == x_t.shape


def test_denoise_step_clamps_execution_horizon_to_prev_chunk_length():
    """Test denoise_step clamps execution_horizon if prev_chunk is shorter."""
    config = RTCConfig(execution_horizon=100)  # Very large
    processor = RTCProcessor(config)

    x_t = torch.randn(1, 50, 6)
    prev_chunk = torch.randn(1, 20, 6)  # Only 20 timesteps

    def mock_denoiser(x):
        return torch.ones_like(x) * 0.5

    # Should clamp execution_horizon to 20 internally
    result = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk,
        inference_delay=5,
        time=torch.tensor(0.5),
        original_denoise_step_partial=mock_denoiser,
    )

    assert result.shape == x_t.shape


def test_denoise_step_guidance_weight_calculation():
    """Test denoise_step calculates guidance weight correctly."""
    config = RTCConfig(max_guidance_weight=10.0)
    processor = RTCProcessor(config)

    x_t = torch.randn(1, 50, 6)
    prev_chunk = torch.randn(1, 50, 6)

    def mock_denoiser(x):
        return torch.ones_like(x) * 0.5

    # Time = 0.5 => tau = 1 - 0.5 = 0.5
    time = 0.5
    result = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk,
        inference_delay=5,
        time=time,
        original_denoise_step_partial=mock_denoiser,
    )

    # Should produce valid output
    assert result.shape == x_t.shape
    assert not torch.any(torch.isnan(result))
    assert not torch.any(torch.isinf(result))


def test_denoise_step_guidance_weight_at_time_zero():
    """Test denoise_step handles time=0 (tau=1) without NaN/Inf."""
    config = RTCConfig(max_guidance_weight=10.0)
    processor = RTCProcessor(config)

    x_t = torch.randn(1, 50, 6)
    prev_chunk = torch.randn(1, 50, 6)

    def mock_denoiser(x):
        return torch.ones_like(x) * 0.5

    # Time = 0 => tau = 1, c = (1-tau)/tau = 0/1 = 0
    result = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk,
        inference_delay=5,
        time=torch.tensor(0.0),
        original_denoise_step_partial=mock_denoiser,
    )

    # Should handle gracefully (no NaN/Inf)
    assert not torch.any(torch.isnan(result))
    assert not torch.any(torch.isinf(result))


def test_denoise_step_guidance_weight_at_time_one():
    """Test denoise_step handles time=1 (tau=0) with max_guidance_weight clamping."""
    config = RTCConfig(max_guidance_weight=10.0)
    processor = RTCProcessor(config)

    x_t = torch.randn(1, 50, 6)
    prev_chunk = torch.randn(1, 50, 6)

    def mock_denoiser(x):
        return torch.ones_like(x) * 0.5

    # Time = 1 => tau = 0, c = (1-tau)/tau = 1/0 = inf (clamped to max_guidance_weight)
    result = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk,
        inference_delay=5,
        time=torch.tensor(1.0),
        original_denoise_step_partial=mock_denoiser,
    )

    # Should clamp to max_guidance_weight (no Inf)
    assert not torch.any(torch.isinf(result))


def test_denoise_step_tracks_debug_info(rtc_processor_debug_enabled):
    """Test denoise_step tracks debug information when enabled."""
    x_t = torch.randn(1, 50, 6)
    prev_chunk = torch.randn(1, 50, 6)

    def mock_denoiser(x):
        return torch.ones_like(x) * 0.5

    rtc_processor_debug_enabled.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk,
        inference_delay=5,
        time=torch.tensor(0.5),
        original_denoise_step_partial=mock_denoiser,
    )

    # Should have tracked one step
    steps = rtc_processor_debug_enabled.get_all_debug_steps()
    assert len(steps) == 1

    # Check tracked values
    step = steps[0]
    assert step.time == 0.5
    assert step.x1_t is not None
    assert step.correction is not None
    assert step.err is not None
    assert step.weights is not None
    assert step.guidance_weight is not None
    assert step.inference_delay == 5


def test_denoise_step_doesnt_track_without_debug(rtc_processor_debug_disabled):
    """Test denoise_step doesn't track when debug disabled."""
    x_t = torch.randn(1, 50, 6)
    prev_chunk = torch.randn(1, 50, 6)

    def mock_denoiser(x):
        return torch.ones_like(x) * 0.5

    rtc_processor_debug_disabled.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk,
        inference_delay=5,
        time=torch.tensor(0.5),
        original_denoise_step_partial=mock_denoiser,
    )

    # Should not track
    steps = rtc_processor_debug_disabled.get_all_debug_steps()
    assert len(steps) == 0


# ====================== Integration Tests ======================


def test_denoise_step_full_workflow():
    """Test complete denoise_step workflow."""
    config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
        max_guidance_weight=5.0,
        execution_horizon=10,
        debug=True,
    )
    processor = RTCProcessor(config)

    # Simulate two denoising steps
    x_t1 = torch.randn(1, 50, 6)
    x_t2 = torch.randn(1, 50, 6)

    def mock_denoiser(x):
        return torch.randn_like(x) * 0.1

    # First step - no guidance
    result1 = processor.denoise_step(
        x_t=x_t1,
        prev_chunk_left_over=None,
        inference_delay=5,
        time=torch.tensor(0.8),
        original_denoise_step_partial=mock_denoiser,
    )

    # Second step - with guidance
    result2 = processor.denoise_step(
        x_t=x_t2,
        prev_chunk_left_over=result1,
        inference_delay=5,
        time=torch.tensor(0.6),
        original_denoise_step_partial=mock_denoiser,
    )

    # Both should complete successfully
    assert result1.shape == (1, 50, 6)
    assert result2.shape == (1, 50, 6)

    # Should have tracked one step (second one, first had no prev_chunk)
    steps = processor.get_all_debug_steps()
    assert len(steps) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_denoise_step_with_cuda_tensors():
    """Test denoise_step works with CUDA tensors."""
    config = RTCConfig(execution_horizon=10, max_guidance_weight=5.0)
    processor = RTCProcessor(config)

    x_t = torch.randn(1, 50, 6, device="cuda")
    prev_chunk = torch.randn(1, 50, 6, device="cuda")

    def mock_denoiser(x):
        return torch.ones_like(x) * 0.5

    result = processor.denoise_step(
        x_t=x_t,
        prev_chunk_left_over=prev_chunk,
        inference_delay=5,
        time=torch.tensor(0.5),
        original_denoise_step_partial=mock_denoiser,
    )

    # Result should be on CUDA
    assert result.device.type == "cuda"
    assert result.shape == x_t.shape


def test_denoise_step_deterministic_with_same_inputs():
    """Test denoise_step produces same output with same inputs."""
    config = RTCConfig(execution_horizon=10, max_guidance_weight=5.0)
    processor = RTCProcessor(config)

    torch.manual_seed(42)
    x_t = torch.randn(1, 50, 6)
    prev_chunk = torch.randn(1, 50, 6)

    def deterministic_denoiser(x):
        return torch.ones_like(x) * 0.5

    result1 = processor.denoise_step(
        x_t=x_t.clone(),
        prev_chunk_left_over=prev_chunk.clone(),
        inference_delay=5,
        time=torch.tensor(0.5),
        original_denoise_step_partial=deterministic_denoiser,
    )

    result2 = processor.denoise_step(
        x_t=x_t.clone(),
        prev_chunk_left_over=prev_chunk.clone(),
        inference_delay=5,
        time=torch.tensor(0.5),
        original_denoise_step_partial=deterministic_denoiser,
    )

    # Should produce identical results
    assert torch.allclose(result1, result2)
