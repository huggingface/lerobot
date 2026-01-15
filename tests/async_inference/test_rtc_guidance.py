# Copyright 2026 The HuggingFace Inc. team.
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

"""Tests for AsyncRTCProcessor in async inference.

These tests verify that RTC guidance works correctly when:
1. No postprocess is used (raw model space)
2. Postprocess preserves dimensions
3. Postprocess changes dimensions (e.g., 32 -> 6)
"""

from __future__ import annotations

import pytest
import torch

from lerobot.async_inference.rtc_guidance import AsyncRTCConfig, AsyncRTCProcessor


class TestAsyncRTCProcessor:
    """Tests for AsyncRTCProcessor.denoise_step with various postprocess configurations."""

    def test_denoise_step_without_postprocess(self) -> None:
        """Test RTC guidance without any postprocess (raw model space)."""
        cfg = AsyncRTCConfig(enabled=True)
        rtc = AsyncRTCProcessor(cfg, postprocess=None)

        # Both x_t and prev are in the same action space (32 dims)
        x_t = torch.randn(1, 50, 32)
        prev = torch.randn(1, 10, 32)

        def mock_denoise(x: torch.Tensor) -> torch.Tensor:
            return torch.randn_like(x)

        result = rtc.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=prev,
            inference_delay=5,
            time=0.5,
            original_denoise_step_partial=mock_denoise,
            overlap_end=10,  # H - s where s = d = 5, so overlap at 50 - 5 = 45, but we test with explicit value
        )

        assert result.shape == x_t.shape

    def test_denoise_step_with_dimension_preserving_postprocess(self) -> None:
        """Test RTC guidance with a postprocess that preserves dimensions."""
        cfg = AsyncRTCConfig(enabled=True)

        # Postprocess that keeps the same dimension (e.g., unnormalization)
        def postprocess_identity(x_bta: torch.Tensor) -> torch.Tensor:
            return x_bta * 2.0  # Just scales, preserves shape

        rtc = AsyncRTCProcessor(cfg, postprocess=postprocess_identity)

        x_t = torch.randn(1, 50, 32)
        prev = torch.randn(1, 10, 32)

        def mock_denoise(x: torch.Tensor) -> torch.Tensor:
            return torch.randn_like(x)

        result = rtc.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=prev,
            inference_delay=5,
            time=0.5,
            original_denoise_step_partial=mock_denoise,
            overlap_end=10,
        )

        assert result.shape == x_t.shape

    def test_denoise_step_with_dimension_changing_postprocess(self) -> None:
        """Test RTC guidance when postprocess changes action dimension (32 -> 6).

        This is the key test case that was failing. The client sends frozen actions
        in executable action space (6 dims), while the model operates in raw action
        space (32 dims). The postprocess converts model output to executable space.
        """
        cfg = AsyncRTCConfig(enabled=True)

        # Postprocess that changes dimensions: 32 -> 6
        # This simulates what happens in the real system where model output
        # (32 dims) is transformed to robot action space (6 dims)
        def postprocess_32_to_6(x_bta: torch.Tensor) -> torch.Tensor:
            b, t, a_in = x_bta.shape
            assert a_in == 32, f"Expected 32 dims, got {a_in}"
            # Simulate reduction to 6 dims (e.g., taking first 6 or linear projection)
            return x_bta[:, :, :6]

        rtc = AsyncRTCProcessor(cfg, postprocess=postprocess_32_to_6)

        # x_t is in raw model space (32 dims)
        x_t = torch.randn(1, 50, 32)
        # prev is in executable action space (6 dims) - what the client sends
        prev = torch.randn(1, 10, 6)

        def mock_denoise(x: torch.Tensor) -> torch.Tensor:
            return torch.randn_like(x)

        # This should NOT raise "size of tensor a (32) must match size of tensor b (6)"
        result = rtc.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=prev,
            inference_delay=5,
            time=0.5,
            original_denoise_step_partial=mock_denoise,
            overlap_end=10,
        )

        # Output should match input shape (still in raw model space)
        assert result.shape == x_t.shape

    def test_denoise_step_with_prev_longer_than_chunk(self) -> None:
        """Test when prev chunk is longer than the model's chunk size."""
        cfg = AsyncRTCConfig(enabled=True)

        def postprocess_32_to_6(x_bta: torch.Tensor) -> torch.Tensor:
            return x_bta[:, :, :6]

        rtc = AsyncRTCProcessor(cfg, postprocess=postprocess_32_to_6)

        x_t = torch.randn(1, 50, 32)
        # prev is longer than x_t's temporal dimension
        prev = torch.randn(1, 100, 6)

        def mock_denoise(x: torch.Tensor) -> torch.Tensor:
            return torch.randn_like(x)

        result = rtc.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=prev,
            inference_delay=5,
            time=0.5,
            original_denoise_step_partial=mock_denoise,
            overlap_end=10,
        )

        assert result.shape == x_t.shape

    def test_denoise_step_disabled(self) -> None:
        """Test that RTC is bypassed when disabled."""
        cfg = AsyncRTCConfig(enabled=False)
        rtc = AsyncRTCProcessor(cfg, postprocess=None)

        x_t = torch.randn(1, 50, 32)
        prev = torch.randn(1, 10, 6)

        call_count = 0

        def mock_denoise(x: torch.Tensor) -> torch.Tensor:
            nonlocal call_count
            call_count += 1
            return torch.randn_like(x)

        result = rtc.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=prev,
            inference_delay=5,
            time=0.5,
            original_denoise_step_partial=mock_denoise,
            overlap_end=10,
        )

        # Should call denoise once and return directly
        assert call_count == 1
        assert result.shape == x_t.shape

    def test_denoise_step_no_prev_chunk(self) -> None:
        """Test that RTC is bypassed when no prev_chunk_left_over is provided."""
        cfg = AsyncRTCConfig(enabled=True)
        rtc = AsyncRTCProcessor(cfg, postprocess=None)

        x_t = torch.randn(1, 50, 32)

        def mock_denoise(x: torch.Tensor) -> torch.Tensor:
            return torch.randn_like(x)

        result = rtc.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=None,
            inference_delay=5,
            time=0.5,
            original_denoise_step_partial=mock_denoise,
            overlap_end=10,
        )

        assert result.shape == x_t.shape

    def test_denoise_step_2d_input(self) -> None:
        """Test that 2D input (without batch dim) is handled correctly."""
        cfg = AsyncRTCConfig(enabled=True)

        def postprocess_32_to_6(x_bta: torch.Tensor) -> torch.Tensor:
            return x_bta[:, :, :6]

        rtc = AsyncRTCProcessor(cfg, postprocess=postprocess_32_to_6)

        # 2D input (no batch dimension)
        x_t = torch.randn(50, 32)
        prev = torch.randn(10, 6)

        def mock_denoise(x: torch.Tensor) -> torch.Tensor:
            return torch.randn_like(x)

        result = rtc.denoise_step(
            x_t=x_t,
            prev_chunk_left_over=prev,
            inference_delay=5,
            time=0.5,
            original_denoise_step_partial=mock_denoise,
            overlap_end=10,
        )

        # Should squeeze back to 2D
        assert result.shape == x_t.shape


class TestPrefixWeights:
    """Tests for the prefix weight calculation."""

    def test_linear_schedule(self) -> None:
        """Test linear weight schedule."""
        cfg = AsyncRTCConfig(enabled=True, prefix_attention_schedule="linear")
        rtc = AsyncRTCProcessor(cfg)

        weights = rtc._get_prefix_weights(start=5, end=15, total=50)
        assert weights.shape == (50,)
        # First 5 should be 1.0
        assert torch.allclose(weights[:5], torch.ones(5))
        # After 15 should be 0.0
        assert torch.allclose(weights[15:], torch.zeros(35))

    def test_zeros_schedule(self) -> None:
        """Test zeros weight schedule."""
        cfg = AsyncRTCConfig(enabled=True, prefix_attention_schedule="zeros")
        rtc = AsyncRTCProcessor(cfg)

        weights = rtc._get_prefix_weights(start=5, end=15, total=50)
        assert weights.shape == (50,)
        # First 5 should be 1.0, rest 0.0
        assert torch.allclose(weights[:5], torch.ones(5))
        assert torch.allclose(weights[5:], torch.zeros(45))

    def test_ones_schedule(self) -> None:
        """Test ones weight schedule."""
        cfg = AsyncRTCConfig(enabled=True, prefix_attention_schedule="ones")
        rtc = AsyncRTCProcessor(cfg)

        weights = rtc._get_prefix_weights(start=5, end=15, total=50)
        assert weights.shape == (50,)
        # First 15 should be 1.0, rest 0.0
        assert torch.allclose(weights[:15], torch.ones(15))
        assert torch.allclose(weights[15:], torch.zeros(35))
