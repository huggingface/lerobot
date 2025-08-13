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

"""Tests for RTCProcessor class."""

import pytest
import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


class TestGetPrefixWeights:
    """Comprehensive tests for get_prefix_weights method."""

    def create_config(self, schedule):
        return RTCConfig(prefix_attention_schedule=schedule)

    def create_processor(self, config):
        return RTCProcessor(config)

    @pytest.fixture
    def zero_scheduler_processor(self):
        config = self.create_config(RTCAttentionSchedule.ZEROS)
        return self.create_processor(config)

    @pytest.fixture
    def ones_scheduler_processor(self):
        config = self.create_config(RTCAttentionSchedule.ONES)
        return self.create_processor(config)

    @pytest.fixture
    def linear_scheduler_processor(self):
        config = self.create_config(RTCAttentionSchedule.LINEAR)
        return self.create_processor(config)

    @pytest.fixture
    def exp_scheduler_processor(self):
        config = self.create_config(RTCAttentionSchedule.EXP)
        return self.create_processor(config)

    def test_zeros_schedule(self, zero_scheduler_processor):
        """Test ZEROS attention schedule."""

        # Test case 1: Normal case
        weights = zero_scheduler_processor.get_prefix_weights(start=3, end=7, total=10)
        expected = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Total = 0
        weights = zero_scheduler_processor.get_prefix_weights(start=3, end=7, total=0)
        expected = torch.tensor([])
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # # Test case 2: Start >= end
        weights = zero_scheduler_processor.get_prefix_weights(start=8, end=5, total=10)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # # Test case 3: Start = 0
        weights = zero_scheduler_processor.get_prefix_weights(start=0, end=5, total=10)
        expected = torch.zeros(10)
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # End == total
        weights = zero_scheduler_processor.get_prefix_weights(start=0, end=10, total=10)
        expected = torch.zeros(10)
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Start > total
        weights = zero_scheduler_processor.get_prefix_weights(start=10, end=0, total=10)
        expected = torch.zeros(10)
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Start = end
        weights = zero_scheduler_processor.get_prefix_weights(start=5, end=5, total=10)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # End > total
        weights = zero_scheduler_processor.get_prefix_weights(start=0, end=11, total=10)
        expected = torch.zeros(10)
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

    def test_ones_schedule(self, ones_scheduler_processor):
        """Test ONES attention schedule."""

        # Test case 1: Normal case
        weights = ones_scheduler_processor.get_prefix_weights(start=3, end=7, total=10)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        assert weights.shape == expected.shape

        torch.testing.assert_close(weights, expected)

        # Total = 0
        weights = ones_scheduler_processor.get_prefix_weights(start=3, end=7, total=0)
        expected = torch.tensor([])
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

    def test_linear_schedule(self, linear_scheduler_processor):
        """Test LINEAR attention schedule."""

        # Test case 1: Normal case
        weights = linear_scheduler_processor.get_prefix_weights(start=3, end=7, total=10)
        expected = torch.tensor([1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0, 0, 0])
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Total = 0
        weights = linear_scheduler_processor.get_prefix_weights(start=3, end=7, total=0)
        expected = torch.tensor([])
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 2: Start is equal 0
        weights = linear_scheduler_processor.get_prefix_weights(start=0, end=7, total=10)
        expected = torch.tensor([0.8750, 0.7500, 0.6250, 0.5000, 0.3750, 0.2500, 0.1250, 0, 0, 0])
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 3: End is equal to total
        weights = linear_scheduler_processor.get_prefix_weights(start=3, end=10, total=10)
        expected = torch.tensor(
            [1.0000, 1.0000, 1.0000, 0.8750, 0.7500, 0.6250, 0.5000, 0.3750, 0.2500, 0.1250]
        )
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 4: Start > total
        weights = linear_scheduler_processor.get_prefix_weights(start=10, end=3, total=10)
        expected = torch.tensor(
            [1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        )
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 5: Start = end
        weights = linear_scheduler_processor.get_prefix_weights(start=3, end=3, total=10)
        expected = torch.tensor(
            [1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        )
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 6: Start > total
        weights = linear_scheduler_processor.get_prefix_weights(start=10, end=3, total=10)
        expected = torch.tensor(
            [1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        )
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 7: End > total
        weights = linear_scheduler_processor.get_prefix_weights(start=3, end=11, total=10)
        expected = torch.tensor(
            [1.0000, 1.0000, 1.0000, 0.8750, 0.7500, 0.6250, 0.5000, 0.3750, 0.2500, 0.1250]
        )
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 8: Total = 1
        weights = linear_scheduler_processor.get_prefix_weights(start=3, end=11, total=10)
        expected = torch.tensor(
            [1.0000, 1.0000, 1.0000, 0.8750, 0.7500, 0.6250, 0.5000, 0.3750, 0.2500, 0.1250]
        )
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 9: Total = 2
        weights = linear_scheduler_processor.get_prefix_weights(start=3, end=11, total=2)
        expected = torch.tensor([1.0000, 1.0000])
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 10: Total = 5
        weights = linear_scheduler_processor.get_prefix_weights(start=1, end=4, total=5)
        expected = torch.tensor([1.0000, 0.750, 0.500, 0.2500, 0.0000])
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

    def test_exp_schedule(self, exp_scheduler_processor):
        """Test EXP attention schedule."""

        # Test case 1: Normal case
        weights = exp_scheduler_processor.get_prefix_weights(start=3, end=7, total=10)
        expected = torch.tensor(
            [1.0000, 1.0000, 1.0000, 0.57058895, 0.28707242, 0.11449217, 0.02577024, 0.0000, 0.0000, 0.0000]
        )
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Total = 0
        weights = exp_scheduler_processor.get_prefix_weights(start=3, end=7, total=0)
        expected = torch.tensor([])
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 2: Start is equal 0
        weights = exp_scheduler_processor.get_prefix_weights(start=0, end=7, total=10)
        expected = torch.tensor(
            [
                0.7123487,
                0.48755103,
                0.31581184,
                0.18877034,
                0.09929788,
                0.04132404,
                0.00968616,
                0.0000,
                0.0000,
                0.0000,
            ]
        )
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 3: End is equal to total
        weights = exp_scheduler_processor.get_prefix_weights(start=3, end=10, total=10)
        expected = torch.tensor(
            [
                1.0000,
                1.0000,
                1.0000,
                0.7123487,
                0.48755103,
                0.31581184,
                0.18877034,
                0.09929788,
                0.04132404,
                0.00968616,
            ]
        )
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 4: Start > end
        weights = exp_scheduler_processor.get_prefix_weights(start=10, end=3, total=10)
        expected = torch.tensor(
            [1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        )
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 5: Start = end
        weights = exp_scheduler_processor.get_prefix_weights(start=3, end=3, total=10)
        expected = torch.tensor(
            [1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        )
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 6: Start > total
        weights = exp_scheduler_processor.get_prefix_weights(start=10, end=3, total=10)
        expected = torch.tensor(
            [1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        )
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 7: End > total
        weights = exp_scheduler_processor.get_prefix_weights(start=3, end=11, total=10)
        expected = torch.tensor(
            [
                1.0000,
                1.0000,
                1.0000,
                0.7123487,
                0.48755103,
                0.31581184,
                0.18877034,
                0.09929788,
                0.04132404,
                0.00968616,
            ]
        )
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 8: Total = 1
        weights = exp_scheduler_processor.get_prefix_weights(start=0, end=1, total=1)
        expected = torch.tensor([0.18877034])
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)

        # Test case 9: Total = 2
        weights = exp_scheduler_processor.get_prefix_weights(start=3, end=11, total=2)
        expected = torch.tensor([1.0000, 1.0000])
        assert weights.shape == expected.shape
        torch.testing.assert_close(weights, expected)


class TestDenoiseStep:
    """Comprehensive tests for denoise_step method."""

    def create_config(self, **kwargs):
        """Create RTCConfig with custom parameters."""
        default_params = {
            "max_guidance_weight": torch.tensor(10.0),
            "prefix_attention_schedule": RTCAttentionSchedule.LINEAR,
        }
        default_params.update(kwargs)
        return RTCConfig(**default_params)

    def create_processor(self, config):
        return RTCProcessor(config)

    @pytest.fixture
    def processor(self):
        """Create default processor for testing."""
        config = self.create_config()
        return self.create_processor(config)

    def test_no_previous_chunk(self, processor):
        """Test first step when prev_chunk is None (no guidance applied)."""
        batch_size = 2
        action_chunk_size = 10
        action_dim = 3
        inference_delay = 3

        noise = torch.randn(batch_size, action_chunk_size, action_dim)
        v_t = torch.randn(batch_size, action_chunk_size, action_dim)
        time = torch.tensor(0.5)

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=None,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        # When prev_chunk is None, should return v_t unchanged
        torch.testing.assert_close(result, v_t)

    def test_previous_chunk_present(self, processor):
        """Test when prev_chunk is present."""
        batch_size = 2
        action_chunk_size = 10
        action_dim = 3

        inference_delay = 3

        noise = torch.full((batch_size, action_chunk_size, action_dim), 0.1)
        prev_chunk = torch.ones(batch_size, action_chunk_size, action_dim)
        v_t = torch.full((batch_size, action_chunk_size, action_dim), 3.0)
        time = torch.tensor(0.5)

        expected_result = torch.tensor(
            [
                [
                    [1.8000, 1.8000, 1.8000],
                    [1.8000, 1.8000, 1.8000],
                    [1.8000, 1.8000, 1.8000],
                    [1.9500, 1.9500, 1.9500],
                    [2.1000, 2.1000, 2.1000],
                    [2.2500, 2.2500, 2.2500],
                    [2.4000, 2.4000, 2.4000],
                    [2.5500, 2.5500, 2.5500],
                    [2.7000, 2.7000, 2.7000],
                    [2.8500, 2.8500, 2.8500],
                ],
                [
                    [1.8000, 1.8000, 1.8000],
                    [1.8000, 1.8000, 1.8000],
                    [1.8000, 1.8000, 1.8000],
                    [1.9500, 1.9500, 1.9500],
                    [2.1000, 2.1000, 2.1000],
                    [2.2500, 2.2500, 2.2500],
                    [2.4000, 2.4000, 2.4000],
                    [2.5500, 2.5500, 2.5500],
                    [2.7000, 2.7000, 2.7000],
                    [2.8500, 2.8500, 2.8500],
                ],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        assert torch.allclose(result, expected_result)

    def test_with_without_batch_dimension(self, processor):
        """Test denoise step with without batch dimension."""
        action_chunk_size = 10
        action_dim = 3
        inference_delay = 3

        noise = torch.full((action_chunk_size, action_dim), 0.1)
        prev_chunk = torch.ones(action_chunk_size, action_dim)
        v_t = torch.full((action_chunk_size, action_dim), 3.0)
        time = torch.tensor(0.5)

        expected_result = torch.tensor(
            [
                [1.8000, 1.8000, 1.8000],
                [1.8000, 1.8000, 1.8000],
                [1.8000, 1.8000, 1.8000],
                [1.9500, 1.9500, 1.9500],
                [2.1000, 2.1000, 2.1000],
                [2.2500, 2.2500, 2.2500],
                [2.4000, 2.4000, 2.4000],
                [2.5500, 2.5500, 2.5500],
                [2.7000, 2.7000, 2.7000],
                [2.8500, 2.8500, 2.8500],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        assert torch.allclose(result, expected_result)

    def test_prev_chunk_smaller_than_action_chunk_size(self, processor):
        """Test denoise step with prev_chunk smaller than action chunk size."""
        action_chunk_size = 10
        action_dim = 1
        inference_delay = 3

        prev_chunk = torch.ones(2, action_dim)
        noise = torch.full((action_chunk_size, action_dim), 0.1)
        v_t = torch.full((action_chunk_size, action_dim), 3.0)
        time = torch.tensor(0.5)

        expected_result = torch.tensor(
            [
                [1.8000],
                [1.8000],
                [-0.2000],
                [0.2000],
                [0.6000],
                [1.0000],
                [1.4000],
                [1.8000],
                [2.2000],
                [2.6000],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        assert torch.allclose(result, expected_result)

    def test_with_execution_horizon(self, processor):
        """Test denoise step with execution horizon."""
        action_chunk_size = 15
        action_dim = 1
        execution_horizon = 11
        inference_delay = 3

        noise = torch.full((action_chunk_size, action_dim), 0.1)
        prev_chunk = torch.ones(7, action_dim)
        v_t = torch.full((action_chunk_size, action_dim), 3.0)
        time = torch.tensor(0.5)

        expected_result = torch.tensor(
            [
                [1.8000],
                [1.8000],
                [1.8000],
                [1.9333],
                [2.0667],
                [2.2000],
                [2.3333],
                [1.5778],
                [1.9333],
                [2.2889],
                [2.6444],
                [3.0000],
                [3.0000],
                [3.0000],
                [3.0000],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
            execution_horizon=execution_horizon,
        )

        assert torch.allclose(result, expected_result, atol=1e-4)

    def test_when_execution_horizon_is_in_config(self, processor):
        """Test denoise step when execution horizon is in config."""
        action_chunk_size = 15
        action_dim = 1
        execution_horizon = 11
        inference_delay = 3

        config = self.create_config(execution_horizon=execution_horizon)
        processor = self.create_processor(config)

        noise = torch.full((action_chunk_size, action_dim), 0.1)
        prev_chunk = torch.ones(7, action_dim)
        v_t = torch.full((action_chunk_size, action_dim), 3.0)
        time = torch.tensor(0.5)

        expected_result = torch.tensor(
            [
                [1.8000],
                [1.8000],
                [1.8000],
                [1.9333],
                [2.0667],
                [2.2000],
                [2.3333],
                [1.5778],
                [1.9333],
                [2.2889],
                [2.6444],
                [3.0000],
                [3.0000],
                [3.0000],
                [3.0000],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        assert torch.allclose(result, expected_result, atol=1e-4)

    def test_realistic_case(self):
        """Test denoise step with realistic case."""
        action_chunk_size = 50
        action_dim = 6
        inference_delay = 5  # 5 * 20ms = 100 ms
        execution_horizon = 40

        config = self.create_config(execution_horizon=execution_horizon)
        processor = self.create_processor(config)

        noise = torch.ones(action_chunk_size, action_dim)
        v_t = torch.linspace(0, 1, action_chunk_size).unsqueeze(1).repeat(1, action_dim)
        time = torch.tensor(0.9)
        prev_chunk = torch.full((20, action_dim), 1.5)

        expected_result = torch.tensor(
            [
                [4.5556, 4.5556, 4.5556, 4.5556, 4.5556, 4.5556],
                [4.4086, 4.4086, 4.4086, 4.4086, 4.4086, 4.4086],
                [4.2617, 4.2617, 4.2617, 4.2617, 4.2617, 4.2617],
                [4.1147, 4.1147, 4.1147, 4.1147, 4.1147, 4.1147],
                [3.9678, 3.9678, 3.9678, 3.9678, 3.9678, 3.9678],
                [3.7176, 3.7176, 3.7176, 3.7176, 3.7176, 3.7176],
                [3.4766, 3.4766, 3.4766, 3.4766, 3.4766, 3.4766],
                [3.2450, 3.2450, 3.2450, 3.2450, 3.2450, 3.2450],
                [3.0226, 3.0226, 3.0226, 3.0226, 3.0226, 3.0226],
                [2.8096, 2.8096, 2.8096, 2.8096, 2.8096, 2.8096],
                [2.6058, 2.6058, 2.6058, 2.6058, 2.6058, 2.6058],
                [2.4114, 2.4114, 2.4114, 2.4114, 2.4114, 2.4114],
                [2.2262, 2.2262, 2.2262, 2.2262, 2.2262, 2.2262],
                [2.0503, 2.0503, 2.0503, 2.0503, 2.0503, 2.0503],
                [1.8838, 1.8838, 1.8838, 1.8838, 1.8838, 1.8838],
                [1.7265, 1.7265, 1.7265, 1.7265, 1.7265, 1.7265],
                [1.5785, 1.5785, 1.5785, 1.5785, 1.5785, 1.5785],
                [1.4399, 1.4399, 1.4399, 1.4399, 1.4399, 1.4399],
                [1.3105, 1.3105, 1.3105, 1.3105, 1.3105, 1.3105],
                [1.1904, 1.1904, 1.1904, 1.1904, 1.1904, 1.1904],
                [-6.5130, -6.5130, -6.5130, -6.5130, -6.5130, -6.5130],
                [-6.2348, -6.2348, -6.2348, -6.2348, -6.2348, -6.2348],
                [-5.9474, -5.9474, -5.9474, -5.9474, -5.9474, -5.9474],
                [-5.6507, -5.6507, -5.6507, -5.6507, -5.6507, -5.6507],
                [-5.3446, -5.3446, -5.3446, -5.3446, -5.3446, -5.3446],
                [-5.0293, -5.0293, -5.0293, -5.0293, -5.0293, -5.0293],
                [-4.7047, -4.7047, -4.7047, -4.7047, -4.7047, -4.7047],
                [-4.3707, -4.3707, -4.3707, -4.3707, -4.3707, -4.3707],
                [-4.0275, -4.0275, -4.0275, -4.0275, -4.0275, -4.0275],
                [-3.6750, -3.6750, -3.6750, -3.6750, -3.6750, -3.6750],
                [-3.3132, -3.3132, -3.3132, -3.3132, -3.3132, -3.3132],
                [-2.9421, -2.9421, -2.9421, -2.9421, -2.9421, -2.9421],
                [-2.5617, -2.5617, -2.5617, -2.5617, -2.5617, -2.5617],
                [-2.1719, -2.1719, -2.1719, -2.1719, -2.1719, -2.1719],
                [-1.7729, -1.7729, -1.7729, -1.7729, -1.7729, -1.7729],
                [-1.3646, -1.3646, -1.3646, -1.3646, -1.3646, -1.3646],
                [-0.9470, -0.9470, -0.9470, -0.9470, -0.9470, -0.9470],
                [-0.5201, -0.5201, -0.5201, -0.5201, -0.5201, -0.5201],
                [-0.0840, -0.0840, -0.0840, -0.0840, -0.0840, -0.0840],
                [0.3615, 0.3615, 0.3615, 0.3615, 0.3615, 0.3615],
                [0.8163, 0.8163, 0.8163, 0.8163, 0.8163, 0.8163],
                [0.8367, 0.8367, 0.8367, 0.8367, 0.8367, 0.8367],
                [0.8571, 0.8571, 0.8571, 0.8571, 0.8571, 0.8571],
                [0.8776, 0.8776, 0.8776, 0.8776, 0.8776, 0.8776],
                [0.8980, 0.8980, 0.8980, 0.8980, 0.8980, 0.8980],
                [0.9184, 0.9184, 0.9184, 0.9184, 0.9184, 0.9184],
                [0.9388, 0.9388, 0.9388, 0.9388, 0.9388, 0.9388],
                [0.9592, 0.9592, 0.9592, 0.9592, 0.9592, 0.9592],
                [0.9796, 0.9796, 0.9796, 0.9796, 0.9796, 0.9796],
                [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        assert torch.allclose(result, expected_result, atol=1e-4)

    def test_with_time_0_1(self, processor):
        """Test denoise step with time 0.1."""
        action_chunk_size = 10
        action_dim = 1
        inference_delay = 3

        noise = torch.full((action_chunk_size, action_dim), 0.1)
        prev_chunk = torch.ones(action_chunk_size, action_dim)
        v_t = torch.full((action_chunk_size, action_dim), 3.0)
        time = torch.tensor(0.1)

        expected_result = torch.tensor(
            [
                [8.4667],
                [8.4667],
                [8.4667],
                [7.7833],
                [7.1000],
                [6.4167],
                [5.7333],
                [5.0500],
                [4.3667],
                [3.6833],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        assert torch.allclose(result, expected_result)

    def test_with_time_1(self, processor):
        """Test denoise step with time 1."""
        action_chunk_size = 10
        action_dim = 1
        inference_delay = 3

        noise = torch.full((action_chunk_size, action_dim), 0.1)
        prev_chunk = torch.ones(action_chunk_size, action_dim)
        v_t = torch.full((action_chunk_size, action_dim), 3.0)
        time = torch.tensor(1.0)

        expected_result = torch.tensor(
            [
                [-18.0000],
                [-18.0000],
                [-18.0000],
                [-15.3750],
                [-12.7500],
                [-10.1250],
                [-7.5000],
                [-4.8750],
                [-2.2500],
                [0.3750],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        assert torch.allclose(result, expected_result)

    def test_time_0_0(self, processor):
        """Test denoise step with time 0.0."""
        action_chunk_size = 10
        action_dim = 1
        inference_delay = 3

        noise = torch.full((action_chunk_size, action_dim), 0.1)
        prev_chunk = torch.ones(action_chunk_size, action_dim)
        v_t = torch.full((action_chunk_size, action_dim), 3.0)
        time = torch.tensor(0.0)

        expected_result = torch.tensor(
            [
                [3.0000],
                [3.0000],
                [3.0000],
                [3.0000],
                [3.0000],
                [3.0000],
                [3.0000],
                [3.0000],
                [3.0000],
                [3.0000],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        assert torch.allclose(result, expected_result, atol=1e-4)

    def test_with_zero_attention_weight(self):
        """Test denoise step with zero attention weight."""

        config = self.create_config()
        config.prefix_attention_schedule = RTCAttentionSchedule.ZEROS
        processor = self.create_processor(config)

        action_chunk_size = 10
        action_dim = 1
        inference_delay = 3

        noise = torch.full((action_chunk_size, action_dim), 0.1)
        prev_chunk = torch.ones(action_chunk_size, action_dim)
        v_t = torch.full((action_chunk_size, action_dim), 3.0)
        time = torch.tensor(0.5)

        expected_result = torch.tensor(
            [
                [1.8000],
                [1.8000],
                [1.8000],
                [3.0000],
                [3.0000],
                [3.0000],
                [3.0000],
                [3.0000],
                [3.0000],
                [3.0000],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        assert torch.allclose(result, expected_result, atol=1e-4)

    def test_with_ones_attention_weight(self):
        """Test denoise step with ones attention weight."""
        config = self.create_config()
        config.prefix_attention_schedule = RTCAttentionSchedule.ONES
        processor = self.create_processor(config)

        action_chunk_size = 10
        action_dim = 1
        inference_delay = 3

        noise = torch.full((action_chunk_size, action_dim), 0.1)
        prev_chunk = torch.ones(action_chunk_size, action_dim)
        v_t = torch.full((action_chunk_size, action_dim), 3.0)
        time = torch.tensor(0.5)

        expected_result = torch.tensor(
            [
                [1.8000],
                [1.8000],
                [1.8000],
                [1.8000],
                [1.8000],
                [1.8000],
                [1.8000],
                [1.8000],
                [1.8000],
                [1.8000],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        assert torch.allclose(result, expected_result)

    def test_with_linear_attention_weight(self):
        """Test denoise step with linear attention weight."""
        config = self.create_config()
        config.prefix_attention_schedule = RTCAttentionSchedule.LINEAR
        processor = self.create_processor(config)

        action_chunk_size = 10
        action_dim = 1
        inference_delay = 3

        noise = torch.full((action_chunk_size, action_dim), 0.1)
        prev_chunk = torch.ones(action_chunk_size, action_dim)
        v_t = torch.full((action_chunk_size, action_dim), 3.0)
        time = torch.tensor(0.5)

        expected_result = torch.tensor(
            [
                [1.8000],
                [1.8000],
                [1.8000],
                [1.9500],
                [2.1000],
                [2.2500],
                [2.4000],
                [2.5500],
                [2.7000],
                [2.8500],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        assert torch.allclose(result, expected_result)

    def test_zero_noise_input(self, processor):
        """Test denoising with zero noise input."""
        action_chunk_size = 10
        action_dim = 1
        inference_delay = 3

        noise = torch.zeros(action_chunk_size, action_dim)
        prev_chunk = torch.ones(action_chunk_size, action_dim)
        v_t = torch.full((action_chunk_size, action_dim), 3.0)
        time = torch.tensor(0.5)

        expected_result = torch.tensor(
            [
                [2.0000],
                [2.0000],
                [2.0000],
                [2.1250],
                [2.2500],
                [2.3750],
                [2.5000],
                [2.6250],
                [2.7500],
                [2.8750],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        assert torch.allclose(result, expected_result, atol=1e-4)

    def test_delay_bigger_than_horizon(self):
        """Test when inference delay is greater than prefix attention horizon."""
        action_chunk_size = 10
        action_dim = 1
        inference_delay = 8
        execution_horizon = 7

        config = self.create_config(execution_horizon=execution_horizon)
        processor = self.create_processor(config)

        noise = torch.full((action_chunk_size, action_dim), 0.1)
        prev_chunk = torch.ones(action_chunk_size, action_dim)
        v_t = torch.full((action_chunk_size, action_dim), 3.0)
        time = torch.tensor(0.5)

        expected_result = torch.tensor(
            [
                [1.8000],
                [1.8000],
                [1.8000],
                [1.8000],
                [1.8000],
                [1.8000],
                [1.8000],
                [3.0000],
                [3.0000],
                [3.0000],
            ]
        )

        original_denoise_step_partial = lambda x: v_t  # noqa: E731
        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=original_denoise_step_partial,
        )

        assert torch.allclose(result, expected_result, atol=1e-4)

    def test_custom_max_guidance_weight(self):
        """Test with custom max_guidance_weight values."""
        config = self.create_config(max_guidance_weight=0.5)
        processor = self.create_processor(config)

        action_chunk_size = 10
        action_dim = 1
        inference_delay = 3

        noise = torch.full((action_chunk_size, action_dim), 0.1)
        prev_chunk = torch.ones(action_chunk_size, action_dim)
        v_t = torch.full((action_chunk_size, action_dim), 3.0)
        time = torch.tensor(1.0)

        expected_result = torch.tensor(
            [
                [1.9500],
                [1.9500],
                [1.9500],
                [2.0813],
                [2.2125],
                [2.3438],
                [2.4750],
                [2.6063],
                [2.7375],
                [2.8688],
            ]
        )

        result = processor.denoise_step(
            x_t=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            time=time,
            original_denoise_step_partial=lambda x: v_t,
        )

        assert torch.allclose(result, expected_result, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
