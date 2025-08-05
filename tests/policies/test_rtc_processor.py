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
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.rtc_config import RTCConfig


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
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
