#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import torch
from unittest.mock import MagicMock, patch

from lerobot.common.memory_utils import (
    estimate_batch_size_from_memory,
    get_device_memory_stats,
)


class TestGetDeviceMemoryStats:
    def test_cuda_not_available(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                get_device_memory_stats()

    def test_device_index_out_of_range(self):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with pytest.raises(RuntimeError, match="Device 5 does not exist"):
                    get_device_memory_stats(device_idx=5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_returns_correct_structure(self):
        result = get_device_memory_stats()
        assert isinstance(result, dict)
        assert "free" in result
        assert "total" in result
        assert "reserved" in result
        assert all(isinstance(v, int) for v in result.values())
        assert result["total"] > 0
        assert result["free"] >= 0
        assert result["reserved"] >= 0


class TestEstimateBatchSizeFromMemory:
    def test_basic_estimation(self):
        peak_memory = 1_000_000_000
        available_memory = 10_000_000_000
        batch_size = estimate_batch_size_from_memory(
            peak_memory_per_sample=peak_memory,
            available_memory=available_memory,
            safety_margin=0.9,
            min_size=1,
            max_size=256,
            is_main_process=False,
        )
        expected = int((available_memory * 0.9) // peak_memory)
        assert batch_size == expected
        assert batch_size >= 1

    def test_respects_min_size(self):
        peak_memory = 100_000_000_000
        available_memory = 1_000_000_000
        batch_size = estimate_batch_size_from_memory(
            peak_memory_per_sample=peak_memory,
            available_memory=available_memory,
            safety_margin=0.9,
            min_size=4,
            max_size=256,
            is_main_process=False,
        )
        assert batch_size >= 4

    def test_respects_max_size(self):
        peak_memory = 100_000_000
        available_memory = 100_000_000_000
        batch_size = estimate_batch_size_from_memory(
            peak_memory_per_sample=peak_memory,
            available_memory=available_memory,
            safety_margin=0.9,
            min_size=1,
            max_size=64,
            is_main_process=False,
        )
        assert batch_size <= 64

    def test_safety_margin_applied(self):
        peak_memory = 1_000_000
        available_memory = 10_000_000
        batch_size_90 = estimate_batch_size_from_memory(
            peak_memory_per_sample=peak_memory,
            available_memory=available_memory,
            safety_margin=0.9,
            min_size=1,
            max_size=256,
            is_main_process=False,
        )
        batch_size_50 = estimate_batch_size_from_memory(
            peak_memory_per_sample=peak_memory,
            available_memory=available_memory,
            safety_margin=0.5,
            min_size=1,
            max_size=256,
            is_main_process=False,
        )
        assert batch_size_90 > batch_size_50

    def test_zero_peak_memory_raises(self):
        with pytest.raises(ValueError, match="peak_memory_per_sample must be > 0"):
            estimate_batch_size_from_memory(
                peak_memory_per_sample=0,
                available_memory=10_000_000_000,
                is_main_process=False,
            )

    def test_negative_available_memory_raises(self):
        with pytest.raises(ValueError, match="available_memory must be >= 0"):
            estimate_batch_size_from_memory(
                peak_memory_per_sample=1_000_000,
                available_memory=-1000,
                is_main_process=False,
            )

    def test_invalid_safety_margin_low_raises(self):
        with pytest.raises(ValueError, match="safety_margin must be in"):
            estimate_batch_size_from_memory(
                peak_memory_per_sample=1_000_000,
                available_memory=10_000_000_000,
                safety_margin=0.0,
                is_main_process=False,
            )

    def test_invalid_safety_margin_high_raises(self):
        with pytest.raises(ValueError, match="safety_margin must be in"):
            estimate_batch_size_from_memory(
                peak_memory_per_sample=1_000_000,
                available_memory=10_000_000_000,
                safety_margin=1.5,
                is_main_process=False,
            )

    def test_zero_available_memory(self):
        batch_size = estimate_batch_size_from_memory(
            peak_memory_per_sample=1_000_000,
            available_memory=0,
            safety_margin=0.9,
            min_size=1,
            max_size=256,
            is_main_process=False,
        )
        assert batch_size == 1

    @patch("lerobot.common.memory_utils.logging.info")
    def test_logging_when_main_process(self, mock_logging):
        estimate_batch_size_from_memory(
            peak_memory_per_sample=1_000_000_000,
            available_memory=10_000_000_000,
            safety_margin=0.9,
            min_size=1,
            max_size=256,
            is_main_process=True,
        )
        mock_logging.assert_called_once()

    @patch("lerobot.common.memory_utils.logging.info")
    def test_no_logging_when_not_main_process(self, mock_logging):
        estimate_batch_size_from_memory(
            peak_memory_per_sample=1_000_000_000,
            available_memory=10_000_000_000,
            safety_margin=0.9,
            min_size=1,
            max_size=256,
            is_main_process=False,
        )
        mock_logging.assert_not_called()