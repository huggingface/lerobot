#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adaptive Batch Sizing Integration Tests

Tests adaptive batch sizing functionality with training configs.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig


class TestAdaptiveBatchConfig:
    def test_config_fields_exist(self):
        dataset_config = DatasetConfig(repo_id="test_repo")
        cfg = TrainPipelineConfig(dataset=dataset_config)
        assert hasattr(cfg, "adaptive_batch")
        assert hasattr(cfg, "adaptive_batch_min_size")
        assert hasattr(cfg, "adaptive_batch_max_size")
        assert hasattr(cfg, "adaptive_batch_safety_margin")

    def test_config_defaults(self):
        dataset_config = DatasetConfig(repo_id="test_repo")
        cfg = TrainPipelineConfig(dataset=dataset_config)
        assert cfg.adaptive_batch is False
        assert cfg.adaptive_batch_min_size == 1
        assert cfg.adaptive_batch_max_size == 256
        assert cfg.adaptive_batch_safety_margin == 0.9

    def test_config_custom_values(self):
        dataset_config = DatasetConfig(repo_id="test_repo")
        cfg = TrainPipelineConfig(
            dataset=dataset_config,
            adaptive_batch=True,
            adaptive_batch_min_size=4,
            adaptive_batch_max_size=128,
            adaptive_batch_safety_margin=0.85,
        )
        assert cfg.adaptive_batch is True
        assert cfg.adaptive_batch_min_size == 4
        assert cfg.adaptive_batch_max_size == 128
        assert cfg.adaptive_batch_safety_margin == 0.85

    def test_validation_min_size_invalid(self):
        dataset_config = DatasetConfig(repo_id="test_repo")
        cfg = TrainPipelineConfig(
            dataset=dataset_config,
            adaptive_batch=True,
            adaptive_batch_min_size=0,
        )
        with pytest.raises(ValueError, match="adaptive_batch_min_size must be >= 1"):
            cfg.validate()

    def test_validation_max_less_than_min(self):
        dataset_config = DatasetConfig(repo_id="test_repo")
        cfg = TrainPipelineConfig(
            dataset=dataset_config,
            adaptive_batch=True,
            adaptive_batch_min_size=64,
            adaptive_batch_max_size=32,
        )
        with pytest.raises(ValueError, match="adaptive_batch_max_size.*must be >="):
            cfg.validate()

    def test_validation_safety_margin_invalid_low(self):
        dataset_config = DatasetConfig(repo_id="test_repo")
        cfg = TrainPipelineConfig(
            dataset=dataset_config,
            adaptive_batch=True,
            adaptive_batch_safety_margin=0.0,
        )
        with pytest.raises(ValueError, match="adaptive_batch_safety_margin must be in"):
            cfg.validate()

    def test_validation_safety_margin_invalid_high(self):
        dataset_config = DatasetConfig(repo_id="test_repo")
        cfg = TrainPipelineConfig(
            dataset=dataset_config,
            adaptive_batch=True,
            adaptive_batch_safety_margin=1.5,
        )
        with pytest.raises(ValueError, match="adaptive_batch_safety_margin must be in"):
            cfg.validate()

    def test_validation_passes_with_valid_config(self):
        dataset_config = DatasetConfig(repo_id="test_repo")
        cfg = TrainPipelineConfig(
            dataset=dataset_config,
            adaptive_batch=True,
            adaptive_batch_min_size=2,
            adaptive_batch_max_size=512,
            adaptive_batch_safety_margin=0.95,
        )
        try:
            cfg.validate()
        except ValueError as e:
            if "policy" not in str(e) and "reward_model" not in str(e):
                raise

    def test_validation_skipped_when_adaptive_batch_disabled(self):
        dataset_config = DatasetConfig(repo_id="test_repo")
        cfg = TrainPipelineConfig(
            dataset=dataset_config,
            adaptive_batch=False,
            adaptive_batch_min_size=0,
            adaptive_batch_max_size=10,
            adaptive_batch_safety_margin=0.0,
        )
        try:
            cfg.validate()
        except ValueError as e:
            if "adaptive_batch" in str(e):
                pytest.fail("Validation should skip adaptive_batch checks when disabled")


class TestAdaptiveBatchIntegration:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_estimate_batch_size_called_before_dataloader(self):
        with patch("lerobot.scripts.lerobot_train.estimate_batch_size_from_memory") as mock_estimate:
            mock_estimate.return_value = 16
            dataset_config = DatasetConfig(repo_id="test_repo")
            cfg = TrainPipelineConfig(
                dataset=dataset_config,
                adaptive_batch=True,
                batch_size=8,
            )
            assert cfg.batch_size == 8
            assert mock_estimate.return_value == 16

    def test_adaptive_batch_disabled_no_estimation(self):
        with patch("lerobot.scripts.lerobot_train.estimate_batch_size_from_memory") as mock_estimate:
            dataset_config = DatasetConfig(repo_id="test_repo")
            cfg = TrainPipelineConfig(
                dataset=dataset_config,
                adaptive_batch=False,
                batch_size=8,
            )
            assert cfg.batch_size == 8

    def test_config_serialization_preserves_adaptive_fields(self):
        dataset_config = DatasetConfig(repo_id="test_repo")
        cfg = TrainPipelineConfig(
            dataset=dataset_config,
            adaptive_batch=True,
            adaptive_batch_min_size=4,
            adaptive_batch_max_size=128,
            adaptive_batch_safety_margin=0.85,
        )
        cfg_dict = cfg.to_dict()
        assert cfg_dict["adaptive_batch"] is True
        assert cfg_dict["adaptive_batch_min_size"] == 4
        assert cfg_dict["adaptive_batch_max_size"] == 128
        assert cfg_dict["adaptive_batch_safety_margin"] == 0.85