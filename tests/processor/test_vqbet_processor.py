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
"""Tests for VQBeT policy processor."""

import tempfile

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.vqbet.processor_vqbet import make_vqbet_pre_post_processors
from lerobot.processor import (
    DeviceProcessor,
    NormalizerProcessor,
    RenameProcessor,
    RobotProcessor,
    ToBatchProcessor,
    UnnormalizerProcessor,
)
from lerobot.processor.pipeline import TransitionKey


def create_transition(observation=None, action=None, **kwargs):
    """Helper function to create a transition dictionary."""
    transition = {}
    if observation is not None:
        transition[TransitionKey.OBSERVATION] = observation
    if action is not None:
        transition[TransitionKey.ACTION] = action
    for key, value in kwargs.items():
        if hasattr(TransitionKey, key.upper()):
            transition[getattr(TransitionKey, key.upper())] = value
    return transition


def create_default_config():
    """Create a default VQBeT configuration for testing."""
    config = VQBeTConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    config.device = "cpu"
    return config


def create_default_stats():
    """Create default dataset statistics for testing."""
    return {
        OBS_STATE: {"mean": torch.zeros(8), "std": torch.ones(8)},
        OBS_IMAGE: {},  # No normalization for images
        ACTION: {"min": torch.full((7,), -1.0), "max": torch.ones(7)},
    }


def test_make_vqbet_processor_basic():
    """Test basic creation of VQBeT processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_vqbet_pre_post_processors(config, stats)

    # Check processor names
    assert preprocessor.name == "robot_preprocessor"
    assert postprocessor.name == "robot_postprocessor"

    # Check steps in preprocessor
    assert len(preprocessor.steps) == 4
    assert isinstance(preprocessor.steps[0], RenameProcessor)
    assert isinstance(preprocessor.steps[1], NormalizerProcessor)
    assert isinstance(preprocessor.steps[2], ToBatchProcessor)
    assert isinstance(preprocessor.steps[3], DeviceProcessor)

    # Check steps in postprocessor
    assert len(postprocessor.steps) == 2
    assert isinstance(postprocessor.steps[0], DeviceProcessor)
    assert isinstance(postprocessor.steps[1], UnnormalizerProcessor)


def test_vqbet_processor_with_images():
    """Test VQBeT processor with image and state observations."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_vqbet_pre_post_processors(config, stats)

    # Create test data with images and states
    observation = {
        OBS_STATE: torch.randn(8),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(7)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data is batched
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 8)
    assert processed[TransitionKey.OBSERVATION][OBS_IMAGE].shape == (1, 3, 224, 224)
    assert processed[TransitionKey.ACTION].shape == (1, 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vqbet_processor_cuda():
    """Test VQBeT processor with CUDA device."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, postprocessor = make_vqbet_pre_post_processors(config, stats)

    # Create CPU data
    observation = {
        OBS_STATE: torch.randn(8),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(7)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data is on CUDA
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cuda"
    assert processed[TransitionKey.OBSERVATION][OBS_IMAGE].device.type == "cuda"
    assert processed[TransitionKey.ACTION].device.type == "cuda"

    # Process through postprocessor
    action_transition = create_transition(action=processed[TransitionKey.ACTION])
    postprocessed = postprocessor(action_transition)

    # Check that action is back on CPU
    assert postprocessed[TransitionKey.ACTION].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vqbet_processor_accelerate_scenario():
    """Test VQBeT processor in simulated Accelerate scenario."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_vqbet_pre_post_processors(config, stats)

    # Simulate Accelerate: data already on GPU and batched
    device = torch.device("cuda:0")
    observation = {
        OBS_STATE: torch.randn(1, 8).to(device),
        OBS_IMAGE: torch.randn(1, 3, 224, 224).to(device),
    }
    action = torch.randn(1, 7).to(device)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data stays on same GPU
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].device == device
    assert processed[TransitionKey.OBSERVATION][OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION].device == device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_vqbet_processor_multi_gpu():
    """Test VQBeT processor with multi-GPU setup."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_vqbet_pre_post_processors(config, stats)

    # Simulate data on different GPU
    device = torch.device("cuda:1")
    observation = {
        OBS_STATE: torch.randn(1, 8).to(device),
        OBS_IMAGE: torch.randn(1, 3, 224, 224).to(device),
    }
    action = torch.randn(1, 7).to(device)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data stays on cuda:1
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].device == device
    assert processed[TransitionKey.OBSERVATION][OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION].device == device


def test_vqbet_processor_without_stats():
    """Test VQBeT processor creation without dataset statistics."""
    config = create_default_config()

    preprocessor, postprocessor = make_vqbet_pre_post_processors(config, dataset_stats=None)

    # Should still create processors
    assert preprocessor is not None
    assert postprocessor is not None

    # Process should still work
    observation = {
        OBS_STATE: torch.randn(8),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(7)
    transition = create_transition(observation, action)

    processed = preprocessor(transition)
    assert processed is not None


def test_vqbet_processor_save_and_load():
    """Test saving and loading VQBeT processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_vqbet_pre_post_processors(config, stats)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save preprocessor
        preprocessor.save_pretrained(tmpdir)

        # Load preprocessor
        loaded_preprocessor = RobotProcessor.from_pretrained(tmpdir)

        # Test that loaded processor works
        observation = {
            OBS_STATE: torch.randn(8),
            OBS_IMAGE: torch.randn(3, 224, 224),
        }
        action = torch.randn(7)
        transition = create_transition(observation, action)

        processed = loaded_preprocessor(transition)
        assert processed[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 8)
        assert processed[TransitionKey.OBSERVATION][OBS_IMAGE].shape == (1, 3, 224, 224)
        assert processed[TransitionKey.ACTION].shape == (1, 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vqbet_processor_mixed_precision():
    """Test VQBeT processor with mixed precision."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    # Create processor
    preprocessor, postprocessor = make_vqbet_pre_post_processors(config, stats)

    # Replace DeviceProcessor with one that uses float16
    for i, step in enumerate(preprocessor.steps):
        if isinstance(step, DeviceProcessor):
            preprocessor.steps[i] = DeviceProcessor(device=config.device, float_dtype="float16")

    # Create test data
    observation = {
        OBS_STATE: torch.randn(8, dtype=torch.float32),
        OBS_IMAGE: torch.randn(3, 224, 224, dtype=torch.float32),
    }
    action = torch.randn(7, dtype=torch.float32)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data is converted to float16
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].dtype == torch.float16
    assert processed[TransitionKey.OBSERVATION][OBS_IMAGE].dtype == torch.float16
    assert processed[TransitionKey.ACTION].dtype == torch.float16


def test_vqbet_processor_large_batch():
    """Test VQBeT processor with large batch sizes."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_vqbet_pre_post_processors(config, stats)

    # Test with large batch
    batch_size = 128
    observation = {
        OBS_STATE: torch.randn(batch_size, 8),
        OBS_IMAGE: torch.randn(batch_size, 3, 224, 224),
    }
    action = torch.randn(batch_size, 7)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that batch dimension is preserved
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].shape == (batch_size, 8)
    assert processed[TransitionKey.OBSERVATION][OBS_IMAGE].shape == (batch_size, 3, 224, 224)
    assert processed[TransitionKey.ACTION].shape == (batch_size, 7)


def test_vqbet_processor_sequential_processing():
    """Test VQBeT processor with sequential data processing."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_vqbet_pre_post_processors(config, stats)

    # Process multiple samples sequentially
    results = []
    for _ in range(5):
        observation = {
            OBS_STATE: torch.randn(8),
            OBS_IMAGE: torch.randn(3, 224, 224),
        }
        action = torch.randn(7)
        transition = create_transition(observation, action)

        processed = preprocessor(transition)
        results.append(processed)

    # Check that all results are consistent
    for result in results:
        assert result[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 8)
        assert result[TransitionKey.OBSERVATION][OBS_IMAGE].shape == (1, 3, 224, 224)
        assert result[TransitionKey.ACTION].shape == (1, 7)
