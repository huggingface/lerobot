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
"""Tests for Reward Classifier processor."""

import tempfile

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.policies.sac.reward_model.processor_classifier import make_classifier_processor
from lerobot.processor import (
    DataProcessorPipeline,
    DeviceProcessorStep,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    TransitionKey,
)
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE


def create_default_config():
    """Create a default Reward Classifier configuration for testing."""
    config = RewardClassifierConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "reward": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),  # Classifier output
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,  # No normalization for classifier output
    }
    config.device = "cpu"
    return config


def create_default_stats():
    """Create default dataset statistics for testing."""
    return {
        OBS_STATE: {"mean": torch.zeros(10), "std": torch.ones(10)},
        OBS_IMAGE: {},  # No normalization for images
        "reward": {},  # No normalization for classifier output
    }


def test_make_classifier_processor_basic():
    """Test basic creation of Classifier processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    # Check processor names
    assert preprocessor.name == "classifier_preprocessor"
    assert postprocessor.name == "classifier_postprocessor"

    # Check steps in preprocessor
    assert len(preprocessor.steps) == 3
    assert isinstance(preprocessor.steps[0], NormalizerProcessorStep)  # For input features
    assert isinstance(preprocessor.steps[1], NormalizerProcessorStep)  # For output features
    assert isinstance(preprocessor.steps[2], DeviceProcessorStep)

    # Check steps in postprocessor
    assert len(postprocessor.steps) == 2
    assert isinstance(postprocessor.steps[0], DeviceProcessorStep)
    assert isinstance(postprocessor.steps[1], IdentityProcessorStep)


def test_classifier_processor_normalization():
    """Test that Classifier processor correctly normalizes data."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(
        config,
        stats,
    )

    # Create test data
    observation = {
        OBS_STATE: torch.randn(10),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(1)  # Dummy action/reward
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data is processed
    assert processed[OBS_STATE].shape == (10,)
    assert processed[OBS_IMAGE].shape == (3, 224, 224)
    assert processed[TransitionKey.ACTION.value].shape == (1,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_classifier_processor_cuda():
    """Test Classifier processor with CUDA device."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(
        config,
        stats,
    )

    # Create CPU data
    observation = {
        OBS_STATE: torch.randn(10),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(1)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data is on CUDA
    assert processed[OBS_STATE].device.type == "cuda"
    assert processed[OBS_IMAGE].device.type == "cuda"
    assert processed[TransitionKey.ACTION.value].device.type == "cuda"

    # Process through postprocessor
    postprocessed = postprocessor(processed[TransitionKey.ACTION.value])

    # Check that output is back on CPU
    assert postprocessed.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_classifier_processor_accelerate_scenario():
    """Test Classifier processor in simulated Accelerate scenario."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(
        config,
        stats,
    )

    # Simulate Accelerate: data already on GPU
    device = torch.device("cuda:0")
    observation = {
        OBS_STATE: torch.randn(10).to(device),
        OBS_IMAGE: torch.randn(3, 224, 224).to(device),
    }
    action = torch.randn(1).to(device)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data stays on same GPU
    assert processed[OBS_STATE].device == device
    assert processed[OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_classifier_processor_multi_gpu():
    """Test Classifier processor with multi-GPU setup."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    # Simulate data on different GPU
    device = torch.device("cuda:1")
    observation = {
        OBS_STATE: torch.randn(10).to(device),
        OBS_IMAGE: torch.randn(3, 224, 224).to(device),
    }
    action = torch.randn(1).to(device)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data stays on cuda:1
    assert processed[OBS_STATE].device == device
    assert processed[OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


def test_classifier_processor_without_stats():
    """Test Classifier processor creation without dataset statistics."""
    config = create_default_config()

    preprocessor, postprocessor = make_classifier_processor(config, dataset_stats=None)

    # Should still create processors
    assert preprocessor is not None
    assert postprocessor is not None

    # Process should still work
    observation = {
        OBS_STATE: torch.randn(10),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(1)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed is not None


def test_classifier_processor_save_and_load():
    """Test saving and loading Classifier processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save preprocessor
        preprocessor.save_pretrained(tmpdir)

        # Load preprocessor
        loaded_preprocessor = DataProcessorPipeline.from_pretrained(
            tmpdir, config_filename="classifier_preprocessor.json"
        )

        # Test that loaded processor works
        observation = {
            OBS_STATE: torch.randn(10),
            OBS_IMAGE: torch.randn(3, 224, 224),
        }
        action = torch.randn(1)
        transition = create_transition(observation, action)
        batch = transition_to_batch(transition)

        processed = loaded_preprocessor(batch)
        assert processed[OBS_STATE].shape == (10,)
        assert processed[OBS_IMAGE].shape == (3, 224, 224)
        assert processed[TransitionKey.ACTION.value].shape == (1,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_classifier_processor_mixed_precision():
    """Test Classifier processor with mixed precision."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    # Replace DeviceProcessorStep with one that uses float16
    modified_steps = []
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            modified_steps.append(DeviceProcessorStep(device=config.device, float_dtype="float16"))
        else:
            modified_steps.append(step)
    preprocessor.steps = modified_steps

    # Create test data
    observation = {
        OBS_STATE: torch.randn(10, dtype=torch.float32),
        OBS_IMAGE: torch.randn(3, 224, 224, dtype=torch.float32),
    }
    action = torch.randn(1, dtype=torch.float32)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data is converted to float16
    assert processed[OBS_STATE].dtype == torch.float16
    assert processed[OBS_IMAGE].dtype == torch.float16
    assert processed[TransitionKey.ACTION.value].dtype == torch.float16


def test_classifier_processor_batch_data():
    """Test Classifier processor with batched data."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(
        config,
        stats,
    )

    # Test with batched data
    batch_size = 16
    observation = {
        OBS_STATE: torch.randn(batch_size, 10),
        OBS_IMAGE: torch.randn(batch_size, 3, 224, 224),
    }
    action = torch.randn(batch_size, 1)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that batch dimension is preserved
    assert processed[OBS_STATE].shape == (batch_size, 10)
    assert processed[OBS_IMAGE].shape == (batch_size, 3, 224, 224)
    assert processed[TransitionKey.ACTION.value].shape == (batch_size, 1)


def test_classifier_processor_postprocessor_identity():
    """Test that Classifier postprocessor uses IdentityProcessor correctly."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(
        config,
        stats,
    )

    # Create test data for postprocessor
    reward = torch.tensor([[0.8], [0.3], [0.9]])  # Batch of rewards/predictions
    transition = create_transition(action=reward)

    _ = transition_to_batch(transition)

    # Process through postprocessor
    processed = postprocessor(reward)

    # IdentityProcessor should leave values unchanged (except device)
    assert torch.allclose(processed.cpu(), reward.cpu())
    assert processed.device.type == "cpu"
