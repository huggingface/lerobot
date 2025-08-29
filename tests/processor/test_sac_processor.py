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
"""Tests for SAC policy processor."""

import tempfile

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.processor_sac import make_sac_pre_post_processors
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
    """Create a default SAC configuration for testing."""
    config = SACConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(5,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    config.device = "cpu"
    return config


def create_default_stats():
    """Create default dataset statistics for testing."""
    return {
        OBS_STATE: {"mean": torch.zeros(10), "std": torch.ones(10)},
        ACTION: {"min": torch.full((5,), -1.0), "max": torch.ones(5)},
    }


def test_make_sac_processor_basic():
    """Test basic creation of SAC processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_sac_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

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


def test_sac_processor_normalization_modes():
    """Test that SAC processor correctly handles different normalization modes."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_sac_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    # Create test data
    observation = {OBS_STATE: torch.randn(10) * 2}  # Larger values to test normalization
    action = torch.rand(5) * 2 - 1  # Range [-1, 1]
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data is normalized and batched
    # State should be mean-std normalized
    # Action should be min-max normalized to [-1, 1]
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 10)
    assert processed[TransitionKey.ACTION].shape == (1, 5)

    # Process action through postprocessor
    action_transition = create_transition(action=processed[TransitionKey.ACTION])
    postprocessed = postprocessor(action_transition)

    # Check that action is unnormalized (but still batched)
    assert postprocessed[TransitionKey.ACTION].shape == (1, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sac_processor_cuda():
    """Test SAC processor with CUDA device."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, postprocessor = make_sac_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    # Create CPU data
    observation = {OBS_STATE: torch.randn(10)}
    action = torch.randn(5)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data is on CUDA
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cuda"
    assert processed[TransitionKey.ACTION].device.type == "cuda"

    # Process through postprocessor
    action_transition = create_transition(action=processed[TransitionKey.ACTION])
    postprocessed = postprocessor(action_transition)

    # Check that action is back on CPU
    assert postprocessed[TransitionKey.ACTION].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sac_processor_accelerate_scenario():
    """Test SAC processor in simulated Accelerate scenario."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_sac_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    # Simulate Accelerate: data already on GPU
    device = torch.device("cuda:0")
    observation = {OBS_STATE: torch.randn(10).to(device)}
    action = torch.randn(5).to(device)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data stays on same GPU
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].device == device
    assert processed[TransitionKey.ACTION].device == device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_sac_processor_multi_gpu():
    """Test SAC processor with multi-GPU setup."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_sac_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    # Simulate data on different GPU
    device = torch.device("cuda:1")
    observation = {OBS_STATE: torch.randn(10).to(device)}
    action = torch.randn(5).to(device)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data stays on cuda:1
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].device == device
    assert processed[TransitionKey.ACTION].device == device


def test_sac_processor_without_stats():
    """Test SAC processor creation without dataset statistics."""
    config = create_default_config()

    # Get the steps from the factory function
    factory_preprocessor, factory_postprocessor = make_sac_pre_post_processors(config, dataset_stats=None)

    # Create new processors with EnvTransition input/output
    preprocessor = RobotProcessor(
        factory_preprocessor.steps,
        name=factory_preprocessor.name,
        to_transition=lambda x: x,
        to_output=lambda x: x,
    )
    postprocessor = RobotProcessor(
        factory_postprocessor.steps,
        name=factory_postprocessor.name,
        to_transition=lambda x: x,
        to_output=lambda x: x,
    )

    # Should still create processors
    assert preprocessor is not None
    assert postprocessor is not None

    # Process should still work
    observation = {OBS_STATE: torch.randn(10)}
    action = torch.randn(5)
    transition = create_transition(observation, action)

    processed = preprocessor(transition)
    assert processed is not None


def test_sac_processor_save_and_load():
    """Test saving and loading SAC processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_sac_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save preprocessor
        preprocessor.save_pretrained(tmpdir)

        # Load preprocessor
        loaded_preprocessor = RobotProcessor.from_pretrained(
            tmpdir, to_transition=lambda x: x, to_output=lambda x: x
        )

        # Test that loaded processor works
        observation = {OBS_STATE: torch.randn(10)}
        action = torch.randn(5)
        transition = create_transition(observation, action)

        processed = loaded_preprocessor(transition)
        assert processed[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 10)
        assert processed[TransitionKey.ACTION].shape == (1, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sac_processor_mixed_precision():
    """Test SAC processor with mixed precision."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    # Create processor
    preprocessor, postprocessor = make_sac_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    # Replace DeviceProcessor with one that uses float16
    for i, step in enumerate(preprocessor.steps):
        if isinstance(step, DeviceProcessor):
            preprocessor.steps[i] = DeviceProcessor(device=config.device, float_dtype="float16")

    # Create test data
    observation = {OBS_STATE: torch.randn(10, dtype=torch.float32)}
    action = torch.randn(5, dtype=torch.float32)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data is converted to float16
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].dtype == torch.float16
    assert processed[TransitionKey.ACTION].dtype == torch.float16


def test_sac_processor_batch_data():
    """Test SAC processor with batched data."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_sac_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    # Test with batched data
    batch_size = 32
    observation = {OBS_STATE: torch.randn(batch_size, 10)}
    action = torch.randn(batch_size, 5)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that batch dimension is preserved
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].shape == (batch_size, 10)
    assert processed[TransitionKey.ACTION].shape == (batch_size, 5)


def test_sac_processor_edge_cases():
    """Test SAC processor with edge cases."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_sac_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    # Test with empty observation
    transition = create_transition(observation={}, action=torch.randn(5))
    processed = preprocessor(transition)
    assert processed[TransitionKey.OBSERVATION] == {}
    assert processed[TransitionKey.ACTION].shape == (1, 5)

    # Test with None action
    transition = create_transition(observation={OBS_STATE: torch.randn(10)}, action=None)
    processed = preprocessor(transition)
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 10)
    # When action is None, it may still be present with None value
    assert TransitionKey.ACTION not in processed or processed[TransitionKey.ACTION] is None
