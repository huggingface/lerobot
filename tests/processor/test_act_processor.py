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
"""Tests for ACT policy processor."""

import tempfile

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import make_act_pre_post_processors
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
    """Create a default ACT configuration for testing."""
    config = ACTConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    config.device = "cpu"
    return config


def create_default_stats():
    """Create default dataset statistics for testing."""
    return {
        OBS_STATE: {"mean": torch.zeros(7), "std": torch.ones(7)},
        ACTION: {"mean": torch.zeros(4), "std": torch.ones(4)},
    }


def test_make_act_processor_basic():
    """Test basic creation of ACT processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(config, stats)

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


def test_act_processor_normalization():
    """Test that ACT processor correctly normalizes and unnormalizes data."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    # Create test data
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data is normalized and batched
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 7)
    assert processed[TransitionKey.ACTION].shape == (1, 4)

    # Process action through postprocessor
    action_transition = create_transition(action=processed[TransitionKey.ACTION])
    postprocessed = postprocessor(action_transition)

    # Check that action is unnormalized
    assert postprocessed[TransitionKey.ACTION].shape == (1, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_processor_cuda():
    """Test ACT processor with CUDA device."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    # Create CPU data
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
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
def test_act_processor_accelerate_scenario():
    """Test ACT processor in simulated Accelerate scenario (data already on GPU)."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    # Simulate Accelerate: data already on GPU
    device = torch.device("cuda:0")
    observation = {OBS_STATE: torch.randn(1, 7).to(device)}  # Already batched and on GPU
    action = torch.randn(1, 4).to(device)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data stays on same GPU (not moved unnecessarily)
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].device == device
    assert processed[TransitionKey.ACTION].device == device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_act_processor_multi_gpu():
    """Test ACT processor with multi-GPU setup."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(config, stats)

    # Simulate data on different GPU (like in multi-GPU training)
    device = torch.device("cuda:1")
    observation = {OBS_STATE: torch.randn(1, 7).to(device)}
    action = torch.randn(1, 4).to(device)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data stays on cuda:1 (not moved to cuda:0)
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].device == device
    assert processed[TransitionKey.ACTION].device == device


def test_act_processor_without_stats():
    """Test ACT processor creation without dataset statistics."""
    config = create_default_config()

    preprocessor, postprocessor = make_act_pre_post_processors(config, dataset_stats=None)

    # Should still create processors, but normalization won't have stats
    assert preprocessor is not None
    assert postprocessor is not None

    # Process should still work (but won't normalize without stats)
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)

    processed = preprocessor(transition)
    assert processed is not None


def test_act_processor_save_and_load():
    """Test saving and loading ACT processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
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
        observation = {OBS_STATE: torch.randn(7)}
        action = torch.randn(4)
        transition = create_transition(observation, action)

        processed = loaded_preprocessor(transition)
        assert processed[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 7)
        assert processed[TransitionKey.ACTION].shape == (1, 4)


def test_act_processor_device_placement_preservation():
    """Test that ACT processor preserves device placement correctly."""
    config = create_default_config()
    stats = create_default_stats()

    # Test with CPU config
    config.device = "cpu"
    preprocessor, _ = make_act_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    # Process CPU data
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)

    processed = preprocessor(transition)
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cpu"
    assert processed[TransitionKey.ACTION].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_processor_mixed_precision():
    """Test ACT processor with mixed precision (float16)."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    # Modify the device processor to use float16
    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    # Replace DeviceProcessor with one that uses float16
    modified_steps = []
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessor):
            modified_steps.append(DeviceProcessor(device=config.device, float_dtype="float16"))
        else:
            modified_steps.append(step)
    preprocessor.steps = modified_steps

    # Create test data
    observation = {OBS_STATE: torch.randn(7, dtype=torch.float32)}
    action = torch.randn(4, dtype=torch.float32)
    transition = create_transition(observation, action)

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data is converted to float16
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].dtype == torch.float16
    assert processed[TransitionKey.ACTION].dtype == torch.float16


def test_act_processor_batch_consistency():
    """Test that ACT processor handles different batch sizes correctly."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
        preprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
        postprocessor_kwargs={"to_transition": lambda x: x, "to_output": lambda x: x},
    )

    # Test single sample (unbatched)
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)

    processed = preprocessor(transition)
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].shape[0] == 1  # Batched

    # Test already batched data
    observation_batched = {OBS_STATE: torch.randn(8, 7)}  # Batch of 8
    action_batched = torch.randn(8, 4)
    transition_batched = create_transition(observation_batched, action_batched)

    processed_batched = preprocessor(transition_batched)
    assert processed_batched[TransitionKey.OBSERVATION][OBS_STATE].shape[0] == 8
    assert processed_batched[TransitionKey.ACTION].shape[0] == 8
