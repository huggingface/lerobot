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
"""Tests for PI0 policy processor."""

from unittest.mock import patch

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.processor_pi0 import Pi0NewLineProcessor, make_pi0_pre_post_processors
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessor,
    RenameProcessor,
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
        elif key == "complementary_data":
            transition[TransitionKey.COMPLEMENTARY_DATA] = value
    return transition


def create_default_config():
    """Create a default PI0 configuration for testing."""
    config = PI0Config()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    config.device = "cpu"
    config.tokenizer_max_length = 128
    return config


def create_default_stats():
    """Create default dataset statistics for testing."""
    return {
        OBS_STATE: {"mean": torch.zeros(10), "std": torch.ones(10)},
        OBS_IMAGE: {},  # No normalization for images
        ACTION: {"min": torch.full((6,), -1.0), "max": torch.ones(6)},
    }


def test_make_pi0_processor_basic():
    """Test basic creation of PI0 processor."""
    config = create_default_config()
    stats = create_default_stats()

    with patch("lerobot.policies.pi0.processor_pi0.TokenizerProcessorStep"):
        preprocessor, postprocessor = make_pi0_pre_post_processors(config, stats)

    # Check processor names
    assert preprocessor.name == "robot_preprocessor"
    assert postprocessor.name == "robot_postprocessor"

    # Check steps in preprocessor
    assert len(preprocessor.steps) == 6
    assert isinstance(preprocessor.steps[0], RenameProcessor)
    assert isinstance(preprocessor.steps[1], NormalizerProcessor)
    assert isinstance(preprocessor.steps[2], AddBatchDimensionProcessorStep)
    assert isinstance(preprocessor.steps[3], Pi0NewLineProcessor)
    # Step 4 would be TokenizerProcessorStep but it's mocked
    assert isinstance(preprocessor.steps[5], DeviceProcessorStep)

    # Check steps in postprocessor
    assert len(postprocessor.steps) == 2
    assert isinstance(postprocessor.steps[0], DeviceProcessorStep)
    assert isinstance(postprocessor.steps[1], UnnormalizerProcessor)


def test_pi0_newline_processor_single_task():
    """Test Pi0NewLineProcessor with single task string."""
    processor = Pi0NewLineProcessor()

    # Test with task that doesn't have newline
    transition = create_transition(complementary_data={"task": "test task"})
    result = processor(transition)
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == "test task\n"

    # Test with task that already has newline
    transition = create_transition(complementary_data={"task": "test task\n"})
    result = processor(transition)
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == "test task\n"


def test_pi0_newline_processor_list_of_tasks():
    """Test Pi0NewLineProcessor with list of task strings."""
    processor = Pi0NewLineProcessor()

    # Test with list of tasks
    tasks = ["task1", "task2\n", "task3"]
    transition = create_transition(complementary_data={"task": tasks})
    result = processor(transition)
    expected = ["task1\n", "task2\n", "task3\n"]
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == expected


def test_pi0_newline_processor_empty_transition():
    """Test Pi0NewLineProcessor with empty transition."""
    processor = Pi0NewLineProcessor()

    # Test with no complementary_data
    transition = create_transition()
    result = processor(transition)
    assert result == transition

    # Test with complementary_data but no task
    transition = create_transition(complementary_data={"other": "data"})
    result = processor(transition)
    assert result == transition

    # Test with None task
    transition = create_transition(complementary_data={"task": None})
    result = processor(transition)
    assert result == transition


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pi0_processor_cuda():
    """Test PI0 processor with CUDA device."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    # Mock the tokenizer processor to act as pass-through
    class MockTokenizerProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, transition):
            return transition

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            pass

        def reset(self):
            pass

        def get_config(self):
            return {"tokenizer_name": "google/paligemma-3b-pt-224"}

        def transform_features(self, features):
            return features

    with patch("lerobot.policies.pi0.processor_pi0.TokenizerProcessorStep", MockTokenizerProcessor):
        preprocessor, postprocessor = make_pi0_pre_post_processors(config, stats)

    # Create CPU data
    observation = {
        OBS_STATE: torch.randn(10),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(6)
    transition = create_transition(observation, action, complementary_data={"task": "test task"})

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data is on CUDA
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].device.type == "cuda"
    assert processed[TransitionKey.OBSERVATION][OBS_IMAGE].device.type == "cuda"
    assert processed[TransitionKey.ACTION].device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pi0_processor_accelerate_scenario():
    """Test PI0 processor in simulated Accelerate scenario."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    # Mock the tokenizer processor to act as pass-through
    class MockTokenizerProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, transition):
            return transition

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            pass

        def reset(self):
            pass

        def get_config(self):
            return {"tokenizer_name": "google/paligemma-3b-pt-224"}

        def transform_features(self, features):
            return features

    with patch("lerobot.policies.pi0.processor_pi0.TokenizerProcessorStep", MockTokenizerProcessor):
        preprocessor, postprocessor = make_pi0_pre_post_processors(config, stats)

    # Simulate Accelerate: data already on GPU and batched
    device = torch.device("cuda:0")
    observation = {
        OBS_STATE: torch.randn(1, 10).to(device),
        OBS_IMAGE: torch.randn(1, 3, 224, 224).to(device),
    }
    action = torch.randn(1, 6).to(device)
    transition = create_transition(observation, action, complementary_data={"task": ["test task"]})

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data stays on same GPU
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].device == device
    assert processed[TransitionKey.OBSERVATION][OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION].device == device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_pi0_processor_multi_gpu():
    """Test PI0 processor with multi-GPU setup."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    # Mock the tokenizer processor to act as pass-through
    class MockTokenizerProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, transition):
            return transition

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            pass

        def reset(self):
            pass

        def get_config(self):
            return {"tokenizer_name": "google/paligemma-3b-pt-224"}

        def transform_features(self, features):
            return features

    with patch("lerobot.policies.pi0.processor_pi0.TokenizerProcessorStep", MockTokenizerProcessor):
        preprocessor, postprocessor = make_pi0_pre_post_processors(config, stats)

    # Simulate data on different GPU
    device = torch.device("cuda:1")
    observation = {
        OBS_STATE: torch.randn(1, 10).to(device),
        OBS_IMAGE: torch.randn(1, 3, 224, 224).to(device),
    }
    action = torch.randn(1, 6).to(device)
    transition = create_transition(observation, action, complementary_data={"task": ["test task"]})

    # Process through preprocessor
    processed = preprocessor(transition)

    # Check that data stays on cuda:1
    assert processed[TransitionKey.OBSERVATION][OBS_STATE].device == device
    assert processed[TransitionKey.OBSERVATION][OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION].device == device


def test_pi0_processor_without_stats():
    """Test PI0 processor creation without dataset statistics."""
    config = create_default_config()

    # Mock the tokenizer processor
    with patch("lerobot.policies.pi0.processor_pi0.TokenizerProcessorStep"):
        preprocessor, postprocessor = make_pi0_pre_post_processors(config, dataset_stats=None)

    # Should still create processors
    assert preprocessor is not None
    assert postprocessor is not None


def test_pi0_newline_processor_state_dict():
    """Test Pi0NewLineProcessor state dict methods."""
    processor = Pi0NewLineProcessor()

    # Test state_dict (should be empty)
    state = processor.state_dict()
    assert state == {}

    # Test load_state_dict (should do nothing)
    processor.load_state_dict({})

    # Test reset (should do nothing)
    processor.reset()

    # Test get_config
    config = processor.get_config()
    assert config == {}
