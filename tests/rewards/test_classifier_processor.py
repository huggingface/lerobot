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

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    DataProcessorPipeline,
    DeviceProcessorStep,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    TransitionKey,
)
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig
from lerobot.rewards.classifier.processor_classifier import make_classifier_processor
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE


def create_default_config():
    """Create a default Reward Classifier configuration for testing."""
    config = RewardClassifierConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "reward": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,
    }
    config.device = "cpu"
    return config


def create_default_stats():
    """Create default dataset statistics for testing."""
    return {
        OBS_STATE: {"mean": torch.zeros(10), "std": torch.ones(10)},
        OBS_IMAGE: {},
        "reward": {},
    }


def test_make_classifier_processor_basic():
    """Test basic creation of Classifier processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    assert preprocessor.name == "classifier_preprocessor"
    assert postprocessor.name == "classifier_postprocessor"

    assert len(preprocessor.steps) == 3
    assert isinstance(preprocessor.steps[0], NormalizerProcessorStep)
    assert isinstance(preprocessor.steps[1], NormalizerProcessorStep)
    assert isinstance(preprocessor.steps[2], DeviceProcessorStep)

    assert len(postprocessor.steps) == 2
    assert isinstance(postprocessor.steps[0], DeviceProcessorStep)
    assert isinstance(postprocessor.steps[1], IdentityProcessorStep)


def test_classifier_processor_normalization():
    """Test that Classifier processor correctly normalizes data."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    observation = {
        OBS_STATE: torch.randn(10),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(1)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)

    assert processed[OBS_STATE].shape == (10,)
    assert processed[OBS_IMAGE].shape == (3, 224, 224)
    assert processed[TransitionKey.ACTION.value].shape == (1,)


def test_classifier_processor_without_stats():
    """Test Classifier processor creation without dataset statistics."""
    config = create_default_config()

    preprocessor, postprocessor = make_classifier_processor(config, dataset_stats=None)

    assert preprocessor is not None
    assert postprocessor is not None

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
        preprocessor.save_pretrained(tmpdir)

        loaded_preprocessor = DataProcessorPipeline.from_pretrained(
            tmpdir, config_filename="classifier_preprocessor.json"
        )

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


def test_classifier_processor_batch_data():
    """Test Classifier processor with batched data."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    batch_size = 16
    observation = {
        OBS_STATE: torch.randn(batch_size, 10),
        OBS_IMAGE: torch.randn(batch_size, 3, 224, 224),
    }
    action = torch.randn(batch_size, 1)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)

    assert processed[OBS_STATE].shape == (batch_size, 10)
    assert processed[OBS_IMAGE].shape == (batch_size, 3, 224, 224)
    assert processed[TransitionKey.ACTION.value].shape == (batch_size, 1)


def test_classifier_processor_postprocessor_identity():
    """Test that Classifier postprocessor uses IdentityProcessor correctly."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    reward = torch.tensor([[0.8], [0.3], [0.9]])
    transition = create_transition(action=reward)
    _ = transition_to_batch(transition)

    processed = postprocessor(reward)

    assert torch.allclose(processed.cpu(), reward.cpu())
    assert processed.device.type == "cpu"
