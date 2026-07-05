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
"""Tests for TDMPC policy processor."""

import tempfile

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.tdmpc.processor_tdmpc import make_tdmpc_pre_post_processors
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    RenameObservationsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def create_default_config():
    """Create a default TDMPC configuration for testing."""
    config = TDMPCConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
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
    return config


def create_default_stats():
    """Create default dataset statistics for testing."""
    return {
        OBS_STATE: {"mean": torch.zeros(12), "std": torch.ones(12)},
        OBS_IMAGE: {},  # No normalization for images
        ACTION: {"min": torch.full((6,), -1.0), "max": torch.ones(6)},
    }


def test_make_tdmpc_processor_basic():
    """Test basic creation of TDMPC processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_tdmpc_pre_post_processors(
        config,
        stats,
    )

    # Check processor names
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"

    # Check steps in preprocessor
    assert len(preprocessor.steps) == 4
    assert isinstance(preprocessor.steps[0], RenameObservationsProcessorStep)
    assert isinstance(preprocessor.steps[1], AddBatchDimensionProcessorStep)
    assert isinstance(preprocessor.steps[2], DeviceProcessorStep)
    assert isinstance(preprocessor.steps[3], NormalizerProcessorStep)

    # Check steps in postprocessor
    assert len(postprocessor.steps) == 2
    assert isinstance(postprocessor.steps[0], UnnormalizerProcessorStep)
    assert isinstance(postprocessor.steps[1], DeviceProcessorStep)


def test_tdmpc_processor_normalization():
    """Test that TDMPC processor correctly normalizes and unnormalizes data."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_tdmpc_pre_post_processors(
        config,
        stats,
    )

    # Create test data
    observation = {
        OBS_STATE: torch.randn(12),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(6)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data is processed and batched
    assert processed[OBS_STATE].shape == (1, 12)
    assert processed[OBS_IMAGE].shape == (1, 3, 224, 224)
    assert processed[TransitionKey.ACTION.value].shape == (1, 6)

    # Process action through postprocessor
    postprocessed = postprocessor(processed[TransitionKey.ACTION.value])

    # Check that action is unnormalized (but still batched)
    assert postprocessed.shape == (1, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tdmpc_processor_cuda():
    """Test TDMPC processor with CUDA device."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, postprocessor = make_tdmpc_pre_post_processors(
        config,
        stats,
    )

    # Create CPU data
    observation = {
        OBS_STATE: torch.randn(12),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(6)
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

    # Check that action is back on CPU
    assert postprocessed.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tdmpc_processor_accelerate_scenario():
    """Test TDMPC processor in simulated Accelerate scenario."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_tdmpc_pre_post_processors(
        config,
        stats,
    )

    # Simulate Accelerate: data already on GPU
    device = torch.device("cuda:0")
    observation = {
        OBS_STATE: torch.randn(12).to(device),
        OBS_IMAGE: torch.randn(3, 224, 224).to(device),
    }
    action = torch.randn(6).to(device)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data stays on same GPU
    assert processed[OBS_STATE].device == device
    assert processed[OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_tdmpc_processor_multi_gpu():
    """Test TDMPC processor with multi-GPU setup."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_tdmpc_pre_post_processors(
        config,
        stats,
    )

    # Simulate data on different GPU
    device = torch.device("cuda:1")
    observation = {
        OBS_STATE: torch.randn(12).to(device),
        OBS_IMAGE: torch.randn(3, 224, 224).to(device),
    }
    action = torch.randn(6).to(device)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data stays on cuda:1
    assert processed[OBS_STATE].device == device
    assert processed[OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


def test_tdmpc_processor_without_stats():
    """Test TDMPC processor creation without dataset statistics."""
    config = create_default_config()

    preprocessor, postprocessor = make_tdmpc_pre_post_processors(config, dataset_stats=None)

    # Should still create processors
    assert preprocessor is not None
    assert postprocessor is not None

    # Process should still work
    observation = {
        OBS_STATE: torch.randn(12),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(6)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed is not None


def test_tdmpc_processor_save_and_load():
    """Test saving and loading TDMPC processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_tdmpc_pre_post_processors(
        config,
        stats,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save preprocessor
        preprocessor.save_pretrained(tmpdir)

        # Load preprocessor
        loaded_preprocessor = DataProcessorPipeline.from_pretrained(
            tmpdir, config_filename="policy_preprocessor.json"
        )

        # Test that loaded processor works
        observation = {
            OBS_STATE: torch.randn(12),
            OBS_IMAGE: torch.randn(3, 224, 224),
        }
        action = torch.randn(6)
        transition = create_transition(observation, action)

        batch = transition_to_batch(transition)
        processed = loaded_preprocessor(batch)
        assert processed[OBS_STATE].shape == (1, 12)
        assert processed[OBS_IMAGE].shape == (1, 3, 224, 224)
        assert processed[TransitionKey.ACTION.value].shape == (1, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tdmpc_processor_mixed_precision():
    """Test TDMPC processor with mixed precision."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    # Create processor
    preprocessor, postprocessor = make_tdmpc_pre_post_processors(
        config,
        stats,
    )

    # Replace DeviceProcessorStep with one that uses float16
    modified_steps = []
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            modified_steps.append(DeviceProcessorStep(device=config.device, float_dtype="float16"))
        elif isinstance(step, NormalizerProcessorStep):
            # Update normalizer to use the same device as the device processor
            modified_steps.append(
                NormalizerProcessorStep(
                    features=step.features,
                    norm_map=step.norm_map,
                    stats=step.stats,
                    device=config.device,
                    dtype=torch.float16,  # Match the float16 dtype
                )
            )
        else:
            modified_steps.append(step)
    preprocessor.steps = modified_steps

    # Create test data
    observation = {
        OBS_STATE: torch.randn(12, dtype=torch.float32),
        OBS_IMAGE: torch.randn(3, 224, 224, dtype=torch.float32),
    }
    action = torch.randn(6, dtype=torch.float32)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data is converted to float16
    assert processed[OBS_STATE].dtype == torch.float16
    assert processed[OBS_IMAGE].dtype == torch.float16
    assert processed[TransitionKey.ACTION.value].dtype == torch.float16


def test_tdmpc_processor_batch_data():
    """Test TDMPC processor with batched data."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_tdmpc_pre_post_processors(
        config,
        stats,
    )

    # Test with batched data
    batch_size = 64
    observation = {
        OBS_STATE: torch.randn(batch_size, 12),
        OBS_IMAGE: torch.randn(batch_size, 3, 224, 224),
    }
    action = torch.randn(batch_size, 6)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that batch dimension is preserved
    assert processed[OBS_STATE].shape == (batch_size, 12)
    assert processed[OBS_IMAGE].shape == (batch_size, 3, 224, 224)
    assert processed[TransitionKey.ACTION.value].shape == (batch_size, 6)


def test_tdmpc_processor_edge_cases():
    """Test TDMPC processor with edge cases."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_tdmpc_pre_post_processors(
        config,
        stats,
    )

    # Test with only state observation (no image)
    observation = {OBS_STATE: torch.randn(12)}
    action = torch.randn(6)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed[OBS_STATE].shape == (1, 12)
    assert OBS_IMAGE not in processed

    # Test with only image observation (no state)
    observation = {OBS_IMAGE: torch.randn(3, 224, 224)}
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed[OBS_IMAGE].shape == (1, 3, 224, 224)
    assert OBS_STATE not in processed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tdmpc_processor_bfloat16_device_float32_normalizer():
    """Test: DeviceProcessor(bfloat16) + NormalizerProcessor(float32) → output bfloat16 via automatic adaptation"""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, _ = make_tdmpc_pre_post_processors(
        config,
        stats,
    )

    # Modify the pipeline to use bfloat16 device processor with float32 normalizer
    modified_steps = []
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            # Device processor converts to bfloat16
            modified_steps.append(DeviceProcessorStep(device=config.device, float_dtype="bfloat16"))
        elif isinstance(step, NormalizerProcessorStep):
            # Normalizer stays configured as float32 (will auto-adapt to bfloat16)
            modified_steps.append(
                NormalizerProcessorStep(
                    features=step.features,
                    norm_map=step.norm_map,
                    stats=step.stats,
                    device=config.device,
                    dtype=torch.float32,  # Deliberately configured as float32
                )
            )
        else:
            modified_steps.append(step)
    preprocessor.steps = modified_steps

    # Verify initial normalizer configuration
    normalizer_step = preprocessor.steps[3]  # NormalizerProcessorStep
    assert normalizer_step.dtype == torch.float32

    # Create test data with both state and visual observations
    observation = {
        OBS_STATE: torch.randn(12, dtype=torch.float32),
        OBS_IMAGE: torch.randn(3, 224, 224, dtype=torch.float32),
    }
    action = torch.randn(6, dtype=torch.float32)
    transition = create_transition(observation, action)

    batch = transition_to_batch(transition)

    # Process through full pipeline
    processed = preprocessor(batch)

    # Verify: DeviceProcessor → bfloat16, NormalizerProcessor adapts → final output is bfloat16
    assert processed[OBS_STATE].dtype == torch.bfloat16
    assert processed[OBS_IMAGE].dtype == torch.bfloat16  # IDENTITY normalization still gets dtype conversion
    assert processed[TransitionKey.ACTION.value].dtype == torch.bfloat16

    # Verify normalizer automatically adapted its internal state
    assert normalizer_step.dtype == torch.bfloat16
    # Check state stats (has normalization)
    for stat_tensor in normalizer_step._tensor_stats[OBS_STATE].values():
        assert stat_tensor.dtype == torch.bfloat16
    # OBS_IMAGE uses IDENTITY normalization, so no stats to check
