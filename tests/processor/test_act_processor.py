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
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import make_act_pre_post_processors
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
from lerobot.utils.constants import ACTION, OBS_STATE


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


def test_act_processor_normalization():
    """Test that ACT processor correctly normalizes and unnormalizes data."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Create test data
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data is normalized and batched
    assert processed[OBS_STATE].shape == (1, 7)
    assert processed[TransitionKey.ACTION.value].shape == (1, 4)

    # Process action through postprocessor
    postprocessed = postprocessor(processed[TransitionKey.ACTION.value])

    # Check that action is unnormalized
    assert postprocessed.shape == (1, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_processor_cuda():
    """Test ACT processor with CUDA device."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Create CPU data
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data is on CUDA
    assert processed[OBS_STATE].device.type == "cuda"
    assert processed[TransitionKey.ACTION.value].device.type == "cuda"

    # Process through postprocessor
    postprocessed = postprocessor(processed[TransitionKey.ACTION.value])

    # Check that action is back on CPU
    assert postprocessed.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_processor_accelerate_scenario():
    """Test ACT processor in simulated Accelerate scenario (data already on GPU)."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Simulate Accelerate: data already on GPU
    device = torch.device("cuda:0")
    observation = {OBS_STATE: torch.randn(1, 7).to(device)}  # Already batched and on GPU
    action = torch.randn(1, 4).to(device)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data stays on same GPU (not moved unnecessarily)
    assert processed[OBS_STATE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_act_processor_multi_gpu():
    """Test ACT processor with multi-GPU setup."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Simulate data on different GPU (like in multi-GPU training)
    device = torch.device("cuda:1")
    observation = {OBS_STATE: torch.randn(1, 7).to(device)}
    action = torch.randn(1, 4).to(device)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data stays on cuda:1 (not moved to cuda:0)
    assert processed[OBS_STATE].device == device
    assert processed[TransitionKey.ACTION.value].device == device


def test_act_processor_without_stats():
    """Test ACT processor creation without dataset statistics."""
    config = create_default_config()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        dataset_stats=None,
    )

    # Should still create processors, but normalization won't have stats
    assert preprocessor is not None
    assert postprocessor is not None

    # Process should still work (but won't normalize without stats)
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed is not None


def test_act_processor_save_and_load():
    """Test saving and loading ACT processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
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
        observation = {OBS_STATE: torch.randn(7)}
        action = torch.randn(4)
        transition = create_transition(observation, action)
        batch = transition_to_batch(transition)

        processed = loaded_preprocessor(batch)
        assert processed[OBS_STATE].shape == (1, 7)
        assert processed[TransitionKey.ACTION.value].shape == (1, 4)


def test_act_processor_device_placement_preservation():
    """Test that ACT processor preserves device placement correctly."""
    config = create_default_config()
    stats = create_default_stats()

    # Test with CPU config
    config.device = "cpu"
    preprocessor, _ = make_act_pre_post_processors(
        config,
        stats,
    )

    # Process CPU data
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed[OBS_STATE].device.type == "cpu"
    assert processed[TransitionKey.ACTION.value].device.type == "cpu"


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
    )

    # Replace DeviceProcessorStep with one that uses float16
    modified_steps = []
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            modified_steps.append(DeviceProcessorStep(device=config.device, float_dtype="float16"))
        elif isinstance(step, NormalizerProcessorStep):
            # Update normalizer to use the same device as the device processor
            norm_step = step  # Now type checker knows this is NormalizerProcessorStep
            modified_steps.append(
                NormalizerProcessorStep(
                    features=norm_step.features,
                    norm_map=norm_step.norm_map,
                    stats=norm_step.stats,
                    device=config.device,
                    dtype=torch.float16,  # Match the float16 dtype
                )
            )
        else:
            modified_steps.append(step)
    preprocessor.steps = modified_steps

    # Create test data
    observation = {OBS_STATE: torch.randn(7, dtype=torch.float32)}
    action = torch.randn(4, dtype=torch.float32)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data is converted to float16
    assert processed[OBS_STATE].dtype == torch.float16
    assert processed[TransitionKey.ACTION.value].dtype == torch.float16


def test_act_processor_batch_consistency():
    """Test that ACT processor handles different batch sizes correctly."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        stats,
    )

    # Test single sample (unbatched)
    observation = {OBS_STATE: torch.randn(7)}
    action = torch.randn(4)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed[OBS_STATE].shape[0] == 1  # Batched

    # Test already batched data
    observation_batched = {OBS_STATE: torch.randn(8, 7)}  # Batch of 8
    action_batched = torch.randn(8, 4)
    transition_batched = create_transition(observation_batched, action_batched)
    batch_batched = transition_to_batch(transition_batched)

    processed_batched = preprocessor(batch_batched)
    assert processed_batched[OBS_STATE].shape[0] == 8
    assert processed_batched[TransitionKey.ACTION.value].shape[0] == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_act_processor_bfloat16_device_float32_normalizer():
    """Test: DeviceProcessor(bfloat16) + NormalizerProcessor(float32) → output bfloat16 via automatic adaptation"""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, _ = make_act_pre_post_processors(
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
            norm_step = step  # Now type checker knows this is NormalizerProcessorStep
            modified_steps.append(
                NormalizerProcessorStep(
                    features=norm_step.features,
                    norm_map=norm_step.norm_map,
                    stats=norm_step.stats,
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

    # Create test data
    observation = {OBS_STATE: torch.randn(7, dtype=torch.float32)}  # Start with float32
    action = torch.randn(4, dtype=torch.float32)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through full pipeline
    processed = preprocessor(batch)

    # Verify: DeviceProcessor → bfloat16, NormalizerProcessor adapts → final output is bfloat16
    assert processed[OBS_STATE].dtype == torch.bfloat16
    assert processed[TransitionKey.ACTION.value].dtype == torch.bfloat16

    # Verify normalizer automatically adapted its internal state
    assert normalizer_step.dtype == torch.bfloat16
    for stat_tensor in normalizer_step._tensor_stats[OBS_STATE].values():
        assert stat_tensor.dtype == torch.bfloat16
