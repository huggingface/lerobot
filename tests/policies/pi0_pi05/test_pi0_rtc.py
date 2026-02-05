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

"""Test PI0 policy with Real-Time Chunking (RTC) enabled during inference."""

import os

import pytest
import torch

# Skip this entire module in CI
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires local OpenPI installation and is not meant for CI",
)

from lerobot.configs.types import FeatureType, PolicyFeature, RTCAttentionSchedule  # noqa: E402
from lerobot.policies.pi0 import PI0Config, PI0Policy, make_pi0_pre_post_processors  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from tests.utils import require_cuda  # noqa: E402


@require_cuda
def test_pi0_rtc_initialization():
    """Test PI0 policy can initialize RTC processor."""
    set_seed(42)

    config = PI0Config(max_action_dim=7, max_state_dim=14, dtype="float32")

    # Add RTC config
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=10,
        max_guidance_weight=5.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        debug=False,
    )

    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    # Instantiate policy
    policy = PI0Policy(config)

    # Verify RTC processor is initialized
    assert hasattr(policy, "rtc_processor")
    assert policy.rtc_processor is not None
    assert policy.rtc_processor.rtc_config.enabled is True


@require_cuda
def test_pi0_rtc_initialization_without_rtc_config():
    """Test PI0 policy can initialize without RTC config."""
    set_seed(42)

    config = PI0Config(max_action_dim=7, max_state_dim=14, dtype="float32")

    # Instantiate policy
    policy = PI0Policy(config)

    # Verify RTC processor is not initialized
    assert hasattr(policy, "rtc_processor")
    assert policy.rtc_processor is None
    assert policy.model.rtc_processor is None
    assert policy._rtc_enabled() is False


@require_cuda
def test_pi0_rtc_alex_soare_optimization():
    """Test PI0 with Alex Soare optimization (max_guidance_weight=None, uses num_inference_steps during denoise_step)."""
    set_seed(42)

    config = PI0Config(
        max_action_dim=7,
        max_state_dim=14,
        dtype="float32",
        num_inference_steps=20,  # This will be passed to denoise_step
    )

    # Add RTC config WITHOUT max_guidance_weight (optimization happens in denoise_step)
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=10,
        max_guidance_weight=None,  # Not provided - optimization happens during denoise_step
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        debug=False,
    )

    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    # Instantiate policy
    policy = PI0Policy(config)

    # Verify RTC processor has max_guidance_weight still None (optimization happens in denoise_step)
    assert policy.rtc_processor is not None
    assert policy.rtc_processor.rtc_config.max_guidance_weight is None


@require_cuda
def test_pi0_rtc_explicit_max_guidance_weight():
    """Test PI0 respects explicit max_guidance_weight when provided."""
    set_seed(42)

    config = PI0Config(
        max_action_dim=7,
        max_state_dim=14,
        dtype="float32",
        num_inference_steps=20,
    )

    # Add RTC config WITH explicit max_guidance_weight
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=10,
        max_guidance_weight=5.0,  # Explicitly set - should NOT be overridden
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        debug=False,
    )

    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    # Instantiate policy
    policy = PI0Policy(config)

    # Verify RTC processor keeps the explicit max_guidance_weight
    assert policy.rtc_processor is not None
    assert policy.rtc_processor.rtc_config.max_guidance_weight == 5.0
    assert policy.rtc_processor.rtc_config.max_guidance_weight != config.num_inference_steps


@require_cuda
def test_pi0_rtc_inference_with_different_sigma_d_and_auto_guidance():
    """Test PI0 inference with different sigma_d values using Alex Soare optimization."""
    set_seed(42)

    config = PI0Config(
        max_action_dim=7,
        max_state_dim=14,
        chunk_size=50,
        dtype="float32",
        num_inference_steps=10,  # Will be used as max_guidance_weight
    )

    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    # Create dataset stats
    dataset_stats = {
        "observation.state": {"mean": torch.zeros(14), "std": torch.ones(14)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "observation.images.base_0_rgb": {"mean": torch.zeros(3, 224, 224), "std": torch.ones(3, 224, 224)},
    }

    # Test with sigma_d = 0.2 (stronger guidance)
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=10,
        max_guidance_weight=None,  # Use Alex Soare optimization
        sigma_d=0.2,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        debug=False,
    )

    policy1 = PI0Policy(config)
    policy1.eval()
    preprocessor, _ = make_pi0_pre_post_processors(config=config, dataset_stats=dataset_stats)

    # Verify max_guidance_weight was auto-set
    assert policy1.rtc_processor.rtc_config.max_guidance_weight is None
    assert policy1.rtc_processor.rtc_config.sigma_d == 0.2

    device = config.device

    # Create dummy batch
    batch = {
        "observation.state": torch.randn(1, 14, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device),
        "task": ["Pick up the object"],
    }
    batch = preprocessor(batch)

    # Create previous chunk
    prev_chunk = torch.randn(1, 25, 7, dtype=torch.float32, device=device)

    with torch.no_grad():
        noise = policy1.model.sample_noise((1, config.chunk_size, 7), device)
        actions_sigma_02 = policy1.predict_action_chunk(
            batch,
            noise=noise.clone(),
            prev_chunk_left_over=prev_chunk,
            inference_delay=4,
            execution_horizon=10,
        )

    # Now test with sigma_d = 1.0 (weaker guidance)
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=10,
        max_guidance_weight=None,  # Use Alex Soare optimization
        sigma_d=1.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        debug=False,
    )

    policy2 = PI0Policy(config)
    policy2.eval()

    # Verify max_guidance_weight was auto-set and sigma_d is different
    assert policy2.rtc_processor.rtc_config.max_guidance_weight is None
    assert policy2.rtc_processor.rtc_config.sigma_d == 1.0

    with torch.no_grad():
        actions_sigma_10 = policy2.predict_action_chunk(
            batch,
            noise=noise.clone(),
            prev_chunk_left_over=prev_chunk,
            inference_delay=4,
            execution_horizon=10,
        )

    # Verify shapes
    assert actions_sigma_02.shape == (1, config.chunk_size, 7)
    assert actions_sigma_10.shape == (1, config.chunk_size, 7)

    # Different sigma_d values should produce different results
    assert not torch.allclose(actions_sigma_02, actions_sigma_10, rtol=1e-3)


def test_pi0_rtc_inference_with_prev_chunk():
    """Test PI0 policy inference with RTC and previous chunk."""
    set_seed(42)

    config = PI0Config(max_action_dim=7, max_state_dim=14, chunk_size=50, dtype="float32")

    # Add RTC config
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=10,
        max_guidance_weight=5.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        debug=False,
    )

    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    # Create dataset stats
    dataset_stats = {
        "observation.state": {"mean": torch.zeros(14), "std": torch.ones(14)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "observation.images.base_0_rgb": {"mean": torch.zeros(3, 224, 224), "std": torch.ones(3, 224, 224)},
    }

    # Instantiate policy and preprocessor
    policy = PI0Policy(config)
    policy.eval()
    preprocessor, _ = make_pi0_pre_post_processors(config=config, dataset_stats=dataset_stats)

    device = config.device

    # Create dummy batch
    batch = {
        "observation.state": torch.randn(1, 14, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device),
        "task": ["Pick up the object"],
    }
    batch = preprocessor(batch)

    # Create previous chunk
    prev_chunk = torch.randn(1, 25, 7, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Use same noise for fair comparison
        noise = policy.model.sample_noise((1, config.chunk_size, 7), device)

        # Test with RTC and previous chunk
        actions_with_rtc = policy.predict_action_chunk(
            batch,
            noise=noise.clone(),
            prev_chunk_left_over=prev_chunk,
            inference_delay=4,
            execution_horizon=10,
        )

        # Test without RTC for comparison
        policy.config.rtc_config.enabled = False
        actions_without_rtc = policy.predict_action_chunk(batch, noise=noise.clone())
        policy.config.rtc_config.enabled = True

    # Verify shapes
    assert actions_with_rtc.shape == (1, config.chunk_size, 7)
    assert actions_without_rtc.shape == (1, config.chunk_size, 7)

    # With previous chunk, actions should be different (RTC guidance applied)
    assert not torch.allclose(actions_with_rtc, actions_without_rtc, rtol=1e-3)


@require_cuda
def test_pi0_rtc_inference_without_prev_chunk():
    """Test PI0 policy inference with RTC but no previous chunk (RTC should have no effect)."""
    set_seed(42)

    config = PI0Config(max_action_dim=7, max_state_dim=14, chunk_size=50, dtype="float32")

    # Add RTC config
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=10,
        max_guidance_weight=5.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        debug=False,
    )

    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    # Create dataset stats
    dataset_stats = {
        "observation.state": {"mean": torch.zeros(14), "std": torch.ones(14)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "observation.images.base_0_rgb": {"mean": torch.zeros(3, 224, 224), "std": torch.ones(3, 224, 224)},
    }

    # Instantiate policy and preprocessor
    policy = PI0Policy(config)
    policy.eval()
    preprocessor, _ = make_pi0_pre_post_processors(config=config, dataset_stats=dataset_stats)

    device = config.device

    # Create dummy batch
    batch = {
        "observation.state": torch.randn(1, 14, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device),
        "task": ["Pick up the object"],
    }
    batch = preprocessor(batch)

    with torch.no_grad():
        # Use same noise for fair comparison
        noise = policy.model.sample_noise((1, config.chunk_size, 7), device)

        # Test with RTC enabled but no previous chunk
        actions_with_rtc_no_prev = policy.predict_action_chunk(
            batch,
            noise=noise.clone(),
            prev_chunk_left_over=None,
        )

        # Test without RTC
        policy.config.rtc_config.enabled = False
        actions_without_rtc = policy.predict_action_chunk(batch, noise=noise.clone())
        policy.config.rtc_config.enabled = True

    # Without previous chunk, RTC should have no effect
    assert torch.allclose(actions_with_rtc_no_prev, actions_without_rtc, rtol=1e-5)


@require_cuda
def test_pi0_rtc_validation_rules():
    """Test PI0 policy with RTC follows all three validation rules."""
    set_seed(42)

    config = PI0Config(max_action_dim=7, max_state_dim=14, chunk_size=50, dtype="float32")

    # Add RTC config
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=10,
        max_guidance_weight=5.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        debug=False,
    )

    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    # Create dataset stats
    dataset_stats = {
        "observation.state": {"mean": torch.zeros(14), "std": torch.ones(14)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "observation.images.base_0_rgb": {"mean": torch.zeros(3, 224, 224), "std": torch.ones(3, 224, 224)},
    }

    # Instantiate policy and preprocessor
    policy = PI0Policy(config)
    policy.eval()
    preprocessor, _ = make_pi0_pre_post_processors(config=config, dataset_stats=dataset_stats)

    device = config.device

    # Create dummy batch
    batch = {
        "observation.state": torch.randn(1, 14, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device),
        "task": ["Pick up the object"],
    }
    batch = preprocessor(batch)

    # Create previous chunk
    prev_chunk = torch.randn(1, 25, 7, dtype=torch.float32, device=device)

    inference_delay = 4
    execution_horizon = 10

    with torch.no_grad():
        # Use same noise for fair comparison
        noise = policy.model.sample_noise((1, config.chunk_size, 7), device)

        # Test with RTC
        actions_with_rtc = policy.predict_action_chunk(
            batch,
            noise=noise.clone(),
            prev_chunk_left_over=prev_chunk,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )

        # Test without RTC
        policy.config.rtc_config.enabled = False
        actions_without_rtc = policy.predict_action_chunk(batch, noise=noise.clone())
        policy.config.rtc_config.enabled = True

    assert not torch.allclose(actions_with_rtc, actions_without_rtc, rtol=1e-3)

    """Test PI0 with different RTC attention schedules."""
    set_seed(42)

    schedules = [
        RTCAttentionSchedule.ZEROS,
        RTCAttentionSchedule.ONES,
        RTCAttentionSchedule.LINEAR,
        RTCAttentionSchedule.EXP,
    ]

    config = PI0Config(max_action_dim=7, max_state_dim=14, chunk_size=50, dtype="float32")

    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    # Create dataset stats
    dataset_stats = {
        "observation.state": {"mean": torch.zeros(14), "std": torch.ones(14)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "observation.images.base_0_rgb": {"mean": torch.zeros(3, 224, 224), "std": torch.ones(3, 224, 224)},
    }

    device = config.device

    for schedule in schedules:
        # Add RTC config with specific schedule
        config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=10,
            max_guidance_weight=5.0,
            prefix_attention_schedule=schedule,
            debug=False,
        )

        # Instantiate policy
        policy = PI0Policy(config)
        policy.eval()
        preprocessor, _ = make_pi0_pre_post_processors(config=config, dataset_stats=dataset_stats)

        # Create dummy batch
        batch = {
            "observation.state": torch.randn(1, 14, dtype=torch.float32, device=device),
            "observation.images.base_0_rgb": torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device),
            "task": ["Pick up the object"],
        }
        batch = preprocessor(batch)

        # Create previous chunk
        prev_chunk = torch.randn(1, 25, 7, dtype=torch.float32, device=device)

        with torch.no_grad():
            noise = policy.model.sample_noise((1, config.chunk_size, 7), device)
            actions = policy.predict_action_chunk(
                batch,
                noise=noise,
                prev_chunk_left_over=prev_chunk,
                inference_delay=4,
                execution_horizon=10,
            )

        # Verify shape
        assert actions.shape == (1, config.chunk_size, 7)
