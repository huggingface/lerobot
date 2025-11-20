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

"""Test PI0.5 policy with Real-Time Chunking (RTC) enabled during inference."""

import os

import pytest
import torch

# Skip this entire module in CI
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires local OpenPI installation and is not meant for CI",
)

from lerobot.configs.types import FeatureType, PolicyFeature, RTCAttentionSchedule  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy, make_pi05_pre_post_processors  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from tests.utils import require_cuda  # noqa: E402


@require_cuda
def test_pi05_rtc_initialization():
    """Test PI0.5 policy can initialize RTC processor."""
    set_seed(42)

    config = PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32")

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
    policy = PI05Policy(config)

    # Verify RTC processor is initialized
    assert hasattr(policy, "rtc_processor")
    assert policy.rtc_processor is not None
    assert policy.rtc_processor.rtc_config.enabled is True

    print("✓ PI0.5 RTC initialization: Test passed")


@require_cuda
def test_pi05_rtc_initialization_without_rtc_config():
    """Test PI0.5 policy can initialize without RTC config."""
    set_seed(42)

    config = PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32")

    # Instantiate policy
    policy = PI05Policy(config)

    # Verify RTC processor is not initialized
    assert hasattr(policy, "rtc_processor")
    assert policy.rtc_processor is None
    assert policy.model.rtc_processor is None
    assert policy._rtc_enabled() is False

    print("✓ PI0.5 RTC initialization without RTC config: Test passed")


@require_cuda
def test_pi05_rtc_inference_with_prev_chunk():
    """Test PI0.5 policy inference with RTC and previous chunk."""
    set_seed(42)

    config = PI05Config(max_action_dim=7, max_state_dim=14, chunk_size=50, dtype="float32")

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

    # Create dataset stats (PI0.5 uses QUANTILES normalization)
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "q01": -torch.ones(14),
            "q99": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "q01": -torch.ones(7),
            "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {"mean": torch.zeros(3, 224, 224), "std": torch.ones(3, 224, 224)},
    }

    # Instantiate policy and preprocessor
    policy = PI05Policy(config)
    policy.eval()
    preprocessor, _ = make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)

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

    print("✓ PI0.5 RTC inference with prev_chunk: Test passed")


@require_cuda
def test_pi05_rtc_inference_without_prev_chunk():
    """Test PI0.5 policy inference with RTC but no previous chunk (RTC should have no effect)."""
    set_seed(42)

    config = PI05Config(max_action_dim=7, max_state_dim=14, chunk_size=50, dtype="float32")

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

    # Create dataset stats (PI0.5 uses QUANTILES normalization)
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "q01": -torch.ones(14),
            "q99": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "q01": -torch.ones(7),
            "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {"mean": torch.zeros(3, 224, 224), "std": torch.ones(3, 224, 224)},
    }

    # Instantiate policy and preprocessor
    policy = PI05Policy(config)
    policy.eval()
    preprocessor, _ = make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)

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

    print("✓ PI0.5 RTC inference without prev_chunk: Test passed")


@require_cuda
def test_pi05_rtc_validation_rules():
    """Test PI0.5 policy with RTC follows all three validation rules."""
    set_seed(42)

    config = PI05Config(max_action_dim=7, max_state_dim=14, chunk_size=50, dtype="float32")

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

    # Create dataset stats (PI0.5 uses QUANTILES normalization)
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "q01": -torch.ones(14),
            "q99": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "q01": -torch.ones(7),
            "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {"mean": torch.zeros(3, 224, 224), "std": torch.ones(3, 224, 224)},
    }

    # Instantiate policy and preprocessor
    policy = PI05Policy(config)
    policy.eval()
    preprocessor, _ = make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)

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
