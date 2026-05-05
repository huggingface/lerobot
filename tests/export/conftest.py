#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Shared fixtures and utilities for export tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    pass


def require_onnx(func):
    """Decorator that skips the test if onnx (the exporter package) is not available."""
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        pytest.importorskip("onnx")
        return func(*args, **kwargs)

    return wrapper


def require_onnxruntime(func):
    """Decorator that skips the test if onnxruntime is not available."""
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        pytest.importorskip("onnxruntime")
        return func(*args, **kwargs)

    return wrapper


def require_openvino(func):
    """Decorator that skips the test if openvino is not available."""
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        pytest.importorskip("openvino")
        return func(*args, **kwargs)

    return wrapper


def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def to_numpy(batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Convert a batch of PyTorch tensors to numpy arrays."""
    return {k: v.detach().cpu().numpy() for k, v in batch.items()}


def assert_numerical_parity(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    msg: str = "",
) -> None:
    """Assert two arrays are numerically close within tolerances."""
    max_diff = np.max(np.abs(actual - expected))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg=f"{msg} (max diff: {max_diff})",
    )


def _make_identity_stats(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    return {
        "mean": np.zeros(shape, dtype=np.float32),
        "std": np.ones(shape, dtype=np.float32),
        "min": -np.ones(shape, dtype=np.float32),
        "max": np.ones(shape, dtype=np.float32),
    }


def _identity_state_dict(features) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for feature_name, feature in features.items():
        shape = tuple(feature.shape)
        state[f"{feature_name}.mean"] = torch.zeros(shape, dtype=torch.float32)
        state[f"{feature_name}.std"] = torch.ones(shape, dtype=torch.float32)
        state[f"{feature_name}.min"] = -torch.ones(shape, dtype=torch.float32)
        state[f"{feature_name}.max"] = torch.ones(shape, dtype=torch.float32)
    return state


def _attach_identity_stats(policy, input_features, output_features) -> None:
    from lerobot.policies.factory import make_pre_post_processors

    state_dict = _identity_state_dict({**input_features, **output_features})
    preprocessor, postprocessor = make_pre_post_processors(policy.config)

    for step in [*preprocessor.steps, *postprocessor.steps]:
        if hasattr(step, "load_state_dict") and hasattr(step, "stats"):
            step.load_state_dict(state_dict)

    for step in preprocessor.steps:
        if hasattr(step, "stats") and step.stats:
            policy.config.stats = {
                key: {stat_name: np.asarray(value, dtype=np.float32) for stat_name, value in stats.items()}
                for key, stats in step.stats.items()
            }
            return

    policy.config.stats = {
        feature_name: _make_identity_stats(tuple(feature.shape))
        for feature_name, feature in {**input_features, **output_features}.items()
    }


def create_act_policy_and_batch(device: str = "cpu"):
    """Create a minimal ACT policy and example batch for testing."""
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }

    config = ACTConfig(
        device=device,
        chunk_size=10,
        n_action_steps=10,
        dim_model=64,
        n_heads=2,
        dim_feedforward=128,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_vae_encoder_layers=2,
        use_vae=False,
        latent_dim=16,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
    )

    policy = ACTPolicy(config)
    _attach_identity_stats(policy, input_features, output_features)
    policy.to(device)
    policy.eval()

    batch_size = 1
    batch = {
        "observation.state": torch.randn(batch_size, 6, device=device),
        "observation.images.top": torch.randn(batch_size, 3, 84, 84, device=device),
    }

    return policy, batch


def create_pi05_policy_and_batch(device: str = "cuda"):
    """Create a PI05 policy and example batch for testing.

    PI05 does not use state input during inference.
    """
    pytest.importorskip("transformers")
    skip_if_no_cuda()

    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
    }

    config = PI05Config(
        device=device,
        chunk_size=10,
        n_action_steps=10,
        max_state_dim=32,
        max_action_dim=32,
        num_inference_steps=3,
        tokenizer_max_length=48,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        dtype="float32",
    )

    policy = PI05Policy(config)
    _attach_identity_stats(policy, input_features, output_features)
    policy.to(device)
    policy.eval()

    batch_size = 1
    batch = {
        OBS_STATE: torch.randn(batch_size, 14, device=device),
        "observation.images.top": torch.rand(batch_size, 3, 224, 224, device=device),
        OBS_LANGUAGE_TOKENS: torch.ones(batch_size, 48, dtype=torch.long, device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 48, dtype=torch.bool, device=device),
    }

    return policy, batch


def load_cached_paligemma_tokenizer():
    """Load the PI05 tokenizer from local Hugging Face cache or skip.

    Export tests use a real tokenizer roundtrip but must stay CI-portable when
    the cache is not pre-populated.
    """
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", local_files_only=True)
    except OSError:
        pytest.skip("paligemma tokenizer not in local HF cache; skipping for CI portability")
