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


def require_diffusers(func):
    """Decorator that skips the test if diffusers is not available."""
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        pytest.importorskip("diffusers")
        return func(*args, **kwargs)

    return wrapper


def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def skip_if_pi0_transformers_unavailable():
    """Skip test if PI0-compatible transformers is not available."""
    transformers = pytest.importorskip("transformers")
    try:
        import importlib

        check = importlib.import_module("transformers.models.siglip.check")

        if not check.check_whether_transformers_replace_is_installed_correctly():
            pytest.skip(
                "PI0 requires a patched Transformers build (SigLIP replace hooks missing). "
                "Install via: pip install 'lerobot[pi]' or "
                "pip install transformers@git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi "
                f"(detected transformers={transformers.__version__})"
            )
    except Exception as e:
        pytest.skip(
            "PI0 requires a patched Transformers build (SigLIP replace hooks missing). "
            "Install via: pip install 'lerobot[pi]' or "
            "pip install transformers@git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi "
            f"(detected transformers={transformers.__version__}, err={type(e).__name__}: {e})"
        )


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
    policy.to(device)
    policy.eval()

    batch_size = 1
    batch = {
        "observation.state": torch.randn(batch_size, 6, device=device),
        "observation.images.top": torch.randn(batch_size, 3, 84, 84, device=device),
    }

    return policy, batch


def create_diffusion_policy_and_batch(device: str = "cpu"):
    """Create a minimal Diffusion policy and example batch for testing."""
    pytest.importorskip("diffusers")

    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }

    config = DiffusionConfig(
        device=device,
        n_action_steps=8,
        horizon=8,
        n_obs_steps=2,
        num_inference_steps=5,
        down_dims=(64, 128),
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        noise_scheduler_type="DDIM",
    )

    policy = DiffusionPolicy(config)
    policy.to(device)
    policy.eval()

    batch_size = 1
    n_obs_steps = config.n_obs_steps
    batch = {
        "observation.state": torch.randn(batch_size, n_obs_steps, 6, device=device),
        "observation.images.top": torch.randn(batch_size, n_obs_steps, 3, 84, 84, device=device),
    }

    return policy, batch


def create_smolvla_policy_and_batch(device: str = "cuda"):
    """Create a SmolVLA policy and example batch for testing."""
    pytest.importorskip("transformers")
    skip_if_no_cuda()

    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
    }

    config = SmolVLAConfig(
        device=device,
        chunk_size=10,
        n_action_steps=10,
        max_state_dim=32,
        max_action_dim=32,
        num_steps=3,
        tokenizer_max_length=48,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        freeze_vision_encoder=True,
        train_expert_only=True,
    )

    policy = SmolVLAPolicy(config)
    policy.to(device)
    policy.eval()

    batch_size = 1
    batch = {
        OBS_STATE: torch.randn(batch_size, 14, device=device),
        "observation.images.top": torch.rand(batch_size, 3, 512, 512, device=device),
        OBS_LANGUAGE_TOKENS: torch.ones(batch_size, 48, dtype=torch.long, device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 48, dtype=torch.bool, device=device),
    }

    return policy, batch


def create_pi0_policy_and_batch(device: str = "cuda"):
    """Create a PI0 policy and example batch for testing."""
    skip_if_pi0_transformers_unavailable()
    skip_if_no_cuda()

    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
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

    config = PI0Config(
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
        freeze_vision_encoder=True,
        train_expert_only=True,
        dtype="float32",
    )

    policy = PI0Policy(config)
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
