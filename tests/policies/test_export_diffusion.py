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
"""Per-policy ONNX export tests for Diffusion Policy.

Run with:
    uv run pytest tests/policies/test_export_diffusion.py -svv
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# Skip the whole module if export extras or diffusers are missing.
pytest.importorskip("onnx", reason="onnx not installed — skip Diffusion export tests")
pytest.importorskip("onnxruntime", reason="onnxruntime not installed — skip Diffusion export tests")
pytest.importorskip("diffusers", reason="diffusers required for DiffusionPolicy")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.export.core import make_export_wrapper  # noqa: E402
from lerobot.export.onnx_export import export_to_onnx  # noqa: E402
from lerobot.export.validation import validate_onnx  # noqa: E402
from lerobot.policies.diffusion.export_diffusion import (  # noqa: E402
    DiffusionUNetWrapper,
    _get_global_cond_dim,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402

_STATE_DIM = 6
_ACTION_DIM = 6
_IMG_C, _IMG_H, _IMG_W = 3, 96, 96


def _make_diffusion_policy():
    """Build a fresh Diffusion policy from a minimal toy config (no pretrained weights).

    DiffusionConfig.validate_features requires at least one image or env_state,
    so we always include a single tiny image feature.
    """
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    cfg = DiffusionConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,)),
            f"{OBS_IMAGES}.cam": PolicyFeature(
                type=FeatureType.VISUAL, shape=(_IMG_C, _IMG_H, _IMG_W)
            ),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))},
        horizon=16,
        n_obs_steps=2,
        n_action_steps=8,
        num_train_timesteps=100,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=2,
    )
    policy = DiffusionPolicy(cfg)
    policy.eval()
    return policy


class _Cfg:
    device = "cpu"
    batch_size = 1
    diffusion_mode = "unet-only"


def test_diffusion_unet_wrapper_forward():
    """DiffusionUNetWrapper.forward returns (B, horizon, action_dim)."""
    policy = _make_diffusion_policy()
    cfg = policy.config
    action_dim = cfg.action_feature.shape[0]

    raw_unet = getattr(policy.diffusion.unet, "_orig_mod", policy.diffusion.unet)
    wrapper = DiffusionUNetWrapper(raw_unet)
    wrapper.eval()

    horizon = cfg.horizon
    gcond_dim = _get_global_cond_dim(policy)
    sample = torch.zeros(1, horizon, action_dim)
    timestep = torch.zeros(1, dtype=torch.long)
    global_cond = torch.zeros(1, gcond_dim)

    with torch.no_grad():
        out = wrapper(sample, timestep, global_cond)
    assert out.shape == (1, horizon, action_dim)


def test_diffusion_auto_discovery_via_make_export_wrapper():
    """make_export_wrapper finds policies/diffusion/export_diffusion.py via convention."""
    policy = _make_diffusion_policy()
    wrapper, spec = make_export_wrapper(policy, _Cfg())
    assert isinstance(wrapper, DiffusionUNetWrapper)
    assert spec.input_names == ["sample", "timestep", "global_cond"]
    assert spec.output_names == ["denoised"]


def test_diffusion_unet_export_spec_dynamic_shapes():
    """ExportSpec for unet-only mode has tuple-form dynamic_shapes (3 entries)."""
    policy = _make_diffusion_policy()
    _, spec = make_export_wrapper(policy, _Cfg())
    assert spec.dynamic_shapes is not None
    assert len(spec.dynamic_shapes) == 3  # (sample, timestep, global_cond)
    for shape_spec in spec.dynamic_shapes:
        assert 0 in shape_spec


def test_onnx_export_diffusion_unet(tmp_path: Path):
    """End-to-end: Diffusion UNet → ONNX → ORT parity check."""
    policy = _make_diffusion_policy()
    wrapper, spec = make_export_wrapper(policy, _Cfg())
    wrapper.eval()

    onnx_path = export_to_onnx(
        wrapper=wrapper,
        spec=spec,
        output_path=tmp_path / "diffusion_unet_fp32",
        opset_version=18,
        precision="fp32",
    )
    assert onnx_path.exists()

    results = validate_onnx(
        wrapper=wrapper,
        sample_inputs=spec.sample_inputs,
        onnx_path=onnx_path,
        rtol=1e-3,
        atol=1e-4,
    )
    assert results["allclose"], (
        f"DiffusionUNet ONNX parity failed: max_abs_error={results['max_abs_error']:.2e}"
    )
