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
"""Unit tests for lerobot.export (CPU, toy configs, no pretrained weights).

Run with:
    uv run pytest tests/test_export.py -svv
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

# Skip the entire module when optional export dependencies are missing.
onnx = pytest.importorskip("onnx", reason="onnx not installed — skip export tests")
ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed — skip export tests")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.export.core import ExportSpec, make_export_wrapper, register_export_wrapper  # noqa: E402
from lerobot.export.normalization import NormalizedWrapper, save_normalization_stats  # noqa: E402
from lerobot.export.onnx_export import export_to_onnx  # noqa: E402
from lerobot.export.sample_inputs import make_sample_inputs  # noqa: E402
from lerobot.export.tensorrt_export import get_gpu_compute_capability  # noqa: E402
from lerobot.export.validation import validate_onnx  # noqa: E402
from lerobot.export.wrappers import (  # noqa: E402
    ACTInferenceWrapper,
    DiffusionUNetWrapper,
    GenericPolicyWrapper,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Shared toy feature definitions
# ──────────────────────────────────────────────────────────────────────────────

_STATE_DIM = 6
_ACTION_DIM = 6
_CHUNK_SIZE = 10
_IMG_C, _IMG_H, _IMG_W = 3, 64, 64


def _make_act_policy():
    """Return a fresh ACT policy built from a minimal toy config (no pretrained weights)."""
    pytest.importorskip("torchvision", reason="torchvision required for ACT backbone")

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,)),
        f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(_IMG_C, _IMG_H, _IMG_W)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,)),
    }
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=_CHUNK_SIZE,
        n_action_steps=_CHUNK_SIZE,
        dim_model=64,
        n_heads=2,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
    )
    policy = ACTPolicy(cfg)
    policy.eval()
    return policy


def _make_diffusion_policy():
    """Return a minimal Diffusion policy (no pretrained weights)."""
    pytest.importorskip("diffusers", reason="diffusers required for DiffusionPolicy")

    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,)),
    }
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
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


# ──────────────────────────────────────────────────────────────────────────────
# ExportSpec
# ──────────────────────────────────────────────────────────────────────────────


def test_export_spec_defaults():
    spec = ExportSpec(
        input_names=["a", "b"],
        output_names=["out"],
        sample_inputs=(torch.zeros(1, 4),),
    )
    assert spec.dynamic_axes is None
    assert spec.policy_note == ""


# ──────────────────────────────────────────────────────────────────────────────
# register_export_wrapper / make_export_wrapper
# ──────────────────────────────────────────────────────────────────────────────


def test_register_export_wrapper_decorator():
    from torch import nn

    class _DummyConfig:
        type = "_test_dummy"
        input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,))}
        n_obs_steps = 1

    class _DummyPolicy:
        config = _DummyConfig()
        model = nn.Linear(4, 2)

    called = {}

    @register_export_wrapper("_test_dummy")
    def _make_dummy(policy, cfg):
        called["yes"] = True
        inputs = (torch.zeros(1, 4),)
        return policy.model, ExportSpec(
            input_names=["x"],
            output_names=["y"],
            sample_inputs=inputs,
            dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}},
        )

    dummy = _DummyPolicy()
    wrapper, spec = make_export_wrapper(dummy, object())
    assert called.get("yes")
    assert spec.input_names == ["x"]


# ──────────────────────────────────────────────────────────────────────────────
# ACT wrapper
# ──────────────────────────────────────────────────────────────────────────────


def test_act_wrapper_forward():
    """ACTInferenceWrapper.forward must return (1, chunk_size, action_dim)."""
    pytest.importorskip("torchvision", reason="torchvision required for ACT backbone")

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    # ACT requires at least one image or env_state in addition to robot_state.
    input_features: dict[str, PolicyFeature] = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,)),
        f"{OBS_IMAGES}.cam": PolicyFeature(
            type=FeatureType.VISUAL, shape=(_IMG_C, _IMG_H, _IMG_W)
        ),
    }
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))}
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=_CHUNK_SIZE,
        n_action_steps=_CHUNK_SIZE,
        dim_model=64,
        n_heads=2,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
    )
    policy = ACTPolicy(cfg)
    policy.eval()

    wrapper = ACTInferenceWrapper(policy.model, cfg)
    wrapper.eval()

    state = torch.zeros(1, _STATE_DIM)
    with torch.no_grad():
        out = wrapper(state, torch.zeros(1, _IMG_C, _IMG_H, _IMG_W))

    assert out.shape == (1, _CHUNK_SIZE, _ACTION_DIM)


# ──────────────────────────────────────────────────────────────────────────────
# Diffusion UNet wrapper forward
# ──────────────────────────────────────────────────────────────────────────────


def test_diffusion_unet_wrapper_forward():
    pytest.importorskip("diffusers", reason="diffusers required")
    policy = _make_diffusion_policy()
    cfg = policy.config
    action_dim = cfg.action_feature.shape[0]

    raw_unet = getattr(policy.diffusion.unet, "_orig_mod", policy.diffusion.unet)
    wrapper = DiffusionUNetWrapper(raw_unet)
    wrapper.eval()

    horizon = cfg.horizon
    # Compute global_cond_dim by running a dummy batch through _prepare_global_conditioning
    from lerobot.export.sample_inputs import _get_diffusion_global_cond_dim

    gcond_dim = _get_diffusion_global_cond_dim(policy)

    sample = torch.zeros(1, horizon, action_dim)
    timestep = torch.zeros(1, dtype=torch.long)
    global_cond = torch.zeros(1, gcond_dim)

    with torch.no_grad():
        out = wrapper(sample, timestep, global_cond)

    assert out.shape == (1, horizon, action_dim)


# ──────────────────────────────────────────────────────────────────────────────
# ONNX export + validation (ACT, state-only to avoid torchvision dep)
# ──────────────────────────────────────────────────────────────────────────────


def test_onnx_export_act_state_only(tmp_path: Path):
    """Export ACT (state only, no cameras) to ONNX and validate numerically."""
    pytest.importorskip("torchvision", reason="torchvision required for ACT backbone")

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,)),
        # env_state as fallback — actually ACT requires at least one image or env_state
        # We provide an image here:
        f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(_IMG_C, _IMG_H, _IMG_W)),
    }
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))}
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=_CHUNK_SIZE,
        n_action_steps=_CHUNK_SIZE,
        dim_model=64,
        n_heads=2,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
    )
    policy = ACTPolicy(cfg)
    policy.eval()

    class _Cfg:
        device = "cpu"
        batch_size = 1
        diffusion_mode = "unet-only"

    wrapper, spec = make_export_wrapper(policy, _Cfg())
    wrapper.eval()

    onnx_path = export_to_onnx(
        wrapper=wrapper,
        spec=spec,
        output_path=tmp_path / "act_fp32",
        opset_version=18,
        precision="fp32",
    )

    assert onnx_path.exists()
    # Check structural validity (onnx.checker is called inside export_to_onnx)

    # Numerical validation
    results = validate_onnx(
        wrapper=wrapper,
        sample_inputs=spec.sample_inputs,
        onnx_path=onnx_path,
        rtol=1e-3,
        atol=1e-4,
    )
    assert results["allclose"], (
        f"ACT ONNX parity failed: max_abs_error={results['max_abs_error']:.2e}, "
        f"cos_sim={results['cos_sim']:.6f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Diffusion UNet ONNX export
# ──────────────────────────────────────────────────────────────────────────────


def test_onnx_export_diffusion_unet(tmp_path: Path):
    """Export DiffusionUNetWrapper to ONNX."""
    pytest.importorskip("diffusers", reason="diffusers required")
    policy = _make_diffusion_policy()

    class _Cfg:
        device = "cpu"
        batch_size = 1
        diffusion_mode = "unet-only"

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


# ──────────────────────────────────────────────────────────────────────────────
# Normalization stats
# ──────────────────────────────────────────────────────────────────────────────


def test_save_normalization_stats_no_preprocessor(tmp_path: Path):
    """save_normalization_stats writes empty JSON when both pipelines are None."""
    out_path = save_normalization_stats(None, None, tmp_path)
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data == {}


def test_save_normalization_stats_roundtrip(tmp_path: Path):
    """Stats from a real preprocessor pipeline round-trip through JSON."""
    pytest.importorskip("torchvision", reason="torchvision required for ACT backbone")

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    cfg = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,)),
            f"{OBS_IMAGES}.cam": PolicyFeature(
                type=FeatureType.VISUAL, shape=(_IMG_C, _IMG_H, _IMG_W)
            ),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))},
        chunk_size=_CHUNK_SIZE,
        n_action_steps=_CHUNK_SIZE,
        dim_model=64,
        n_heads=2,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
    )

    dataset_stats = {
        OBS_STATE: {
            "mean": torch.zeros(_STATE_DIM),
            "std": torch.ones(_STATE_DIM),
        },
        ACTION: {
            "mean": torch.full((_ACTION_DIM,), 0.5),
            "std": torch.full((_ACTION_DIM,), 2.0),
        },
    }
    preprocessor, postprocessor = make_act_pre_post_processors(cfg, dataset_stats=dataset_stats)

    out_path = save_normalization_stats(preprocessor, postprocessor, tmp_path)
    data = json.loads(out_path.read_text())

    # Both observation.state and action stats should be present.
    assert OBS_STATE in data, f"missing {OBS_STATE} in {list(data)}"
    assert ACTION in data, f"missing {ACTION} in {list(data)}"
    assert "mean" in data[OBS_STATE] and "std" in data[OBS_STATE]
    assert "mean" in data[ACTION] and "std" in data[ACTION]
    assert data[ACTION]["mean"] == [0.5] * _ACTION_DIM
    assert data[ACTION]["std"] == [2.0] * _ACTION_DIM


def test_normalized_wrapper_forward():
    """NormalizedWrapper normalizes inputs and denormalizes outputs."""
    from torch import nn

    state_dim = 4
    action_dim = 3

    class _Identity(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :action_dim]

    mean_in = torch.zeros(state_dim)
    std_in = torch.ones(state_dim)
    mean_out = torch.zeros(action_dim)
    std_out = torch.ones(action_dim) * 2.0

    input_stats = {
        "observation.state": {"mean": mean_in, "std": std_in},
    }
    action_stats = {"mean": mean_out, "std": std_out}

    wrapper = NormalizedWrapper(_Identity(), input_stats, action_stats)
    wrapper.eval()

    x = torch.ones(1, state_dim)
    with torch.no_grad():
        out = wrapper(x)

    # After normalization: x_norm = (1 - 0) / (1 + 1e-8) ≈ 1.0
    # _Identity returns first action_dim elements → [1, 1, 1]
    # Denormalize: out = 1 * 2 + 0 = 2
    assert out.shape == (1, action_dim)
    assert torch.allclose(out, torch.full((1, action_dim), 2.0), atol=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
# TensorRT (import-only check — no CUDA required)
# ──────────────────────────────────────────────────────────────────────────────


def test_get_gpu_compute_capability_no_cuda():
    """get_gpu_compute_capability returns None when CUDA is not available."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available — skipping no-CUDA path test")
    assert get_gpu_compute_capability() is None


def test_export_to_tensorrt_raises_without_cuda(tmp_path: Path):
    """export_to_tensorrt raises RuntimeError when CUDA is not available."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available — skipping no-CUDA test")

    from lerobot.export.tensorrt_export import export_to_tensorrt

    with pytest.raises(RuntimeError, match="CUDA"):
        export_to_tensorrt(
            onnx_path=tmp_path / "dummy.onnx",
            output_path=tmp_path,
            precision="fp16",
        )


# ──────────────────────────────────────────────────────────────────────────────
# make_sample_inputs
# ──────────────────────────────────────────────────────────────────────────────


def test_make_sample_inputs_act():
    pytest.importorskip("torchvision", reason="torchvision required")
    policy = _make_act_policy()

    class _Cfg:
        device = "cpu"
        batch_size = 1

    inputs = make_sample_inputs(policy, _Cfg(), mode="act")
    # First tensor: state; second tensor: camera image
    assert len(inputs) == 2
    state, img = inputs
    assert state.shape == (1, _STATE_DIM)
    assert img.shape == (1, _IMG_C, _IMG_H, _IMG_W)


def test_make_sample_inputs_diffusion_unet():
    pytest.importorskip("diffusers", reason="diffusers required")
    policy = _make_diffusion_policy()

    class _Cfg:
        device = "cpu"
        batch_size = 1

    inputs = make_sample_inputs(policy, _Cfg(), mode="diffusion-unet")
    assert len(inputs) == 3  # sample, timestep, global_cond
    sample, timestep, global_cond = inputs
    cfg = policy.config
    assert sample.shape[1] == cfg.horizon
    assert sample.shape[2] == cfg.action_feature.shape[0]
    assert timestep.shape == (1,)
    assert global_cond.ndim == 2


def test_make_sample_inputs_unknown_mode():
    pytest.importorskip("torchvision", reason="torchvision required")
    policy = _make_act_policy()

    class _Cfg:
        device = "cpu"
        batch_size = 1

    with pytest.raises(ValueError, match="Unknown sample_inputs mode"):
        make_sample_inputs(policy, _Cfg(), mode="bogus")


# ──────────────────────────────────────────────────────────────────────────────
# Generic wrapper
# ──────────────────────────────────────────────────────────────────────────────


def test_generic_wrapper_forward():
    """GenericPolicyWrapper passes inputs through as a dict to policy.model."""
    from torch import nn

    class _DummyModel(nn.Module):
        def forward(self, batch: dict) -> torch.Tensor:
            return batch[OBS_STATE]

    wrapper = GenericPolicyWrapper(
        policy_model=_DummyModel(),
        input_feature_keys=[OBS_STATE],
    )
    wrapper.eval()

    x = torch.rand(2, _STATE_DIM)
    with torch.no_grad():
        out = wrapper(x)

    assert torch.allclose(out, x)


@pytest.mark.parametrize("unsupported_type", ["sac", "vqbet", "tdmpc", "pi0", "smolvla"])
def test_generic_wrapper_unsupported_raises(unsupported_type: str):
    """make_export_wrapper raises NotImplementedError for known-incompatible types."""
    from torch import nn

    class _Cfg:
        type = unsupported_type
        input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,))}
        output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
        n_obs_steps = 1

    class _DummyPolicy:
        config = _Cfg()
        model = nn.Linear(4, 2)

    with pytest.raises(NotImplementedError, match="register_export_wrapper"):
        make_export_wrapper(_DummyPolicy(), object())


def test_generic_wrapper_multistep_raises():
    """Generic wrapper rejects policies with n_obs_steps > 1."""
    from torch import nn

    class _Cfg:
        type = "_test_multistep"
        input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,))}
        output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
        n_obs_steps = 2

    class _DummyPolicy:
        config = _Cfg()
        model = nn.Linear(4, 2)

    with pytest.raises(NotImplementedError, match="n_obs_steps"):
        make_export_wrapper(_DummyPolicy(), object())


# ──────────────────────────────────────────────────────────────────────────────
# Exporter backends (auto / dynamo / legacy)
# ──────────────────────────────────────────────────────────────────────────────


def test_onnx_export_auto_falls_back_to_legacy(tmp_path: Path, monkeypatch, caplog):
    """When dynamo export raises, auto mode logs a warning and falls back to legacy."""
    pytest.importorskip("torchvision", reason="torchvision required for ACT backbone")
    import logging

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    cfg = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,)),
            f"{OBS_IMAGES}.cam": PolicyFeature(
                type=FeatureType.VISUAL, shape=(_IMG_C, _IMG_H, _IMG_W)
            ),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))},
        chunk_size=_CHUNK_SIZE,
        n_action_steps=_CHUNK_SIZE,
        dim_model=64,
        n_heads=2,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
    )
    policy = ACTPolicy(cfg)
    policy.eval()

    class _Cfg:
        device = "cpu"
        batch_size = 1
        diffusion_mode = "unet-only"

    wrapper, spec = make_export_wrapper(policy, _Cfg())
    wrapper.eval()

    # Force the dynamo branch to fail by intercepting torch.onnx.export.
    real_export = torch.onnx.export

    def _fake_export(*args, **kwargs):
        if kwargs.get("dynamo"):
            raise RuntimeError("simulated dynamo failure")
        return real_export(*args, **kwargs)

    monkeypatch.setattr(torch.onnx, "export", _fake_export)

    with caplog.at_level(logging.WARNING, logger="lerobot.export.onnx_export"):
        onnx_path = export_to_onnx(
            wrapper=wrapper,
            spec=spec,
            output_path=tmp_path / "act_auto",
            opset_version=18,
            precision="fp32",
            exporter="auto",
        )

    assert onnx_path.exists()
    assert any("falling back to legacy" in rec.message for rec in caplog.records), (
        "Expected fallback warning not found in logs"
    )


def test_onnx_export_legacy_explicit(tmp_path: Path):
    """exporter='legacy' produces a working ONNX file (smoke test on ACT)."""
    pytest.importorskip("torchvision", reason="torchvision required for ACT backbone")

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    cfg = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,)),
            f"{OBS_IMAGES}.cam": PolicyFeature(
                type=FeatureType.VISUAL, shape=(_IMG_C, _IMG_H, _IMG_W)
            ),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))},
        chunk_size=_CHUNK_SIZE,
        n_action_steps=_CHUNK_SIZE,
        dim_model=64,
        n_heads=2,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
    )
    policy = ACTPolicy(cfg)
    policy.eval()

    class _Cfg:
        device = "cpu"
        batch_size = 1
        diffusion_mode = "unet-only"

    wrapper, spec = make_export_wrapper(policy, _Cfg())
    wrapper.eval()

    onnx_path = export_to_onnx(
        wrapper=wrapper,
        spec=spec,
        output_path=tmp_path / "act_legacy",
        opset_version=18,
        precision="fp32",
        exporter="legacy",
    )
    assert onnx_path.exists()


def test_onnx_export_invalid_exporter_raises(tmp_path: Path):
    """Invalid exporter value raises ValueError."""
    spec = ExportSpec(
        input_names=["x"],
        output_names=["y"],
        sample_inputs=(torch.zeros(1, 4),),
        dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}},
    )

    from torch import nn

    with pytest.raises(ValueError, match="Invalid exporter"):
        export_to_onnx(
            wrapper=nn.Linear(4, 2),
            spec=spec,
            output_path=tmp_path / "bogus",
            exporter="not-a-valid-mode",
        )


def test_act_export_spec_has_dynamic_shapes():
    """ACT wrapper's ExportSpec must populate dynamic_shapes for the dynamo path.

    The dynamo path (torch.export) expects ``dynamic_shapes`` as a tuple/list
    matching the positional sample inputs (one entry per arg).
    """
    pytest.importorskip("torchvision", reason="torchvision required for ACT backbone")
    policy = _make_act_policy()

    class _Cfg:
        device = "cpu"
        batch_size = 1
        diffusion_mode = "unet-only"

    _, spec = make_export_wrapper(policy, _Cfg())
    assert spec.dynamic_shapes is not None
    # Tuple form must match positional sample_inputs and declare a batch dim
    # for every input.
    assert len(spec.dynamic_shapes) == len(spec.sample_inputs)
    assert len(spec.dynamic_shapes) == len(spec.input_names)
    for shape_spec in spec.dynamic_shapes:
        assert 0 in shape_spec, f"missing batch axis in {shape_spec!r}"
