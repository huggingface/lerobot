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
"""Per-policy ONNX export tests for ACT.

Run with:
    uv run pytest tests/policies/test_export_act.py -svv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

# Skip the whole module if export extras are missing.
pytest.importorskip("onnx", reason="onnx not installed — skip ACT export tests")
pytest.importorskip("onnxruntime", reason="onnxruntime not installed — skip ACT export tests")
pytest.importorskip("torchvision", reason="torchvision required for ACT backbone")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.export.core import make_export_wrapper  # noqa: E402
from lerobot.export.onnx_export import export_to_onnx  # noqa: E402
from lerobot.export.validation import validate_onnx  # noqa: E402
from lerobot.policies.act.export_act import (  # noqa: E402
    ACTInferenceWrapper,
    make_act_export_wrapper,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402

_STATE_DIM = 6
_ACTION_DIM = 6
_CHUNK_SIZE = 10
_IMG_C, _IMG_H, _IMG_W = 3, 64, 64


def _make_act_policy():
    """Build a fresh ACT policy from a minimal toy config (no pretrained weights)."""
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
    return policy


class _Cfg:
    device = "cpu"
    batch_size = 1


def test_act_wrapper_forward():
    """ACTInferenceWrapper.forward returns (1, chunk_size, action_dim)."""
    policy = _make_act_policy()
    wrapper = ACTInferenceWrapper(policy.model, policy.config)
    wrapper.eval()
    state = torch.zeros(1, _STATE_DIM)
    with torch.no_grad():
        out = wrapper(state, torch.zeros(1, _IMG_C, _IMG_H, _IMG_W))
    assert out.shape == (1, _CHUNK_SIZE, _ACTION_DIM)


def test_make_act_export_wrapper_returns_correct_spec():
    """The factory produces an ExportSpec with the expected structure."""
    policy = _make_act_policy()
    wrapper, spec = make_act_export_wrapper(policy, _Cfg())

    assert isinstance(wrapper, ACTInferenceWrapper)
    assert spec.input_names == ["observation_state", "observation_images_cam"]
    assert spec.output_names == ["action_chunk"]
    assert len(spec.sample_inputs) == 2
    # dynamic_shapes must align with positional inputs (tuple form).
    assert spec.dynamic_shapes is not None
    assert len(spec.dynamic_shapes) == 2
    for shape_spec in spec.dynamic_shapes:
        assert 0 in shape_spec


def test_act_auto_discovery_via_make_export_wrapper():
    """make_export_wrapper finds policies/act/export_act.py via convention."""
    policy = _make_act_policy()
    wrapper, spec = make_export_wrapper(policy, _Cfg())
    assert isinstance(wrapper, ACTInferenceWrapper)
    assert spec.output_names == ["action_chunk"]


def test_onnx_export_act(tmp_path: Path):
    """End-to-end: ACT → ONNX → ORT parity check."""
    policy = _make_act_policy()
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


def test_onnx_export_act_legacy_explicit(tmp_path: Path):
    """exporter='legacy' produces a working ONNX file for ACT."""
    policy = _make_act_policy()
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


def test_onnx_export_act_auto_falls_back_to_legacy(tmp_path: Path, monkeypatch, caplog):
    """When dynamo export raises, auto mode logs a warning and falls back to legacy."""
    policy = _make_act_policy()
    wrapper, spec = make_export_wrapper(policy, _Cfg())
    wrapper.eval()

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
