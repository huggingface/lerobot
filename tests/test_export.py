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
"""Framework-level tests for lerobot.export.

Per-policy export tests live alongside each policy in
``tests/policies/test_export_<policy>.py``.

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
from lerobot.export.core import (  # noqa: E402
    WRAPPER_REGISTRY,
    ExportSpec,
    _try_load_builtin_factory,
    make_batch_dynamic_axes_and_shapes,
    make_export_wrapper,
    register_export_wrapper,
)
from lerobot.export.normalization import NormalizedWrapper, save_normalization_stats  # noqa: E402
from lerobot.export.onnx_export import export_to_onnx  # noqa: E402
from lerobot.export.sample_inputs import make_zero_inputs_from_features  # noqa: E402
from lerobot.export.tensorrt_export import get_gpu_compute_capability  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

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
    assert spec.dynamic_shapes is None
    assert spec.policy_note == ""


# ──────────────────────────────────────────────────────────────────────────────
# Auto-discovery and registration
# ──────────────────────────────────────────────────────────────────────────────


def test_try_load_builtin_factory_discovers_act():
    """The auto-discovery helper finds policies/act/export_act.py:make_act_export_wrapper."""
    factory = _try_load_builtin_factory("act")
    assert factory is not None
    assert factory.__name__ == "make_act_export_wrapper"


def test_try_load_builtin_factory_returns_none_for_unknown_policy():
    """Unknown policy types return None (no exception)."""
    assert _try_load_builtin_factory("policy_that_does_not_exist") is None


def test_register_export_wrapper_decorator():
    """Explicitly registered factories take precedence over auto-discovery."""
    from torch import nn

    class _DummyConfig:
        type = "_test_dummy_register"
        input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,))}
        n_obs_steps = 1

    class _DummyPolicy:
        config = _DummyConfig()
        model = nn.Linear(4, 2)

    called = {}

    @register_export_wrapper("_test_dummy_register")
    def _make_dummy(policy, cfg):
        called["yes"] = True
        return policy.model, ExportSpec(
            input_names=["x"],
            output_names=["y"],
            sample_inputs=(torch.zeros(1, 4),),
            dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}},
        )

    try:
        wrapper, spec = make_export_wrapper(_DummyPolicy(), object())
        assert called.get("yes")
        assert spec.input_names == ["x"]
    finally:
        # Clean up registry to avoid leaking state into other tests.
        WRAPPER_REGISTRY.pop("_test_dummy_register", None)


def test_make_export_wrapper_unknown_policy_raises_helpful_error():
    """Unknown policy types raise NotImplementedError with a clear extension guide."""
    from torch import nn

    policy_type = "test_truly_unknown_policy"

    class _Cfg:
        type = policy_type
        input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,))}
        n_obs_steps = 1

    class _DummyPolicy:
        config = _Cfg()
        model = nn.Linear(4, 2)

    with pytest.raises(NotImplementedError) as exc_info:
        make_export_wrapper(_DummyPolicy(), object())
    msg = str(exc_info.value)
    assert policy_type in msg
    assert f"export_{policy_type}.py" in msg
    assert "register_export_wrapper" in msg


# ──────────────────────────────────────────────────────────────────────────────
# make_batch_dynamic_axes_and_shapes helper
# ──────────────────────────────────────────────────────────────────────────────


def test_make_batch_dynamic_axes_and_shapes_basic():
    """Helper generates aligned legacy and dynamo dynamic specs."""
    sample_inputs = (torch.zeros(1, 4), torch.zeros(1, 3, 8, 8))
    dynamic_axes, dynamic_shapes = make_batch_dynamic_axes_and_shapes(
        input_names=["state", "image"],
        sample_inputs=sample_inputs,
        output_names=["action"],
    )
    assert dynamic_axes == {
        "state": {0: "batch_size"},
        "image": {0: "batch_size"},
        "action": {0: "batch_size"},
    }
    assert len(dynamic_shapes) == 2
    for shape_spec in dynamic_shapes:
        assert 0 in shape_spec


# ──────────────────────────────────────────────────────────────────────────────
# make_zero_inputs_from_features (generic helper for new policies)
# ──────────────────────────────────────────────────────────────────────────────


def test_make_zero_inputs_from_features_single_step():
    """Generic helper produces (B, *shape) zeros for n_obs_steps=1 policies."""

    class _Cfg:
        n_obs_steps = 1
        input_features = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        }

    inputs = make_zero_inputs_from_features(_Cfg(), batch_size=2)
    # Non-image keys come first.
    assert len(inputs) == 2
    assert inputs[0].shape == (2, 6)
    assert inputs[1].shape == (2, 3, 32, 32)


def test_make_zero_inputs_from_features_multistep():
    """Multi-step policies get an extra n_obs_steps axis."""

    class _Cfg:
        n_obs_steps = 3
        input_features = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        }

    (state,) = make_zero_inputs_from_features(_Cfg(), batch_size=1)
    assert state.shape == (1, 3, 6)


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
    from lerobot.utils.constants import OBS_IMAGES

    cfg = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=10,
        n_action_steps=10,
        dim_model=64,
        n_heads=2,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
    )
    dataset_stats = {
        OBS_STATE: {"mean": torch.zeros(6), "std": torch.ones(6)},
        ACTION: {"mean": torch.full((6,), 0.5), "std": torch.full((6,), 2.0)},
    }
    preprocessor, postprocessor = make_act_pre_post_processors(cfg, dataset_stats=dataset_stats)
    out_path = save_normalization_stats(preprocessor, postprocessor, tmp_path)
    data = json.loads(out_path.read_text())

    assert OBS_STATE in data
    assert ACTION in data
    assert data[ACTION]["mean"] == [0.5] * 6
    assert data[ACTION]["std"] == [2.0] * 6


def test_normalized_wrapper_forward():
    """NormalizedWrapper normalizes inputs and denormalizes outputs."""
    from torch import nn

    state_dim = 4
    action_dim = 3

    class _Identity(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :action_dim]

    input_stats = {"observation.state": {"mean": torch.zeros(state_dim), "std": torch.ones(state_dim)}}
    action_stats = {"mean": torch.zeros(action_dim), "std": torch.ones(action_dim) * 2.0}

    wrapper = NormalizedWrapper(_Identity(), input_stats, action_stats)
    wrapper.eval()
    x = torch.ones(1, state_dim)
    with torch.no_grad():
        out = wrapper(x)
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
# Exporter backends (auto / dynamo / legacy)
# ──────────────────────────────────────────────────────────────────────────────


def test_onnx_export_invalid_exporter_raises(tmp_path: Path):
    """Invalid exporter value raises ValueError."""
    from torch import nn

    spec = ExportSpec(
        input_names=["x"],
        output_names=["y"],
        sample_inputs=(torch.zeros(1, 4),),
        dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}},
    )
    with pytest.raises(ValueError, match="Invalid exporter"):
        export_to_onnx(
            wrapper=nn.Linear(4, 2),
            spec=spec,
            output_path=tmp_path / "bogus",
            exporter="not-a-valid-mode",
        )
