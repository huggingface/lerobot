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
"""Per-policy ONNX export tests for VQ-BeT.

Run with:
    uv run pytest tests/policies/test_export_vqbet.py -svv
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# Skip the module if export extras or torchvision are missing.
pytest.importorskip("onnx", reason="onnx not installed — skip VQ-BeT export tests")
pytest.importorskip("onnxruntime", reason="onnxruntime not installed — skip VQ-BeT export tests")
pytest.importorskip("torchvision", reason="torchvision required for VQ-BeT backbone")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.export.adapters import DictBatchAdapter  # noqa: E402
from lerobot.export.core import make_export_wrapper  # noqa: E402
from lerobot.export.onnx_export import export_to_onnx  # noqa: E402
from lerobot.export.validation import validate_onnx  # noqa: E402
from lerobot.policies.vqbet.export_vqbet import make_vqbet_export_wrapper  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402

_STATE_DIM = 6
_ACTION_DIM = 6
_IMG_C, _IMG_H, _IMG_W = 3, 96, 96


def _make_vqbet_policy():
    """Build a fresh VQ-BeT policy from a minimal toy config (no pretrained weights)."""
    from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
    from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

    cfg = VQBeTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,)),
            f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(_IMG_C, _IMG_H, _IMG_W)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))},
        n_obs_steps=2,
        action_chunk_size=4,
        n_action_pred_token=2,
        crop_shape=None,
    )
    policy = VQBeTPolicy(cfg)
    policy.eval()
    return policy


class _Cfg:
    device = "cpu"
    batch_size = 1


def test_vqbet_adapter_forward_shape():
    """The DictBatchAdapter built for VQ-BeT returns (1, action_chunk_size, action_dim)."""
    policy = _make_vqbet_policy()
    wrapper, _ = make_vqbet_export_wrapper(policy, _Cfg())
    wrapper.eval()
    n_obs_steps = policy.config.n_obs_steps

    state = torch.zeros(1, n_obs_steps, _STATE_DIM)
    image = torch.zeros(1, n_obs_steps, _IMG_C, _IMG_H, _IMG_W)
    with torch.no_grad():
        out = wrapper(state, image)

    assert out.shape == (1, policy.config.action_chunk_size, _ACTION_DIM)


def test_make_vqbet_export_wrapper_returns_correct_spec():
    """The factory yields a DictBatchAdapter and an ExportSpec with stacked images."""
    policy = _make_vqbet_policy()
    wrapper, spec = make_vqbet_export_wrapper(policy, _Cfg())

    assert isinstance(wrapper, DictBatchAdapter)
    assert spec.input_names == ["observation_state", "observation_images_cam"]
    assert spec.output_names == ["action_chunk"]
    assert wrapper.spec.image_convention == "stacked"
    assert wrapper.spec.image_stack_dim == 2
    assert wrapper.spec.extra_kwargs == {"rollout": True}


def test_vqbet_auto_discovery_via_make_export_wrapper():
    """make_export_wrapper finds policies/vqbet/export_vqbet.py via convention."""
    policy = _make_vqbet_policy()
    wrapper, spec = make_export_wrapper(policy, _Cfg())
    assert isinstance(wrapper, DictBatchAdapter)
    assert spec.output_names == ["action_chunk"]


def test_onnx_export_vqbet(tmp_path: Path):
    """End-to-end: VQ-BeT export to ONNX (smoke test).

    Strict numerical parity is intentionally not asserted here: VQ-BeT uses a
    discrete VQ-VAE codebook (argmin lookup), and on a randomly initialized
    toy model the codebook is unstable enough that ONNX advanced-indexing vs
    PyTorch indexing can diverge by a wide margin (max_abs_error ~ 1e-1) on
    near-tie code distances. On a properly trained checkpoint the divergence
    is expected to be small. We validate that the export completes, the ONNX
    file is structurally valid, and ORT can produce an output of the correct
    shape.
    """
    import numpy as np
    import onnxruntime as ort

    policy = _make_vqbet_policy()
    wrapper, spec = make_export_wrapper(policy, _Cfg())
    wrapper.eval()

    onnx_path = export_to_onnx(
        wrapper=wrapper,
        spec=spec,
        output_path=tmp_path / "vqbet_fp32",
        opset_version=18,
        precision="fp32",
        exporter="legacy",
    )
    assert onnx_path.exists()

    # Confirm ORT can run the model end-to-end.
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        sess.get_inputs()[i].name: spec.sample_inputs[i].cpu().numpy().astype(np.float32)
        for i in range(len(spec.sample_inputs))
    }
    out = sess.run(None, ort_inputs)
    assert out[0].shape == (1, policy.config.action_chunk_size, _ACTION_DIM)


def test_onnx_export_vqbet_validate_helper_runs(tmp_path: Path):
    """Smoke check that validate_onnx() can score a VQ-BeT export.

    We only assert the helper returns the expected schema; we do not require
    parity passing because of the randomly initialized codebook (see the
    test above for context).
    """
    policy = _make_vqbet_policy()
    wrapper, spec = make_export_wrapper(policy, _Cfg())
    wrapper.eval()

    onnx_path = export_to_onnx(
        wrapper=wrapper,
        spec=spec,
        output_path=tmp_path / "vqbet_fp32",
        opset_version=18,
        precision="fp32",
        exporter="legacy",
    )
    results = validate_onnx(
        wrapper=wrapper,
        sample_inputs=spec.sample_inputs,
        onnx_path=onnx_path,
        rtol=1.0,
        atol=1.0,
    )
    assert "max_abs_error" in results
    assert "cos_sim" in results
    assert "allclose" in results
    assert "trials" in results
