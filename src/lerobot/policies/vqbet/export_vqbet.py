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
"""ONNX export adapter for VQ-BeT policy.

VQ-BeT's network is exposed at ``policy.vqbet`` and its forward signature is
``forward(batch: dict, rollout: bool) -> Tensor | tuple``. With ``rollout=True``
it returns a single action chunk tensor — exactly the contract this adapter
exports.

Inference contract: each per-camera ONNX input has shape
``(B, n_obs_steps, C, H, W)``; the n_obs_steps axis is part of the input
because VQ-BeT needs the trailing window of observations (the policy itself
manages no queue inside the exported graph).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lerobot.export.adapters import DictBatchAdapter, DictBatchSpec
from lerobot.export.core import ExportSpec, make_batch_dynamic_axes_and_shapes
from lerobot.utils.constants import OBS_STATE

if TYPE_CHECKING:
    from .modeling_vqbet import VQBeTPolicy


def _key_to_input_name(key: str) -> str:
    """Convert a feature key like 'observation.images.cam' to 'observation_images_cam'."""
    return re.sub(r"[^a-zA-Z0-9]", "_", key)


def _make_vqbet_sample_inputs(config, batch_size: int, device: torch.device) -> tuple[Tensor, ...]:
    """Zero-initialized sample tensors matching the VQ-BeT export wrapper's input order.

    Per-input shapes:
      - observation.state:           (B, n_obs_steps, state_dim)
      - observation.images.<cam>:    (B, n_obs_steps, C, H, W)
    """
    inputs: list[Tensor] = []
    n_obs_steps = config.n_obs_steps

    if config.robot_state_feature is None:
        raise ValueError("VQ-BeT export currently requires a robot_state_feature.")
    state_dim = config.robot_state_feature.shape[0]
    inputs.append(torch.zeros(batch_size, n_obs_steps, state_dim, device=device))

    for _key, ft in (config.image_features or {}).items():
        # ft.shape is (C, H, W).
        inputs.append(torch.zeros(batch_size, n_obs_steps, *ft.shape, device=device))

    return tuple(inputs)


def make_vqbet_export_wrapper(policy: VQBeTPolicy, cfg) -> tuple[nn.Module, ExportSpec]:
    """Build (wrapper, ExportSpec) for VQ-BeT export.

    Auto-discovered by ``lerobot.export.core.make_export_wrapper``.
    """
    config = policy.config
    device = torch.device(getattr(cfg, "device", "cpu"))
    batch_size = getattr(cfg, "batch_size", 1)

    policy.vqbet.eval()

    image_keys = list(config.image_features or {})
    input_feature_keys = [OBS_STATE, *image_keys]
    input_names = ["observation_state", *(_key_to_input_name(k) for k in image_keys)]

    adapter_spec = DictBatchSpec(
        input_feature_keys=input_feature_keys,
        image_keys=image_keys,
        image_convention="stacked",
        # Per-camera inputs are (B, n_obs_steps, C, H, W); inserting the n_cams
        # axis at dim=2 yields VQ-BeT's expected (B, n_obs_steps, n_cams, C, H, W).
        image_stack_dim=2,
        extra_kwargs={"rollout": True},
        output_index=None,  # rollout=True returns a Tensor directly.
    )
    wrapper = DictBatchAdapter(policy.vqbet, adapter_spec)
    wrapper.eval()

    sample_inputs = _make_vqbet_sample_inputs(config, batch_size, device)
    output_names = ["action_chunk"]
    dynamic_axes, _ = make_batch_dynamic_axes_and_shapes(
        input_names=input_names,
        sample_inputs=sample_inputs,
        output_names=output_names,
    )
    # Adapter forward uses *positional varargs, so dynamo dynamic_shapes must
    # wrap per-input dim specs in a one-element outer tuple.
    from torch.export import Dim

    batch_dim = Dim("batch_size", min=1, max=64)
    dynamic_shapes = (tuple({0: batch_dim} for _ in sample_inputs),)

    return wrapper, ExportSpec(
        input_names=input_names,
        output_names=output_names,
        sample_inputs=sample_inputs,
        dynamic_axes=dynamic_axes,
        dynamic_shapes=dynamic_shapes,
        policy_note=(
            "Exports VQ-BeT (policy.vqbet) with rollout=True. "
            "Per-camera input shape: (B, n_obs_steps, C, H, W) — the trailing "
            "n_obs_steps observations must be stacked by the caller. "
            "Output: (B, action_chunk_size, action_dim)."
        ),
    )
