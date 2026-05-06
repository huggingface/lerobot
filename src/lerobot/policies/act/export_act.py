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
"""ONNX export adapter for ACT (Action Chunking Transformer) policy.

Composed from the reusable :class:`DictBatchAdapter` primitive — see
``src/lerobot/export/adapters/dict_batch.py``.

Auto-discovered by ``lerobot.export.core.make_export_wrapper`` via the
naming convention ``policies/<type>/export_<type>.py:make_<type>_export_wrapper``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lerobot.export.adapters import DictBatchAdapter, DictBatchSpec
from lerobot.export.core import ExportSpec
from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE

if TYPE_CHECKING:
    from .modeling_act import ACTPolicy


def _key_to_input_name(key: str) -> str:
    """Convert a feature key like 'observation.images.cam_high' to 'observation_images_cam_high'."""
    return re.sub(r"[^a-zA-Z0-9]", "_", key)


def _make_act_sample_inputs(config, batch_size: int, device: torch.device) -> tuple[Tensor, ...]:
    """Zero-initialized sample tensors matching the ACT export wrapper's input order."""
    inputs: list[Tensor] = []

    if config.robot_state_feature is not None:
        state_dim = config.robot_state_feature.shape[0]
        inputs.append(torch.zeros(batch_size, state_dim, device=device))
    elif config.env_state_feature is not None:
        env_dim = config.env_state_feature.shape[0]
        inputs.append(torch.zeros(batch_size, env_dim, device=device))
    else:
        raise ValueError("ACT policy must have at least robot_state_feature or env_state_feature.")

    for _key, ft in (config.image_features or {}).items():
        # ft.shape is (C, H, W).
        inputs.append(torch.zeros(batch_size, *ft.shape, device=device))

    return tuple(inputs)


def make_act_export_wrapper(policy: ACTPolicy, cfg) -> tuple[nn.Module, ExportSpec]:
    """Build (wrapper, ExportSpec) for ACT export.

    ACT exposes its network as ``policy.model``; that network's
    ``forward(batch: dict)`` returns a ``(actions, (mu, sigma))`` tuple.
    We use :class:`DictBatchAdapter` with ``image_convention='list'`` (ACT's
    convention is ``OBS_IMAGES`` as a Python list of per-camera tensors) and
    ``output_index=0`` to drop the VAE statistics.
    """
    from torch.export import Dim

    config = policy.config
    device = torch.device(getattr(cfg, "device", "cpu"))
    batch_size = getattr(cfg, "batch_size", 1)

    policy.model.eval()

    # Choose the state feature key the model expects.
    if config.robot_state_feature is not None:
        state_key = OBS_STATE
        state_input_name = "observation_state"
    elif config.env_state_feature is not None:
        state_key = OBS_ENV_STATE
        state_input_name = "observation_env_state"
    else:
        raise ValueError("ACT policy must have at least robot_state_feature or env_state_feature.")

    image_keys = list(config.image_features or {})
    input_feature_keys = [state_key, *image_keys]
    input_names = [state_input_name, *(_key_to_input_name(k) for k in image_keys)]

    adapter_spec = DictBatchSpec(
        input_feature_keys=input_feature_keys,
        image_keys=image_keys,
        image_convention="list",  # ACT.forward expects OBS_IMAGES as a list.
        output_index=0,  # ACT.forward returns (actions, (mu, sigma)).
    )
    wrapper = DictBatchAdapter(policy.model, adapter_spec)
    wrapper.eval()

    sample_inputs = _make_act_sample_inputs(config, batch_size, device)
    output_names = ["action_chunk"]

    # The adapter's `forward(*positional)` collects inputs into a varargs tuple,
    # so torch.export sees a single packed-tuple parameter; dynamic_shapes must
    # mirror that structure (one-element outer tuple containing per-input dim
    # specs). ACT's internal `torch.zeros([batch_size, latent_dim])` still
    # specializes the batch axis to 1 in practice; the Dim entries below are
    # supplied so dynamo accepts the call without erroring out.
    batch_dim = Dim("batch_size", min=1, max=64)
    dynamic_shapes = (tuple({0: batch_dim} for _ in sample_inputs),)

    return wrapper, ExportSpec(
        input_names=input_names,
        output_names=output_names,
        sample_inputs=sample_inputs,
        # Legacy tracer fixes batch=1 because of in-model `torch.zeros([B, ...])`
        # allocations; dynamo currently specializes too. Genuine batch>1 export
        # for ACT requires upstream changes to ACT.forward (preallocate as
        # buffers), which is out of scope here.
        dynamic_axes=None,
        dynamic_shapes=dynamic_shapes,
        policy_note=(
            "Exports ACT backbone + Transformer encoder/decoder + action head. "
            "Action queue and temporal ensembler must be managed in Python. "
            "batch_size is fixed=1 in the resulting ONNX (model-side limitation, "
            "not exporter-side)."
        ),
    )
