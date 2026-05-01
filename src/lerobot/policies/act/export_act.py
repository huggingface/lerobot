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

Auto-discovered by ``lerobot.export.core.make_export_wrapper`` via the
naming convention ``policies/<type>/export_<type>.py:make_<type>_export_wrapper``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lerobot.export.core import ExportSpec, make_batch_dynamic_axes_and_shapes
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

if TYPE_CHECKING:
    from .modeling_act import ACT, ACTPolicy


def _key_to_input_name(key: str) -> str:
    """Convert a feature key like 'observation.images.cam_high' to 'observation_images_cam_high'."""
    return re.sub(r"[^a-zA-Z0-9]", "_", key)


class ACTInferenceWrapper(nn.Module):
    """ONNX-compatible wrapper for ACT inference.

    Exports the backbone + Transformer encoder/decoder + action head.
    Excluded: VAE encoder (latent is zeros at inference), action queue, temporal ensembler.

    Inside ACT.forward, ``torch.zeros([batch_size, latent_dim])`` is allocated.
    Under legacy tracing this is baked as a B=1 constant; under the dynamo
    exporter (``torch.export``) it is symbolicized via ``Dim``.
    """

    def __init__(self, act_model: ACT, config) -> None:
        super().__init__()
        self.model = act_model
        self.config = config
        self._has_robot_state = config.robot_state_feature is not None
        self._has_env_state = config.env_state_feature is not None
        self._image_keys: list[str] = list(config.image_features.keys()) if config.image_features else []

    def forward(self, robot_state: Tensor, *camera_images: Tensor) -> Tensor:
        """
        Args:
            robot_state: ``(B, state_dim)`` robot joint state. If the policy has no
                ``robot_state_feature`` and uses ``env_state_feature`` instead, pass
                the environment state here with the same shape contract.
            *camera_images: One ``(B, C, H, W)`` tensor per camera, in the order
                defined by ``config.image_features``.

        Returns:
            ``(B, chunk_size, action_dim)`` predicted action chunk.
        """
        batch: dict[str, Tensor | list[Tensor]] = {}

        if self._has_robot_state:
            batch[OBS_STATE] = robot_state
        elif self._has_env_state:
            batch[OBS_ENV_STATE] = robot_state  # env_state passed via robot_state slot

        if self._image_keys:
            # ACT.forward expects OBS_IMAGES as a Python list of per-camera tensors.
            batch[OBS_IMAGES] = list(camera_images)

        actions, _ = self.model(batch)  # VAE encoder skipped (eval mode, latent = zeros)
        return actions


def _make_act_sample_inputs(config, batch_size: int, device: torch.device) -> tuple[Tensor, ...]:
    """Zero-initialized sample tensors matching ``ACTInferenceWrapper.forward``."""
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
        # ft.shape is (C, H, W)
        inputs.append(torch.zeros(batch_size, *ft.shape, device=device))

    return tuple(inputs)


def make_act_export_wrapper(policy: ACTPolicy, cfg) -> tuple[nn.Module, ExportSpec]:
    """Build (wrapper, ExportSpec) for ACT export.

    Auto-discovered by ``lerobot.export.core.make_export_wrapper``.
    """
    config = policy.config
    device = torch.device(getattr(cfg, "device", "cpu"))
    batch_size = getattr(cfg, "batch_size", 1)

    policy.model.eval()
    wrapper = ACTInferenceWrapper(policy.model, config)
    wrapper.eval()

    input_names: list[str] = []
    if config.robot_state_feature:
        input_names.append("observation_state")
    elif config.env_state_feature:
        input_names.append("observation_env_state")
    for key in config.image_features or {}:
        input_names.append(_key_to_input_name(key))

    sample_inputs = _make_act_sample_inputs(config, batch_size, device)
    output_names = ["action_chunk"]
    _, dynamic_shapes = make_batch_dynamic_axes_and_shapes(
        input_names=input_names,
        sample_inputs=sample_inputs,
        output_names=output_names,
    )

    return wrapper, ExportSpec(
        input_names=input_names,
        output_names=output_names,
        sample_inputs=sample_inputs,
        # batch_size=1 fixed under legacy tracing; dynamic under dynamo.
        dynamic_axes=None,
        dynamic_shapes=dynamic_shapes,
        policy_note=(
            "Exports ACT backbone + Transformer encoder/decoder + action head. "
            "Action queue and temporal ensembler must be managed in Python. "
            "batch_size is fixed=1 under exporter='legacy', dynamic under exporter='dynamo'."
        ),
    )
