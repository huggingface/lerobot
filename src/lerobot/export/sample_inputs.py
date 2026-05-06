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
"""Generic helpers for synthesizing sample inputs from a policy config.

Per-policy export modules (``policies/<type>/export_<type>.py``) typically
compose their own sample inputs. This module exposes a fallback helper that
covers the common case of "one zero tensor per non-image input feature, plus
one per image feature".
"""

from __future__ import annotations

import torch
from torch import Tensor


def make_zero_inputs_from_features(
    config,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, ...]:
    """Derive zero-tensor sample inputs from ``config.input_features``.

    Non-image features are emitted first, then image features, in dictionary
    iteration order. Each tensor has shape ``(batch_size, *feature.shape)``.
    For policies with ``n_obs_steps > 1``, an extra observation-step axis is
    inserted: ``(batch_size, n_obs_steps, *feature.shape)``.

    This helper is meant for new policies whose forward signature is a flat
    list of positional tensors. Policies with bespoke shapes (e.g. Diffusion
    UNet's pre-computed ``global_cond``) should build their own sample inputs.
    """
    device = torch.device(device)
    n_obs_steps = getattr(config, "n_obs_steps", 1)

    inputs: list[Tensor] = []

    non_image_keys = [k for k in (config.input_features or {}) if "image" not in k.lower()]
    image_keys = [k for k in (config.input_features or {}) if "image" in k.lower()]

    for key in non_image_keys + image_keys:
        feature = config.input_features[key]
        if n_obs_steps > 1:
            inputs.append(torch.zeros(batch_size, n_obs_steps, *feature.shape, device=device))
        else:
            inputs.append(torch.zeros(batch_size, *feature.shape, device=device))

    return tuple(inputs)
