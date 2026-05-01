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
"""Reusable adapter for policies whose inference forward consumes a single dict batch.

This pattern covers ACT, VQBET (with ``rollout=True``), and any policy that
exposes a single submodule whose ``forward(batch: dict) -> Tensor | tuple``
contract can be normalized into ``positional_inputs -> dict_batch -> output``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.utils.constants import OBS_IMAGES


@dataclass
class DictBatchSpec:
    """Declarative description of a dict-batch policy adapter.

    The list ``input_feature_keys`` defines the order of positional ONNX inputs.
    For each entry, the corresponding tensor is placed under that key in the
    dict passed to the wrapped submodule. Image entries (``image_keys``) are
    repacked into one of three conventions before being placed under the
    canonical ``OBS_IMAGES`` key.
    """

    input_feature_keys: list[str]
    image_keys: list[str] = field(default_factory=list)
    # "list":    OBS_IMAGES becomes a Python list of (B, C, H, W) tensors (ACT)
    # "stacked": OBS_IMAGES becomes a stacked tensor; the n_cams axis is
    #            inserted at ``image_stack_dim`` via torch.stack (VQBET, Diffusion)
    # "single":  the lone image tensor stays at its key (TDMPC etc.)
    image_convention: str = "list"
    # When image_convention='stacked', axis at which to insert the n_cameras
    # dimension. For per-camera inputs of shape (B, C, H, W) use dim=1 to get
    # (B, n_cams, C, H, W). For per-camera inputs (B, n_obs_steps, C, H, W) use
    # dim=2 to get (B, n_obs_steps, n_cams, C, H, W) — VQBET convention.
    image_stack_dim: int = 1
    # Extra kwargs forwarded to submodule.forward, e.g. {"rollout": True}.
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    # How to extract the action tensor from the submodule's return value:
    #   - None           => the submodule returns a Tensor directly
    #   - int            => index into a tuple/list output
    #   - str            => key into a dict output
    output_index: int | str | None = None


class DictBatchAdapter(nn.Module):
    """ONNX-traceable wrapper: positional tensors -> dict batch -> submodule -> tensor.

    The wrapped ``submodule`` is whatever submodule of a Policy holds the
    network you actually want exported (e.g. ``policy.model`` for ACT,
    ``policy.vqbet`` for VQBET). Stateful queue / ensembler logic on the parent
    Policy is bypassed by going through this adapter directly.
    """

    def __init__(self, submodule: nn.Module, spec: DictBatchSpec) -> None:
        super().__init__()
        self.submodule = submodule
        self.spec = spec
        self._non_image_keys = [k for k in spec.input_feature_keys if k not in spec.image_keys]
        if spec.image_convention not in ("list", "stacked", "single"):
            raise ValueError(
                f"Invalid image_convention='{spec.image_convention}'. "
                "Use 'list', 'stacked', or 'single'."
            )

    def forward(self, *positional: Tensor) -> Tensor:
        if len(positional) != len(self.spec.input_feature_keys):
            raise ValueError(
                f"Expected {len(self.spec.input_feature_keys)} positional inputs "
                f"matching input_feature_keys, got {len(positional)}"
            )

        batch: dict[str, Any] = {}
        image_tensors: list[Tensor] = []
        for key, tensor in zip(self.spec.input_feature_keys, positional, strict=True):
            if key in self.spec.image_keys:
                image_tensors.append(tensor)
            else:
                batch[key] = tensor

        if image_tensors:
            if self.spec.image_convention == "list":
                # ACT: OBS_IMAGES is a Python list of per-camera (B, C, H, W) tensors.
                batch[OBS_IMAGES] = list(image_tensors)
            elif self.spec.image_convention == "stacked":
                # VQBET / Diffusion: OBS_IMAGES is (B, [n_obs_steps,] n_cams, C, H, W).
                # The n_cams axis is inserted at ``image_stack_dim`` (default: 1).
                batch[OBS_IMAGES] = torch.stack(image_tensors, dim=self.spec.image_stack_dim)
            else:  # "single"
                # TDMPC: a single OBS_IMAGE key with a (B, C, H, W) tensor.
                if len(image_tensors) != 1:
                    raise ValueError(
                        f"image_convention='single' expects exactly one image tensor, "
                        f"got {len(image_tensors)}"
                    )
                batch[self.spec.image_keys[0]] = image_tensors[0]

        result = self.submodule(batch, **self.spec.extra_kwargs)
        return _extract_output(result, self.spec.output_index)


def _extract_output(result: Any, index: int | str | None) -> Tensor:
    """Pull a Tensor out of a submodule's possibly-nested return value."""
    if index is None:
        if isinstance(result, Tensor):
            return result
        # Common case: a tuple where the first element is the action tensor.
        if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], Tensor):
            return result[0]
        raise TypeError(
            f"Submodule returned {type(result).__name__}; specify output_index to extract a tensor"
        )
    if isinstance(index, int):
        return result[index]
    if isinstance(index, str):
        return result[index]
    raise TypeError(f"output_index must be int, str, or None; got {type(index).__name__}")
