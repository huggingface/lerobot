#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Checkpoint loading shared by the openpi-derived policies that remap keys (pi0, pi05, pi0_fast)."""

from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import T, _load_state_dict_into_meta_model, _parameters_on_meta


def load_complete_checkpoint(
    policy_cls: type[T],
    pretrained_name_or_path: str | Path,
    config: PreTrainedConfig,
    download_kwargs: dict[str, Any],
    **kwargs,
) -> T | None:
    """Build `policy_cls` with its parameters on the meta device and stream the checkpoint straight into them.

    This skips randomly initializing weights that the checkpoint replaces, and never holds a second full
    copy of them. Returns None, before reading any weight, when the checkpoint cannot be read or its keys or
    shapes (after the policy's `_fix_pytorch_state_dict_keys`) differ from the policy's, so that
    `from_pretrained` loads the regular way, with the same result as before.
    """
    from transformers.utils import cached_file

    try:
        model_file = cached_file(pretrained_name_or_path, "model.safetensors", **download_kwargs)
        checkpoint = safe_open(model_file, framework="pt", device="cpu")
    except Exception:
        # The regular path tries again and reports the failure as it always has.
        return None
    with checkpoint:
        with _parameters_on_meta():
            policy = policy_cls(config, **kwargs)

        def policy_keys(key: str, shape: list[int]) -> dict[str, torch.Size]:
            fixed = policy._fix_pytorch_state_dict_keys({key: torch.empty(shape, device="meta")}, config)
            return {k if k.startswith("model.") else f"model.{k}": v.shape for k, v in fixed.items()}

        file_shapes = {key: checkpoint.get_slice(key).get_shape() for key in checkpoint.keys()}  # noqa: SIM118
        targets = {key: policy_keys(key, shape) for key, shape in file_shapes.items()}
        shapes = {name: shape for names in targets.values() for name, shape in names.items()}
        if shapes != {name: tensor.shape for name, tensor in policy.state_dict().items()}:
            return None
        print(f"Loading model from: {pretrained_name_or_path}")
        tensors = ((name, checkpoint.get_tensor(key)) for key, names in targets.items() for name in names)
        _load_state_dict_into_meta_model(policy, tensors, config.device)
    policy.model.to(config.device)
    print("All keys loaded successfully!")
    return policy
