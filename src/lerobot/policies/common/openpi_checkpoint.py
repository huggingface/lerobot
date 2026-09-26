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

import torch
from safetensors import safe_open

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import (
    T,
    _load_state_dict_into_meta_model,
    _parameters_on_meta,
    _shares_parameters,
)


def load_complete_checkpoint(
    policy_cls: type[T], model_file: str, config: PreTrainedConfig, **kwargs
) -> T | None:
    """Build `policy_cls` with its parameters on the meta device and stream `model_file` straight into them.

    This skips randomly initializing weights that the checkpoint replaces, and never holds a second full copy
    of them. Returns None, before reading any weight, when the file cannot be read, its keys or shapes differ
    from the policy's, or the policy shares a parameter or computed a buffer from one. `from_pretrained` then
    builds the policy and calls `load_state_dict`. Errors while reading the weights raise.

    Only a class that sets `_supports_meta_load` in its own body takes this path, so a subclass has to opt in.
    That promises the constructor never reads, moves or holds on to a parameter, that
    `_fix_pytorch_state_dict_keys` handles each key on its own and only renames, copies or drops values (this
    path calls it once per key and uses only names and shapes), and that `_prepare_pretrained_state_dict`,
    which this path skips, changes nothing in a complete checkpoint.
    """
    # Read from the class itself, so that a subclass does not inherit its parent's promise.
    if not vars(policy_cls).get("_supports_meta_load", False):
        return None
    try:
        checkpoint = safe_open(model_file, framework="pt", device="cpu")
    except Exception:
        return None
    with checkpoint:
        with _parameters_on_meta():
            policy = policy_cls(config, **kwargs)

        def remap(key: str, shape: list[int]) -> dict[str, torch.Size]:
            fixed = policy._fix_pytorch_state_dict_keys({key: torch.empty(shape, device="meta")}, config)
            return {k if k.startswith("model.") else f"model.{k}": v.shape for k, v in fixed.items()}

        # A file key gives zero, one or two model names: the fixes drop some keys and copy lm_head.
        names = {key: remap(key, checkpoint.get_slice(key).get_shape()) for key in checkpoint.keys()}  # noqa: SIM118
        shapes = {name: shape for fixed in names.values() for name, shape in fixed.items()}
        if (
            shapes != {name: tensor.shape for name, tensor in policy.state_dict().items()}
            or _shares_parameters(policy)
            or any(buffer.is_meta for buffer in policy.buffers())
        ):
            return None
        tensors = ((name, checkpoint.get_tensor(key)) for key, fixed in names.items() for name in fixed)
        _load_state_dict_into_meta_model(policy, tensors, config.device)
    # Buffers the constructor computed, like rotary tables, are not in the checkpoint and are still on the CPU.
    policy.model.to(config.device)
    print("All keys loaded successfully!")
    return policy
