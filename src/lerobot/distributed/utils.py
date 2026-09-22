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
"""Rank utilities and post-`prepare()` sharding finalization."""

import logging
from typing import TYPE_CHECKING

import torch.distributed as dist
from torch import nn

if TYPE_CHECKING:
    from lerobot.distributed.parallel_dims import ParallelDims


def is_main_process() -> bool:
    """True on the process that owns rank-0-only side effects (file writes, uploads, logging).

    Torch-native on purpose: persistence code must not depend on an `Accelerator` handle —
    `_save_pretrained` and the hub publishers run in contexts that have none. Outside
    distributed runs every process is the main process.

    Returns:
        bool: True when this process is rank 0 or no process group is initialized.
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def strip_accelerate_cp_hooks(model: nn.Module) -> int:
    """Remove accelerate's context-parallel forward-pre-hooks from every module.

    When `cp_size > 1` is declared, `accelerator.prepare()` unconditionally attaches hooks that
    silently replace any `attention_mask` kwarg of `*self_attn` modules with `is_causal=True`
    (`accelerate.big_modeling._attach_context_parallel_hooks`) — mask corruption for policies
    with non-causal attention. LeRobot implements CP itself and never enters accelerate's CP
    context, so these hooks are pure hazard. Deterministically identified by their defining
    module; a version canary pins that identity.

    Args:
        model (nn.Module): The prepared model to strip the hooks from (all submodules are
            visited).

    Returns:
        int: The number of hooks removed.
    """
    removed = 0
    for module in model.modules():
        for hook_id, hook in list(module._forward_pre_hooks.items()):
            if getattr(hook, "__module__", None) == "accelerate.big_modeling":
                del module._forward_pre_hooks[hook_id]
                module._forward_pre_hooks_with_kwargs.pop(hook_id, None)
                removed += 1
    return removed


def finalize_sharded_policy(policy: nn.Module, parallel_dims: "ParallelDims") -> None:
    """Sharding correctness protocol, applied once, immediately after `accelerator.prepare()`.

    1. Strip accelerate's CP mask hooks (only attached when cp > 1 was declared).
    2. Register the policy's non-`forward` entry points (`_fsdp_forward_methods`) so FSDP2
       unshards parameters around `select_action` & co. — without this, any inference-style
       call on a sharded policy crashes on mixed Tensor/DTensor.

    No-op for DDP/single-process runs.

    Args:
        policy (nn.Module): The policy as returned by `accelerator.prepare()`.
        parallel_dims (ParallelDims): The run's resolved topology; decides whether the protocol
            applies.
    """
    if not parallel_dims.is_sharded:
        return
    if parallel_dims.cp_size > 1:
        removed = strip_accelerate_cp_hooks(policy)
        logging.info("Stripped %d accelerate context-parallel attention-mask hooks.", removed)

    from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method

    if isinstance(policy, FSDPModule):
        for method_name in getattr(type(policy), "_fsdp_forward_methods", ()):
            if callable(getattr(policy, method_name, None)):
                register_fsdp_forward_method(policy, method_name)
