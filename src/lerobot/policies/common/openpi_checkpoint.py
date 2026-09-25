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

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn.modules.module import register_module_parameter_registration_hook
from torch.utils.weak import WeakIdKeyDictionary

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import T
from lerobot.policies.utils import log_model_loading_keys, rename_legacy_vision_model_keys

logger = logging.getLogger(__name__)


class _MetaInit(threading.local):
    replacements: WeakIdKeyDictionary | None = None


_meta_init = _MetaInit()
_hook_lock = threading.Lock()
_hook_registered = False


def _parameter_to_meta(module: nn.Module, name: str, param: nn.Parameter) -> nn.Parameter | None:
    replacements = _meta_init.replacements
    if replacements is None or param.is_meta:
        return None
    if param not in replacements:
        replacements[param] = nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)
    return replacements[param]


def _register_hook_once() -> None:
    # Registered on first use rather than on import, since every policy import reaches this module, and
    # never removed, so that later blocks do not change PyTorch's process-wide hook registry while another
    # thread iterates it to register a parameter. This first registration still changes it, once.
    global _hook_registered
    with _hook_lock:
        if not _hook_registered:
            register_module_parameter_registration_hook(_parameter_to_meta)
            _hook_registered = True


@contextmanager
def _parameters_on_meta() -> Iterator[None]:
    """Create the parameters of modules built inside this block on the meta device.

    A meta tensor has a shape and a dtype but no memory. Modules like `nn.Linear` allocate a parameter and
    register it before they initialize it, so building them here frees that memory at once and skips
    randomly initializing weights that a checkpoint will replace; `nn.Parameter(torch.randn(n))` is still
    drawn before it moves. Buffers and plain tensors are still computed, so values built from the config
    (rotary tables, masks) stay correct, but one computed from a parameter ends up on meta. Only the thread
    that entered the block is affected, and a parameter registered in two modules stays shared.
    """
    _register_hook_once()
    previous = _meta_init.replacements
    _meta_init.replacements = WeakIdKeyDictionary() if previous is None else previous
    try:
        yield
    finally:
        _meta_init.replacements = previous


def building_on_meta() -> bool:
    """Whether `load_complete_checkpoint` is building a policy in this thread: if so, do not move it.

    Returns:
        `bool`: True inside the constructor that `load_complete_checkpoint` runs, False everywhere else.
    """
    return _meta_init.replacements is not None


def _load_state_dict_into_meta_model(
    model: nn.Module, state_dict: dict[str, Tensor], device: str | None
) -> None:
    """Load a complete checkpoint into a model built under `_parameters_on_meta`.

    `state_dict` must have exactly the model's keys. Each tensor is copied into the dtype the constructor gave
    the matching parameter or buffer, on the device it was read on, then moved to `device`, so the
    constructor's precision choices are kept and, for a GPU, the host holds one copied tensor at a time.
    `load_complete_checkpoint` has already checked that the model shares no parameter between modules, since
    `assign=True` would give each name its own tensor, and that no buffer is on the meta device.
    """
    targets = model.state_dict(keep_vars=True)
    tensors = {}
    for name, tensor in state_dict.items():
        # A device copy straight from a memory-mapped file can keep every page it read as private memory
        # while the file is mapped (measured on Jetson), a second copy of the weights; a CPU copy does not.
        # The regular load copies into contiguous parameters; `copy=True` alone would keep a view's strides.
        tensors[name] = tensor.to(
            dtype=targets[name].dtype, memory_format=torch.contiguous_format, copy=True
        ).to(device)
    model.load_state_dict(tensors, strict=True, assign=True)


def _log_fallback(reason: str, level: int = logging.INFO) -> None:
    logger.log(level, "Loading the weights the regular way: %s.", reason)


def load_complete_checkpoint(
    policy_cls: type[T], model_file: Callable[[], str], config: PreTrainedConfig, **kwargs
) -> T | None:
    """Build `policy_cls` with its parameters on the meta device and stream its checkpoint into them.

    This skips randomly initializing weights that the checkpoint replaces. The checkpoint goes through the
    policy's `_convert_openpi_state_dict`, as on the regular path, but on tensors that still point into the
    file, so converting it reads only the values it changes. A conversion that only renames or reuses
    tensors, as the in-tree ones do, therefore never makes a second full copy of the weights; one that
    computes new values holds them until they are copied. Weights the policy does not have are skipped. The
    regular path gives the same weights: it copies all the other weights, then catches the error about the
    extra ones.

    Only a class that sets `_supports_meta_load` in its own body takes this path, so a subclass has to opt in.
    That promises the constructor never reads a parameter's values, holds on to one or attaches a hook or an
    attribute to one (loading replaces each parameter), and moves nothing to another device while
    `building_on_meta()` is true (changing a dtype is fine).

    Args:
        policy_cls (`type[PreTrainedPolicy]`):
            The policy class to build. It must have a `_convert_openpi_state_dict` method.
        model_file (`Callable[[], str]`):
            Returns the local path of the checkpoint's `model.safetensors`, downloading it if needed. It is
            called only once the policy is built, so that a constructor error comes before the download.
        config (`PreTrainedConfig`):
            The policy's configuration. Its `device` is where the weights end up.
        kwargs (`dict[str, Any]`, *optional*):
            Passed on to the policy's constructor.

    Returns:
        `PreTrainedPolicy | None`: The loaded policy, or None, before loading any weight, when the constructor
        raises `NotImplementedError` on the meta device, the policy shares a parameter or has a buffer on the
        meta device, the file cannot be read or converted, or the converted checkpoint does not fill every
        weight the policy has with a tensor of the right shape read from the file. `from_pretrained` then
        loads the regular way, and each of these cases ends as it does there.
    """
    # Read from the class itself, so that a subclass does not inherit its parent's promise.
    if not vars(policy_cls).get("_supports_meta_load", False):
        _log_fallback(f"{policy_cls.__name__} does not set _supports_meta_load in its own body")
        return None
    # Warn for the next two: they mean an opted-in class does not work here, so every load would be slow.
    # PyTorch raises NotImplementedError for an operation with no meta kernel. Any other constructor error
    # is raised here, once, instead of again by the regular build.
    try:
        with _parameters_on_meta():
            policy = policy_cls(config, **kwargs)
    except NotImplementedError as e:
        _log_fallback(f"cannot build {policy_cls.__name__} on the meta device ({e})", logging.WARNING)
        return None
    shared = len(list(policy.parameters())) != len(list(policy.named_parameters(remove_duplicate=False)))
    if shared or any(buffer.is_meta for buffer in policy.buffers()):
        _log_fallback("the policy shares a parameter or has a buffer on the meta device", logging.WARNING)
        return None
    path = model_file()
    try:
        checkpoint = load_file(path)
    except Exception as e:
        _log_fallback(f"cannot read {path} ({e})")
        return None
    try:
        state_dict = policy._convert_openpi_state_dict(checkpoint)
    except Exception as e:
        _log_fallback(f"cannot convert the checkpoint ({e})")
        return None
    # load_state_dict renames these in a pre-hook, too late for the checks below.
    rename_legacy_vision_model_keys(policy, state_dict, "")
    expected = {name: tensor.shape for name, tensor in policy.state_dict().items()}
    for name, shape in expected.items():
        if name not in state_dict or state_dict[name].shape != shape:
            found = f"shape {tuple(state_dict[name].shape)}" if name in state_dict else "nothing"
            _log_fallback(f"the checkpoint has {found} for {name}, which needs shape {tuple(shape)}")
            return None
    # A value still on the meta device is the policy's own: pi0.5 keeps one for a weight the file lacks, for
    # the regular load to initialize. That is how such a checkpoint is meant to load, hence only a debug line.
    placeholder = next((name for name in expected if state_dict[name].is_meta), None)
    if placeholder is not None:
        _log_fallback(f"the policy initializes {placeholder} itself", logging.DEBUG)
        return None
    _load_state_dict_into_meta_model(
        policy, {name: tensor for name, tensor in state_dict.items() if name in expected}, config.device
    )
    # Buffers the constructor computed, like rotary tables, are not in the checkpoint and are still on the CPU.
    policy.to(config.device)
    log_model_loading_keys([], sorted(name for name in state_dict if name not in expected))
    return policy
