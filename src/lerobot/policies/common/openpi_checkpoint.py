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
from collections.abc import Callable, Iterator, Mapping
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
    """Whether a meta-device load is building a policy in this thread: if so, do not move it.

    Returns:
        `bool`: True inside the constructor that `load_complete_checkpoint` or `load_checkpoint_modules` runs,
        False everywhere else.
    """
    return _meta_init.replacements is not None


def _load_state_dict_into_meta_model(
    model: nn.Module, state_dict: dict[str, Tensor], device: str | None, strict: bool = True
) -> None:
    """Load a complete checkpoint into a model built under `_parameters_on_meta`.

    `state_dict` must have exactly the model's keys, or, with `strict=False`, some of them. Each tensor is copied into the dtype the constructor gave
    the matching parameter or buffer, on the device it was read on, then moved to `device`, so the
    constructor's precision choices are kept and, for a GPU, the host holds one copied tensor at a time.
    `assign=True` would give each name of a parameter shared between modules its own tensor, so
    `load_complete_checkpoint` refuses such a model and `load_modules` passes one name of each.
    `load_complete_checkpoint` has also checked that no buffer is on the meta device.
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
    model.load_state_dict(tensors, strict=strict, assign=True)


def _log_fallback(reason: str, level: int = logging.INFO) -> None:
    logger.log(level, "Loading the weights the regular way: %s.", reason)


def _convert_checkpoint(policy: nn.Module, checkpoint: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map `checkpoint` onto the names of the policy's weights, as the regular load does."""
    state_dict = policy._convert_openpi_state_dict(checkpoint)
    # load_state_dict renames these in a pre-hook, too late for the checks that read the names first.
    rename_legacy_vision_model_keys(policy, state_dict, "")
    return state_dict


def load_complete_checkpoint(
    policy_cls: type[T], model_file: Callable[[], str], config: PreTrainedConfig, **kwargs
) -> T | None:
    """Build `policy_cls` with its parameters on the meta device and stream its checkpoint into them.

    This skips randomly initializing weights that the checkpoint replaces. The checkpoint goes through the
    policy's `_convert_openpi_state_dict`, as on the regular path, but on tensors that still point into the
    file, so converting it reads only the values it changes. A conversion that only renames or reuses
    tensors, as the in-tree ones do, therefore never makes a second full copy of the weights; one that
    computes new values holds them until they are copied. Weights the policy does not have are skipped, as the
    regular path skips them.

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
        state_dict = _convert_checkpoint(policy, checkpoint)
    except Exception as e:
        _log_fallback(f"cannot convert the checkpoint ({e})")
        return None
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


def load_checkpoint_modules(
    policy_cls: type[T], model_file: str, config: PreTrainedConfig, modules: list[str], **kwargs
) -> T:
    """Build `policy_cls` on the meta device and load only the weights of `modules` from `model_file`.

    For exporting one part of a policy too large for the machine's memory: only the listed submodules get real
    tensors, as `load_modules` says. The checkpoint is converted as `load_complete_checkpoint` converts it, on
    tensors that still point into the file, so only the listed weights are read. The same promises as
    `load_complete_checkpoint` apply.

    Args:
        policy_cls (`type[PreTrainedPolicy]`):
            The policy class to build. It must set `_supports_meta_load` in its own body.
        model_file (`str`):
            The local path of the checkpoint's `model.safetensors`.
        config (`PreTrainedConfig`):
            The policy's configuration. Its `device` is where the weights end up.
        modules (`list[str]`):
            The names of the submodules to load, like `model.paligemma_with_expert.gemma_expert`.
        kwargs (`dict[str, Any]`, *optional*):
            Passed on to the policy's constructor.

    Returns:
        `PreTrainedPolicy`: The policy, as `load_modules` returns it.

    Raises:
        ValueError: If `policy_cls` does not support loading on the meta device.
        AttributeError: If a listed name is not a submodule.
        RuntimeError: As `load_modules` raises it with `strict=False`, before anything is copied.
    """
    if not vars(policy_cls).get("_supports_meta_load", False):
        raise ValueError(f"{policy_cls.__name__} does not support loading on the meta device.")
    with _parameters_on_meta():
        policy = policy_cls(config, **kwargs)
    return load_modules(policy, _convert_checkpoint(policy, load_file(model_file)), modules, config.device)


def load_modules(
    policy: T, state_dict: Mapping[str, Tensor], modules: list[str], device: str | None, strict: bool = False
) -> T:
    """Fill the `modules` of `policy`, built on the meta device, from `state_dict`, and drop the rest.

    `state_dict` maps names in the policy's state dict to tensors and may hold more than `modules` needs. Each
    weight of a listed submodule is copied in the dtype the constructor gave it. Every other submodule holding
    parameters is replaced by None, as the policy already does for modules it never runs, so code that reaches
    one fails instead of reading random weights, and an export of one part holds only its own weights. A
    parameter or buffer still on the meta device, such as one held directly by a module on the way to a listed
    one, is replaced by None too. A listed submodule and the modules on its way are kept, even without
    parameters. A parameter shared between modules is copied once, from whichever of its names `state_dict`
    holds, and stays shared. The result is for running or exporting the listed parts: saving it writes only
    their weights.

    Args:
        policy (`nn.Module`):
            The policy, built under `parameters_on_meta`.
        state_dict (`Mapping[str, Tensor]`):
            Tensors under the names of the policy's state dict.
        modules (`list[str]`):
            The names of the submodules to load.
        device (`str`, *optional*):
            Where the loaded weights end up.
        strict (`bool`, *optional*, defaults to `False`):
            Whether to raise, rather than log, for a tensor that the policy lacks, and to raise for a shared
            parameter of a listed submodule that `state_dict` holds under more than one name.

    Returns:
        `nn.Module`: The policy, in evaluation mode as from `from_pretrained`, on `device`.

    Raises:
        AttributeError: If a listed name is not a submodule.
        RuntimeError: Before anything is copied, if a listed submodule has a weight or a persistent buffer
            that `state_dict` does not hold or holds with another shape, or a non-persistent buffer still on
            the meta device. Only with `strict`, also if `state_dict` holds a tensor that the policy lacks, or
            holds a shared parameter of a listed submodule under more than one name. It names every problem.
    """
    listed = {name: policy.get_submodule(name) for name in modules}
    prefixes = tuple(f"{name}." for name in modules)
    targets = policy.state_dict()
    # A tensor on the meta device is a placeholder, like one a conversion fills in from the policy itself.
    held = {name: tensor for name, tensor in state_dict.items() if name in targets and not tensor.is_meta}
    chosen = {name: tensor for name, tensor in held.items() if name.startswith(prefixes)}
    unexpected = [name for name in state_dict if name not in targets]
    aliases: dict[int, list[str]] = {}
    for name, parameter in policy.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    twins: dict[str, str] = {}
    duplicated = []
    for names in aliases.values():
        wanted = [name for name in names if name.startswith(prefixes)]
        sources = [name for name in names if name in held]
        if len(names) > 1 and wanted and sources:
            if len(sources) > 1:
                duplicated.append(", ".join(sources))
            for name in wanted:
                chosen.pop(name, None)
            chosen[wanted[0]] = held[sources[0]]
            twins.update(dict.fromkeys(wanted[1:], wanted[0]))
    missing = []
    for name, module in listed.items():
        weights = dict(module.named_parameters(remove_duplicate=False))
        unfilled = [p for p in weights if f"{name}.{p}" not in twins]
        # A persistent buffer the file lacks would silently keep the value its constructor gave it.
        unfilled += [b for b in module.state_dict() if b not in weights]
        unfilled += [b for b, buffer in module.named_buffers() if buffer.is_meta]
        missing += [f"{name}.{n}" for n in unfilled if f"{name}.{n}" not in chosen]
    problems = [f"does not hold {', '.join(dict.fromkeys(missing))}"] if missing else []
    problems += [
        f"holds {name} with shape {tuple(tensor.shape)} instead of {tuple(targets[name].shape)}"
        for name, tensor in chosen.items()
        if tensor.shape != targets[name].shape
    ]
    if strict:
        problems += [f"holds {name}, which the policy lacks" for name in unexpected]
        problems += [f"holds one shared weight as {names}" for names in duplicated]
    elif unexpected:
        log_model_loading_keys([], unexpected)
    if problems:
        raise RuntimeError(f"The checkpoint {'; '.join(problems)}.")
    _load_state_dict_into_meta_model(policy, chosen, device, strict=False)
    for name, first in twins.items():
        module, _, leaf = name.rpartition(".")
        policy.get_submodule(module).register_parameter(leaf, policy.get_parameter(first))
    policy.eval()
    _drop_unloaded(policy, modules)
    # Buffers the constructor computed, like rotary tables, are not in the checkpoint and are still on the CPU.
    return policy.to(device)


def _drop_unloaded(module: nn.Module, modules: list[str], prefix: str = "") -> None:
    for name, tensor in [*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)]:
        if tensor.is_meta:
            setattr(module, name, None)
    for name, child in list(module.named_children()):
        path = f"{prefix}{name}"
        parameters = list(child.parameters())
        listed = any(m == path or m.startswith(f"{path}.") for m in modules)
        if parameters and all(parameter.is_meta for parameter in parameters) and not listed:
            setattr(module, name, None)
        else:
            _drop_unloaded(child, modules, f"{path}.")
