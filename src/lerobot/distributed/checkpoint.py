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
"""Sharding-aware checkpoint primitives.

Two artifact channels with distinct owners:

- the **distributable** ``model.safetensors``: produced by ``PreTrainedPolicy.save_pretrained``
  through :func:`full_model_state_dict` — a collective full gather when the model is sharded
  (a PEFT run writes its adapter, ``adapter_model.safetensors``, through :func:`save_peft_adapter`);
- the **resume** channel (sharded runs): torch DCP directories written/read through accelerate's
  ``save/load_fsdp_model`` and ``save/load_fsdp_optimizer`` (``pytorch_model_fsdp_0/`` and
  ``optimizer_0/``, names imported from accelerate constants), which reshard on load across
  topology changes.

Every function that touches sharded state is a collective and must run on ALL ranks.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from lerobot.optim.ema import EMA
from lerobot.utils.constants import EMA_SHARDED_STATE

from .utils import is_main_process

if TYPE_CHECKING:
    from accelerate import Accelerator


def is_sharded_module(module: nn.Module) -> bool:
    """True when `fully_shard` owns this module's parameters (FSDP2's in-place class swap).

    Args:
        module (nn.Module): The module to inspect (a torch.compile wrapper is looked through
            via `_orig_mod`).

    Returns:
        bool: True when the module (or its compiled `_orig_mod`) is an `FSDPModule`.
    """
    from torch.distributed.fsdp import FSDPModule

    if isinstance(module, FSDPModule):
        return True
    # torch.compile wraps the sharded module; mirror accelerate's `_orig_mod` check.
    orig_mod = getattr(module, "_orig_mod", None)
    return orig_mod is not None and isinstance(orig_mod, FSDPModule)


def full_model_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """The module's full (unsharded) state dict, however its parameters are laid out.

    Sharded modules gather through torch's DCP state-dict API: a COLLECTIVE that must run on
    every rank; with ``cpu_offload=True`` the full dict materializes on the main rank only and
    every other rank receives a literal ``{}`` (runtime-verified — a
    rank-0-gated call deadlocks). Plain modules return ``module.state_dict()`` on every rank.

    Args:
        module (nn.Module): The (possibly sharded) module to read the state dict from.

    Returns:
        dict[str, torch.Tensor]: The full state dict — on the main rank only (``{}``
            elsewhere) when the module is sharded, on every rank otherwise.
    """
    if not is_sharded_module(module):
        return module.state_dict()

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(module, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def save_peft_adapter(model: nn.Module, save_directory: Path) -> None:
    """Write a PEFT model's adapter (`adapter_model.safetensors`, `adapter_config.json`).

    Collective when the model is sharded: call on all ranks. Only the main process writes.

    Args:
        model (nn.Module): The (possibly sharded) `PeftModel` to save.
        save_directory (Path): The directory the adapter files are written into.
    """
    if not is_sharded_module(model):
        if is_main_process():
            model.save_pretrained(save_directory)
        return

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    options = StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)
    state_dict = get_model_state_dict(model, options=options)  # {} off the main rank
    if is_main_process():
        model.save_pretrained(save_directory, state_dict=state_dict)


def _fsdp_plugin(accelerator: "Accelerator") -> object:
    """The accelerator's FSDP plugin, required by every DCP save/load helper below.

    Args:
        accelerator (Accelerator): The accelerator that prepared the sharded model.

    Returns:
        object: The FSDP plugin held by `accelerator.state`.

    Raises:
        RuntimeError: If the accelerator was not configured with an FSDP plugin.
    """
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if plugin is None:
        raise RuntimeError("Sharded checkpointing requires an FSDP-prepared Accelerator.")
    return plugin


def _inner_optimizer(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """The torch optimizer underneath accelerate's wrapper, for the DCP channel only.

    torch's optimizer DCP APIs call ``_init_optim_state``, which materializes an empty
    optimizer state by running one zero-gradient, zero-lr ``optimizer.step()``. Routed through
    an ``AcceleratedOptimizer`` that carries an fp16 ``GradScaler``, that dummy step goes
    through the scaler: it asserts outright while the scaler is still lazily uninitialized, and
    — once initialized — silently advances the scale and resets the growth tracker, so a
    resume would change the very loss scale it just restored. (accelerate's own
    ``Accelerator.load_state`` pre-initializes the scaler to dodge the assert and hits the
    second failure mode instead.)

    Unwrapping is safe and complete: the wrapper only interposes on ``step``/``zero_grad``,
    delegates ``state``, ``param_groups``, ``state_dict`` and ``load_state_dict`` verbatim, and
    ``accelerator.prepare()`` has already rebound the param groups to DTensors. The one thing
    bypassed is the wrapper's XLA device placement in ``load_state_dict``, which the sharded
    path (CUDA-only) never needs. Training itself keeps using the wrapper.

    Args:
        optimizer (torch.optim.Optimizer): The prepared optimizer, wrapped or not.

    Returns:
        torch.optim.Optimizer: The innermost torch optimizer.
    """
    from accelerate.optimizer import AcceleratedOptimizer

    # `while`, mirroring accelerate's own unwrap idiom: a wrapper may be nested.
    while isinstance(optimizer, AcceleratedOptimizer):
        optimizer = optimizer.optimizer
    return optimizer


def save_sharded_model(accelerator: "Accelerator", model: nn.Module, output_dir: Path) -> None:
    """Write the DCP model shards (`pytorch_model_fsdp_0/`). Collective: call on all ranks.

    Args:
        accelerator (Accelerator): The accelerator that prepared the sharded model.
        model (nn.Module): The prepared (sharded) model to save.
        output_dir (Path): The directory the shard subdirectory is created in.
    """
    from accelerate.utils import is_peft_model, save_fsdp_model

    save_fsdp_model(
        _fsdp_plugin(accelerator), accelerator, model, str(output_dir), adapter_only=is_peft_model(model)
    )


def load_sharded_model(accelerator: "Accelerator", model: nn.Module, input_dir: Path) -> None:
    """Load DCP model shards into the prepared (sharded) model. Collective: call on all ranks.

    Args:
        accelerator (Accelerator): The accelerator that prepared the sharded model.
        model (nn.Module): The prepared (sharded) model to load into.
        input_dir (Path): The directory containing the `pytorch_model_fsdp_0/` shard
            subdirectory.
    """
    from accelerate.utils import is_peft_model, load_fsdp_model
    from accelerate.utils.constants import FSDP_MODEL_NAME

    load_fsdp_model(
        _fsdp_plugin(accelerator),
        accelerator,
        model,
        str(input_dir / f"{FSDP_MODEL_NAME}_0"),
        adapter_only=is_peft_model(model),
    )


def save_sharded_optimizer(
    accelerator: "Accelerator", optimizer: torch.optim.Optimizer, model: nn.Module, output_dir: Path
) -> None:
    """Write the DCP optimizer shards (`optimizer_0/`). Collective: call on all ranks.

    Args:
        accelerator (Accelerator): The accelerator that prepared the model and optimizer.
        optimizer (torch.optim.Optimizer): The prepared optimizer to save the state from.
        model (nn.Module): The prepared (sharded) model the optimizer state is keyed by.
        output_dir (Path): The directory the shard subdirectory is created in.
    """
    from accelerate.utils import save_fsdp_optimizer

    # Unwrapped: this path can reach torch's `_init_optim_state` too. It bails out early when
    # any parameter still holds a gradient, so a save mid-accumulation-window is not the case
    # to worry about; what does reach the dummy step is an optimizer whose state is empty
    # after `zero_grad()` — every update so far skipped, or a stateless optimizer such as
    # momentum-free SGD, which never populates `state` at all. See `_inner_optimizer`.
    save_fsdp_optimizer(
        _fsdp_plugin(accelerator), accelerator, _inner_optimizer(optimizer), model, str(output_dir)
    )


def load_sharded_optimizer(
    accelerator: "Accelerator", optimizer: torch.optim.Optimizer, model: nn.Module, input_dir: Path
) -> None:
    """Load DCP optimizer shards into the prepared optimizer. Collective: call on all ranks.

    Must run AFTER ``accelerator.prepare()``: FSDP2's prepare rebinds the optimizer's param
    groups to sharded DTensors but never migrates ``optimizer.state`` — the resharding load is
    the only correct way to restore it.

    Args:
        accelerator (Accelerator): The accelerator that prepared the model and optimizer.
        optimizer (torch.optim.Optimizer): The prepared optimizer to restore the state into.
        model (nn.Module): The prepared (sharded) model the optimizer state is keyed by.
        input_dir (Path): The directory containing the `optimizer_0/` shard subdirectory.
    """
    from accelerate.utils import load_fsdp_optimizer
    from accelerate.utils.constants import OPTIMIZER_NAME

    # Unwrapped optimizer: torch's `_init_optim_state` runs a dummy step here (the state is
    # always empty on a fresh resume), which must not reach the fp16 scaler — see
    # `_inner_optimizer`.
    load_fsdp_optimizer(
        _fsdp_plugin(accelerator),
        accelerator,
        _inner_optimizer(optimizer),
        model,
        str(input_dir / f"{OPTIMIZER_NAME}_0"),
    )


def save_sharded_ema(ema: EMA, output_dir: Path) -> None:
    """Write the DCP shards of the EMA averages (`ema_state/`). Collective: call on all ranks.

    Args:
        ema (EMA): The EMA of the sharded model.
        output_dir (Path): The directory the shard subdirectory is created in.
    """
    import torch.distributed.checkpoint as dcp

    dcp.save({"ema": ema}, checkpoint_id=output_dir / EMA_SHARDED_STATE)


def load_sharded_ema(ema: EMA, input_dir: Path) -> None:
    """Load the DCP shards of the EMA averages, resharded to this topology. Collective: call on all ranks.

    A checkpoint without them restarts the EMA from the resumed weights.

    Args:
        ema (EMA): The EMA of the prepared (sharded) model to restore into.
        input_dir (Path): The directory containing the `ema_state/` shard subdirectory.
    """
    if not (input_dir / EMA_SHARDED_STATE).is_dir():
        if is_main_process():
            logging.warning(
                "No %s in the checkpoint: the EMA restarts from the resumed weights.", EMA_SHARDED_STATE
            )
        return
    import torch.distributed.checkpoint as dcp

    dcp.load({"ema": ema}, checkpoint_id=input_dir / EMA_SHARDED_STATE)  # nosec B614


def dcp_to_safetensors(dcp_dir: Path, output_dir: Path, *, delete_dcp: bool = False) -> Path:
    """Merge a DCP shard directory into a single `model.safetensors` (offline, single process).

    Thin wrapper over `accelerate.utils.merge_fsdp_weights`, which loads the shards without a
    process group, writes safetensors directly, and — when asked — removes the merged shard
    directory itself, only on the main process and only once the merge has succeeded.

    Args:
        dcp_dir (Path): The DCP shard directory to merge (e.g. `.../pytorch_model_fsdp_0`).
        output_dir (Path): The directory the merged `model.safetensors` is written into.
        delete_dcp (bool): Whether to remove the shard directory once it has been merged.
            Defaults to False.

    Returns:
        Path: The written `model.safetensors` file's path.
    """
    from accelerate.utils import merge_fsdp_weights
    from accelerate.utils.constants import SAFE_WEIGHTS_NAME

    merge_fsdp_weights(
        str(dcp_dir), str(output_dir), safe_serialization=True, remove_checkpoint_dir=delete_dcp
    )
    return output_dir / SAFE_WEIGHTS_NAME
