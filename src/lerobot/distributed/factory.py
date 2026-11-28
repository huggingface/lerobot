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
"""The `Accelerator` factory — the only place accelerate gets configured.

`torchrun` is the launcher; every accelerate parameter comes from `TrainPipelineConfig`
(`cfg.parallelism` + `cfg.accelerator`) so a run is reproducible from its `train_config.json`
alone. `accelerate launch` without a `--config_file` remains equivalent (it only sets rendezvous
env vars in that mode); the yaml flow is superseded.
"""

import logging
import os
from typing import TYPE_CHECKING

from torch import nn

from lerobot.configs.parallelism import world_size_from_env
from lerobot.configs.train import CheckpointFormat, TrainPipelineConfig

from .utils import unwrap_peft_model

if TYPE_CHECKING:
    from accelerate import Accelerator
    from accelerate.utils import FullyShardedDataParallelPlugin

    from lerobot.policies.pretrained import PreTrainedPolicy

# Env vars through which `accelerate launch --config_file` (or a stray shell) would configure
# accelerate behind the config system's back, making train_config.json lie about what ran.
_ACCELERATE_ENV_VARS = (
    "ACCELERATE_USE_FSDP",
    "ACCELERATE_USE_PARALLELISM_CONFIG",
    "ACCELERATE_GRADIENT_ACCUMULATION_STEPS",
)
_ENV_OVERRIDE = "LEROBOT_ALLOW_ACCELERATE_ENV"


def guard_against_env_interference() -> None:
    """Hard-error when accelerate-configuring env vars are set.

    A silently env-overridden "reproducible" config is worse than a stop: users migrating from
    the old `accelerate launch --config_file fsdp.yaml` flow get a precise error instead of a
    config that lies. Set LEROBOT_ALLOW_ACCELERATE_ENV=1 to acknowledge and proceed.

    Raises:
        RuntimeError: If any accelerate-configuring environment variable is set and the
            LEROBOT_ALLOW_ACCELERATE_ENV override is not.
    """
    if os.environ.get(_ENV_OVERRIDE):
        return
    offending = sorted(name for name in _ACCELERATE_ENV_VARS if name in os.environ)
    if offending:
        raise RuntimeError(
            f"Accelerate-configuring environment variables are set: {', '.join(offending)}. "
            "LeRobot manages accelerate exclusively through TrainPipelineConfig "
            "(--parallelism.* / --accelerator.*); launch with plain torchrun and remove these "
            "variables (the `accelerate launch --config_file` flow is superseded), or set "
            f"{_ENV_OVERRIDE}=1 to acknowledge that they may override your config."
        )


def make_accelerator(cfg: TrainPipelineConfig) -> "Accelerator":
    """Resolve the topology against the launched world and build the `Accelerator`.

    Must run once per process, before any other component needs the device or the process
    group (`Accelerator.__init__` initializes both and builds the device mesh).

    Args:
        cfg (TrainPipelineConfig): The full training config; `cfg.parallelism` is resolved in
            place against the launched world size and `cfg.accelerator` builds the result.

    Returns:
        Accelerator: The configured accelerator, with device and process group initialized.

    Raises:
        ValueError: If `cfg.checkpoint_format` requires DCP but the topology resolved to a
            non-sharded run.
    """
    guard_against_env_interference()
    cfg.parallelism.resolve(world_size_from_env())
    # The parse-time format check ran against the declared degrees, where the dp_shard=-1
    # sentinel counts as sharded; it may resolve to an unsharded run (e.g. -1 at world size 1).
    # Re-check against the concrete degrees so the recorded format never lies about the
    # artifacts a checkpoint will actually contain.
    if (
        cfg.checkpoint_format in (CheckpointFormat.DCP, CheckpointFormat.SAFETENSORS_DCP)
        and not cfg.parallelism.is_sharded
    ):
        raise ValueError(
            f"checkpoint_format={cfg.checkpoint_format.value} requires a sharded run, but the "
            f"topology resolved to a non-sharded one (dp_replicate={cfg.parallelism.dp_replicate}, "
            f"dp_shard={cfg.parallelism.dp_shard}); non-sharded checkpoints are always safetensors."
        )
    return cfg.accelerator.build(
        cfg.parallelism,
        cpu=cfg.trainable_config.device == "cpu",
    )


def set_fsdp_wrap_modules(accelerator: "Accelerator", policy: "PreTrainedPolicy") -> None:
    """Resolve the FSDP wrap-unit class names onto the plugin before `accelerator.prepare()`.

    Resolution order: user override (`--accelerator.fsdp.wrap_modules`, already on the plugin)
    -> the policy's `_fsdp_wrap_modules` declaration -> hard error. Root-only wrapping — the
    silent default when no wrap source exists — is never accepted: it quietly forfeits all
    sharding memory savings.

    The size-based policy (`--accelerator.fsdp.min_num_params`) needs no class names, unless it
    would select an `nn.ModuleList` or `nn.ModuleDict`: see `_avoid_unshardable_size_units`.
    No-op for non-sharded runs (no fsdp plugin).

    Args:
        accelerator (Accelerator): The accelerator whose FSDP plugin receives the wrap-unit
            class names.
        policy (PreTrainedPolicy): The trainable whose class may declare `_fsdp_wrap_modules`.
            Under PEFT the declaration is read from the policy the `PeftModel` wraps.

    Raises:
        ValueError: If sharded class-based wrapping is configured but neither a user override
            nor a policy declaration supplies wrap-unit class names, or if the size-based policy
            selects a container and no unambiguous wrap-unit classes replace it.
    """
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if plugin is None:
        return
    if plugin.min_num_params:
        _avoid_unshardable_size_units(plugin, policy)
        return
    if plugin.transformer_cls_names_to_wrap:  # user override, set at build time
        return
    policy = unwrap_peft_model(policy)
    # getattr, not attribute access: non-policy trainables (no `_fsdp_wrap_modules` attribute)
    # must reach the actionable error below, not an AttributeError.
    declared = getattr(type(policy), "_fsdp_wrap_modules", None)
    if not declared:
        raise ValueError(
            f"Policy '{type(policy).__name__}' declares no FSDP wrap units. Sharded training "
            "requires wrap-unit class names: set --accelerator.fsdp.wrap_modules='[\"MyBlock\"]' "
            "(or --accelerator.fsdp.min_num_params for a size-based policy), or declare "
            "`_fsdp_wrap_modules` on the policy class."
        )
    plugin.transformer_cls_names_to_wrap = list(declared)


def _is_unshardable_container(module: nn.Module) -> bool:
    """Return whether `fully_shard` rejects this module, using torch's own check."""
    return isinstance(module, (nn.ModuleList, nn.ModuleDict)) and type(module).forward is nn.Module.forward


def _wrap_unit_class(module: nn.Module) -> type[nn.Module]:
    """Return the class accelerate matches for this module, looking through activation checkpointing."""
    return type(getattr(module, "_checkpoint_wrapped_module", module))


def _avoid_unshardable_size_units(plugin: "FullyShardedDataParallelPlugin", model: nn.Module) -> None:
    """Switch the size-based policy to class names when it would select a container.

    accelerate's FSDP2 size policy selects every non-root module with more than `min_num_params`
    parameters, including `nn.ModuleList` and `nn.ModuleDict`, which `fully_shard` rejects (#5475).
    In that case the classes of the other selected modules become the wrap units, and every
    instance of those classes is wrapped. A size policy that selects no container is unchanged.

    Args:
        plugin (FullyShardedDataParallelPlugin): The size-based plugin, updated in place.
        model (nn.Module): The module passed to `accelerator.prepare()`, PEFT wrapper included.

    Raises:
        ValueError: If only containers are selected, or if a selected class name also names
            another class in the model, since accelerate resolves wrap units by class name.
    """
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    min_num_params = plugin.min_num_params
    modules = list(model.modules())
    # accelerate always shards the root itself, so it is never a candidate.
    selected = [m for m in modules[1:] if sum(p.numel() for p in m.parameters()) > min_num_params]
    if not any(_is_unshardable_container(m) for m in selected):
        return

    selected_classes: dict[str, set[type[nn.Module]]] = {}
    for module in selected:
        if not _is_unshardable_container(module):
            cls = _wrap_unit_class(module)
            selected_classes.setdefault(cls.__name__, set()).add(cls)
    hint = "set --accelerator.fsdp.wrap_modules to the class names of the policy's blocks"
    if not selected_classes:
        raise ValueError(
            f"fsdp.min_num_params={min_num_params} selects only ModuleList/ModuleDict containers, "
            f"which fully_shard rejects. Lower it, or {hint}."
        )

    # accelerate resolves a class name to the first module with that name in `model.modules()` order.
    resolved: dict[str, type[nn.Module]] = {}
    for module in modules:
        cls = _wrap_unit_class(module)
        resolved.setdefault(cls.__name__, cls)
    ambiguous = sorted(name for name, classes in selected_classes.items() if classes != {resolved[name]})
    if ambiguous:
        raise ValueError(
            f"fsdp.min_num_params={min_num_params} selects ModuleList/ModuleDict containers, which "
            f"fully_shard rejects, and their blocks cannot be named unambiguously: {ambiguous} each "
            f"name more than one class in the model. Instead, {hint}."
        )

    plugin.auto_wrap_policy = transformer_auto_wrap_policy
    plugin.transformer_cls_names_to_wrap = sorted(selected_classes)
    logging.info(
        "fsdp.min_num_params=%d selects ModuleList/ModuleDict containers, which fully_shard rejects. "
        "Wrapping these classes instead: %s",
        min_num_params,
        plugin.transformer_cls_names_to_wrap,
    )
