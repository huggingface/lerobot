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
"""Execution-runtime configuration: everything handed to (or applied by) the `Accelerator`.

Each sub-config mirrors the plain-typed subset of the corresponding accelerate object and
builds it at runtime (the way ``OptimizerConfig.build()`` constructs a ``torch.optim.Optimizer``),
so the whole tree round-trips through the CLI and ``train_config.json`` and parsing a config
never imports accelerate.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from lerobot.configs.parallelism import ParallelismConfig

if TYPE_CHECKING:
    from accelerate import Accelerator
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        FullyShardedDataParallelPlugin,
        GradientAccumulationPlugin,
    )


@dataclass
class FSDPConfig:
    """Mirror of the `FullyShardedDataParallelPlugin` subset LeRobot supports (FSDP2 only).

    Exactly one wrap policy applies: `wrap_modules` (module *class names* forming the FSDP
    units — and, later, the activation-checkpointing units) or `min_num_params` (size-based).
    When both are None, the policy's own `_fsdp_wrap_modules` declaration is used; a run where
    no wrap source exists at all fails loudly rather than silently wrapping only the root.
    """

    reshard_after_forward: bool = True
    wrap_modules: list[str] | None = None
    min_num_params: int | None = None
    cpu_offload: bool = False
    # Regex matched against module FQNs to exclude their parameters from sharding.
    ignored_modules: str | None = None

    def __post_init__(self) -> None:
        """Validate the wrap-policy fields.

        Raises:
            ValueError: If both ``wrap_modules`` and ``min_num_params`` are set (they are
                mutually exclusive wrap policies), or if ``min_num_params`` is < 1.
        """
        if self.wrap_modules is not None and self.min_num_params is not None:
            raise ValueError(
                "fsdp.wrap_modules and fsdp.min_num_params are mutually exclusive wrap policies."
            )
        if self.min_num_params is not None and self.min_num_params < 1:
            raise ValueError(f"fsdp.min_num_params must be >= 1, got {self.min_num_params}.")

    def build_plugin(self) -> "FullyShardedDataParallelPlugin":
        """Build the FSDP2 plugin for `Accelerator(fsdp_plugin=...)`.

        Returns:
            FullyShardedDataParallelPlugin: FSDP2 (`fsdp_version=2`) plugin carrying the
                mirrored wrap policy, resharding, CPU-offload, and ignored-modules settings.
        """
        from accelerate.utils import FullyShardedDataParallelPlugin

        use_size_policy = self.min_num_params is not None
        return FullyShardedDataParallelPlugin(
            fsdp_version=2,
            reshard_after_forward=self.reshard_after_forward,
            auto_wrap_policy="size_based_wrap" if use_size_policy else "transformer_based_wrap",
            # May legitimately still be None here: the policy-declared default is applied right
            # before `accelerator.prepare()` (see lerobot.distributed.factory.set_fsdp_wrap_modules).
            transformer_cls_names_to_wrap=list(self.wrap_modules) if self.wrap_modules else None,
            min_num_params=self.min_num_params,
            cpu_offload=self.cpu_offload,
            ignored_modules=self.ignored_modules,
            # state_dict_type stays at the FSDP2 default (SHARDED_STATE_DICT) and is never
            # switched: full gathers go through torch's state-dict API, which does not consult
            # the plugin. activation_checkpointing stays False: AC is LeRobot-owned.
        )


@dataclass
class DDPConfig:
    """Mirror of the `DistributedDataParallelKwargs` subset LeRobot exposes."""

    # Today's in-script default, kept for models with conditional computation.
    find_unused_parameters: bool = True
    gradient_as_bucket_view: bool = False
    static_graph: bool = False

    def build_kwargs_handler(self) -> "DistributedDataParallelKwargs":
        """Build the DDP kwargs handler for `Accelerator(kwargs_handlers=[...])`.

        Returns:
            DistributedDataParallelKwargs: Handler carrying the mirrored DDP fields, applied
                by accelerate when it wraps the model in `DistributedDataParallel`.
        """
        from accelerate.utils import DistributedDataParallelKwargs

        return DistributedDataParallelKwargs(
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
            static_graph=self.static_graph,
        )


@dataclass
class GradientAccumulationConfig:
    """Mirror of the `GradientAccumulationPlugin` subset LeRobot supports.

    Only the step count is a knob. ``sync_with_dataloader`` is pinned to False by
    :meth:`build_plugin`: the training loop cycles a finite dataloader, so accelerate's default
    of syncing at every dataloader end would force an optimizer step at every dataset epoch
    boundary instead of every ``steps`` micro-batches.
    """

    steps: int = 1

    def __post_init__(self) -> None:
        """Validate the accumulation step count.

        Raises:
            ValueError: If ``steps`` is < 1.
        """
        if self.steps < 1:
            raise ValueError(f"gradient_accumulation.steps must be >= 1, got {self.steps}.")

    def build_plugin(self) -> "GradientAccumulationPlugin":
        """Build the plugin for `Accelerator(gradient_accumulation_plugin=...)`.

        A named plugin argument, not a `kwargs_handlers` entry: accelerate consumes this object
        through its dedicated constructor parameter — the `KwargsHandler` base class only lends
        it `to_kwargs()`, so the consumption site, not the inheritance, decides its role.

        Returns:
            GradientAccumulationPlugin: Carrying the mirrored step count, with
                ``sync_with_dataloader=False`` pinned (see the class docstring).
        """
        from accelerate.utils import GradientAccumulationPlugin

        return GradientAccumulationPlugin(num_steps=self.steps, sync_with_dataloader=False)


@dataclass
class CompileConfig:
    """torch.compile knobs — a configured placeholder: wiring lands in a later round.

    The setup-order contract it will follow is already fixed: compile applies
    after CP dispatch install and activation checkpointing, before `fully_shard`, regionally
    (per wrap unit) — the only combination proven with FSDP2.
    """

    enabled: bool = False
    backend: str = "inductor"
    mode: str | None = None
    regional: bool = True


class ActivationCheckpointingMode(str, Enum):
    NONE = "none"
    FULL = "full"


@dataclass
class ActivationCheckpointingConfig:
    """Activation-checkpointing knobs — a configured placeholder: wiring lands in a later round.

    AC units will coincide with the FSDP wrap units (one declaration drives both), applied
    before torch.compile and `fully_shard` (the same ordering contract as CompileConfig).
    """

    mode: ActivationCheckpointingMode = ActivationCheckpointingMode.NONE


@dataclass
class AcceleratorConfig:
    """Builds the `Accelerator` — the runtime counterpart of the `parallelism` topology.

    `mixed_precision` selects accelerate-native AMP for DDP/single-GPU runs and the FSDP2
    `MixedPrecisionPolicy` for sharded runs (accelerate derives it). Sharded runs support
    "no" and "bf16" only; fp16's GradScaler-over-DTensor path is unverified and fails fast
    at config validation.
    """

    mixed_precision: str = "no"
    gradient_accumulation: GradientAccumulationConfig = field(default_factory=GradientAccumulationConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    ddp: DDPConfig = field(default_factory=DDPConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    activation_checkpointing: ActivationCheckpointingConfig = field(
        default_factory=ActivationCheckpointingConfig
    )

    def __post_init__(self) -> None:
        """Validate the accelerate-facing scalar fields.

        Raises:
            ValueError: If ``mixed_precision`` is not one of ``"no"``, ``"fp16"``, ``"bf16"``.
        """
        if self.mixed_precision not in ("no", "fp16", "bf16"):
            raise ValueError(
                f"mixed_precision must be one of 'no', 'fp16', 'bf16', got {self.mixed_precision!r}."
            )

    def build(self, parallelism: ParallelismConfig, *, cpu: bool = False) -> "Accelerator":
        """Translate the mirrored fields into a ready `Accelerator` (call once per process).

        `parallelism` must already be resolved against the world size. The degradation matrix
        is encoded here and nowhere else: sharded -> FSDP2 (+HSDP via the accelerate
        `ParallelismConfig` mesh), replicated-only -> DDP kwargs, single process -> plain.

        Args:
            parallelism (ParallelismConfig): The resolved process topology; selects which
                accelerate path (FSDP2 mesh, DDP kwargs handler, or plain) is configured.
            cpu (bool): Force CPU execution even when CUDA is available. Defaults to False.

        Returns:
            Accelerator: The configured accelerate entry point for this process.
        """
        from accelerate import Accelerator

        kwargs: dict = {
            # LeRobot steps its scheduler manually once per training step; accelerate must not
            # rescale scheduler stepping by num_processes.
            "step_scheduler_with_optimizer": False,
            "gradient_accumulation_plugin": self.gradient_accumulation.build_plugin(),
            "mixed_precision": self.mixed_precision,
            "cpu": cpu,
        }
        if parallelism.is_sharded:
            kwargs["fsdp_plugin"] = self.fsdp.build_plugin()
            kwargs["parallelism_config"] = _accelerate_parallelism_config(parallelism)
        elif parallelism.is_replicated_only:
            kwargs["kwargs_handlers"] = [self.ddp.build_kwargs_handler()]
        return Accelerator(**kwargs)


def _accelerate_parallelism_config(parallelism: ParallelismConfig) -> object:
    """LeRobot topology -> accelerate `ParallelismConfig`.

    CP is declared honestly (`cp_size = ring x ulysses`) so accelerate builds the canonical
    mesh, folds CP into the FSDP shard group (`dp_shard_cp`), and duplicates batches within CP
    groups. The ring/ulysses sub-structure stays private to `lerobot.distributed.ParallelDims`.

    Args:
        parallelism (ParallelismConfig): The resolved LeRobot topology to translate.

    Returns:
        object: The accelerate `ParallelismConfig` mirroring `dp_replicate`, `dp_shard`, and
            the collapsed `cp_size` (annotated as `object` so importing this module never
            imports accelerate).
    """
    from accelerate.parallelism_config import ParallelismConfig as AccelerateParallelismConfig

    return AccelerateParallelismConfig(
        dp_replicate_size=parallelism.dp_replicate,
        dp_shard_size=parallelism.dp_shard,
        cp_size=parallelism.cp_size,
    )
