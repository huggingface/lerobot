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
"""Declarative process topology for distributed training and inference.

The mesh convention (canonical row-major rank layout, outermost first)::

    (dp_replicate, dp_shard, ring, ulysses)

- ``dp_replicate x dp_shard`` is the data-parallel world: HSDP replicates over
  ``dp_replicate`` and shards parameters over ``dp_shard``. FSDP2's actual shard
  group folds context parallelism in (``dp_shard x ring x ulysses``), matching
  accelerate's ``dp_shard_cp`` flattening and torchtitan's ``fsdp`` axis.
- ``ring`` is the outer and ``ulysses`` the inner context-parallel dim
  (diffusers convention: ulysses all-to-all exchanges run over adjacent, typically
  NVLink-connected ranks).
- ``cfg_parallel`` (classifier-free-guidance parallelism) is a branch-parallel,
  inference-only dim that sits between dp and the sequence dims. It never
  affects weight sharding or checkpoints.

This module is pure configuration: plain-typed dataclasses that draccus can
round-trip through the CLI and ``train_config.json``. Runtime objects (device
meshes, process groups) live in :mod:`lerobot.distributed`.
"""

import os
from dataclasses import dataclass, field


@dataclass
class ContextParallelConfig:
    """Ring x Ulysses context parallelism (sequence parallelism for attention).

    Both degrees are configured placeholders in this release: the CP engine is not implemented
    yet, and enabling either degree > 1 fails fast at config validation. The fields exist now so
    that the CLI surface, checkpoint metadata, and mesh math are stable when the engine lands.
    """

    ring_degree: int = 1
    ulysses_degree: int = 1

    def __post_init__(self) -> None:
        """Validate the declared context-parallel degrees.

        Raises:
            ValueError: If ``ring_degree`` or ``ulysses_degree`` is < 1.
        """
        if self.ring_degree < 1 or self.ulysses_degree < 1:
            raise ValueError(
                f"Context-parallel degrees must be >= 1, got ring_degree={self.ring_degree}, "
                f"ulysses_degree={self.ulysses_degree}."
            )

    @property
    def size(self) -> int:
        """Total number of ranks a full sequence is sharded across."""
        return self.ring_degree * self.ulysses_degree


@dataclass
class ParallelismConfig:
    """Degrees of every parallelism dim. Invariant: their product equals the world size.

    Degradations are expressed purely through the degrees (no mode flags):

    - single process: all degrees 1;
    - DDP: ``dp_replicate == world_size`` (auto-filled when every sharding field is left at its
      default — plain ``torchrun`` keeps today's out-of-the-box behavior);
    - FSDP: ``dp_shard > 1`` (or ``-1`` to fill the remaining world into the shard dim);
    - HSDP: ``dp_replicate > 1`` and ``dp_shard > 1``.

    ``resolve()`` turns the declared degrees into concrete ones once the world size is known and
    is the single place the world-size equation is enforced. It is called by
    :func:`lerobot.distributed.factory.make_accelerator`; the config is inert until then.
    """

    dp_replicate: int = 1
    # -1 is an explicit opt-in sentinel: shard over world_size // (dp_replicate * cp).
    dp_shard: int = 1
    context_parallel: ContextParallelConfig = field(default_factory=ContextParallelConfig)
    # Classifier-free-guidance parallelism — inference-only (cosmos/vllm-omni precedent:
    # cond/uncond branches on different ranks). Reserved for the serving round; training
    # validates it to 1. Meaningful values are 1 or 2 (Cosmos3 has two CFG branches).
    cfg_parallel: int = 1

    def __post_init__(self) -> None:
        """Validate the declared degrees (world-size-independent checks only).

        Raises:
            ValueError: If ``dp_replicate`` is < 1, ``dp_shard`` is neither >= 1 nor the
                ``-1`` infer sentinel, or ``cfg_parallel`` is not 1 or 2.
        """
        if self.dp_replicate < 1:
            raise ValueError(f"dp_replicate must be >= 1, got {self.dp_replicate}.")
        if self.dp_shard < 1 and self.dp_shard != -1:
            raise ValueError(f"dp_shard must be >= 1, or -1 to infer, got {self.dp_shard}.")
        if self.cfg_parallel not in (1, 2):
            raise ValueError(f"cfg_parallel must be 1 or 2, got {self.cfg_parallel}.")

    @property
    def cp_size(self) -> int:
        """Total context-parallel size (``ring_degree * ulysses_degree``)."""
        return self.context_parallel.size

    @property
    def is_sharded(self) -> bool:
        """True when the run uses FSDP2 (parameters sharded); selects the sharded engine path."""
        return self.dp_shard != 1 or self.cp_size > 1

    @property
    def is_replicated_only(self) -> bool:
        """True for plain DDP (weights replicated, no sharding)."""
        return not self.is_sharded and self.dp_replicate > 1

    @property
    def dp_world_size(self) -> int:
        """Number of distinct data-parallel workers (batches are sharded this many ways).

        Returns:
            int: ``dp_replicate * dp_shard``.

        Raises:
            RuntimeError: If accessed while ``dp_shard`` is still the ``-1`` sentinel, i.e.
                before :meth:`resolve` has bound the degrees to a world size.
        """
        if self.dp_shard == -1:
            raise RuntimeError("dp_world_size is undefined before resolve() fills dp_shard=-1.")
        return self.dp_replicate * self.dp_shard

    def resolve(self, world_size: int) -> None:
        """Bind the declared degrees to a concrete world size (idempotent).

        Fills the ``dp_shard=-1`` sentinel, auto-fills ``dp_replicate`` for the DDP degradation,
        and enforces ``dp_replicate * dp_shard * cp == world_size`` with every degree echoed on
        failure.

        Args:
            world_size (int): Total number of launched processes (torchrun's ``WORLD_SIZE``).

        Raises:
            ValueError: If a context-parallel degree is > 1 (the CP engine is not implemented
                yet), if ``dp_shard=-1`` cannot be inferred because ``world_size`` is not
                divisible by ``dp_replicate * cp``, or if the resolved degrees do not multiply
                to ``world_size``.
        """
        if self.cp_size > 1:
            raise ValueError(
                "Context parallelism is not implemented yet: ring_degree and ulysses_degree "
                "must be 1. The fields are reserved for the CP engine round."
            )
        if self.is_sharded:
            if self.dp_shard == -1:
                self.dp_shard, remainder = divmod(world_size, self.dp_replicate * self.cp_size)
                if remainder or self.dp_shard < 1:
                    raise ValueError(
                        f"Cannot infer dp_shard: world_size={world_size} is not divisible by "
                        f"dp_replicate={self.dp_replicate} * cp={self.cp_size}."
                    )
        elif self.dp_replicate == 1:
            # Untouched config on a multi-process launch: fill the DDP degradation.
            self.dp_replicate = world_size
        total = self.dp_replicate * self.dp_shard * self.cp_size
        if total != world_size:
            raise ValueError(
                f"Parallelism degrees do not multiply to the world size: dp_replicate="
                f"{self.dp_replicate} * dp_shard={self.dp_shard} * ring="
                f"{self.context_parallel.ring_degree} * ulysses="
                f"{self.context_parallel.ulysses_degree} = {total} != WORLD_SIZE={world_size}."
            )


def world_size_from_env() -> int:
    """World size as set by torchrun (or 1 outside distributed launches).

    Returns:
        int: The ``WORLD_SIZE`` environment variable, or 1 when unset.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))
