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
"""Runtime mesh math derived from the declarative :class:`ParallelismConfig`.

`ParallelDims` is the training script's single source of truth for topology-derived numbers
(data-parallel world size and rank, sample accounting inputs) and — once the CP engine lands —
the owner of LeRobot's private ``(dp_replicate, dp_shard, ring, ulysses)`` mesh. It is a runtime
object and is never serialized (the config it derives from is what lands in
``train_config.json``).
"""

from dataclasses import dataclass

import torch.distributed as dist

from lerobot.configs.parallelism import ParallelismConfig


@dataclass(frozen=True)
class ParallelDims:
    """Concrete parallelism degrees bound to a world size (canonical row-major rank layout)."""

    dp_replicate: int
    dp_shard: int
    ring: int
    ulysses: int
    world_size: int
    device_type: str

    @classmethod
    def from_config(cls, cfg: ParallelismConfig, world_size: int, device_type: str) -> "ParallelDims":
        """Bind a *resolved* config to the actual runtime world size (cross-checked here).

        Args:
            cfg (ParallelismConfig): The declarative topology, already resolved via
                `ParallelismConfig.resolve(world_size)`.
            world_size (int): The launched world size the declared degrees must multiply to.
            device_type (str): The accelerator device type backing the mesh (e.g. "cuda").

        Returns:
            ParallelDims: The concrete parallelism degrees bound to this world.

        Raises:
            ValueError: If the config is unresolved (`dp_shard == -1`) or its degrees do not
                multiply to `world_size`.
        """
        total = cfg.dp_replicate * cfg.dp_shard * cfg.cp_size
        if cfg.dp_shard == -1 or total != world_size:
            raise ValueError(
                f"ParallelismConfig is not resolved against this world: dp_replicate="
                f"{cfg.dp_replicate} * dp_shard={cfg.dp_shard} * cp={cfg.cp_size} != "
                f"world_size={world_size}. Call ParallelismConfig.resolve(world_size) first "
                "(make_accelerator does this)."
            )
        return cls(
            dp_replicate=cfg.dp_replicate,
            dp_shard=cfg.dp_shard,
            ring=cfg.context_parallel.ring_degree,
            ulysses=cfg.context_parallel.ulysses_degree,
            world_size=world_size,
            device_type=device_type,
        )

    @property
    def cp_size(self) -> int:
        """Total context-parallel degree (`ring * ulysses`)."""
        return self.ring * self.ulysses

    @property
    def is_sharded(self) -> bool:
        """Whether parameters are sharded (`dp_shard > 1` or any context parallelism)."""
        return self.dp_shard > 1 or self.cp_size > 1

    @property
    def dp_world_size(self) -> int:
        """Number of distinct data-parallel workers — the divisor for all sample accounting."""
        return self.dp_replicate * self.dp_shard

    @property
    def dp_rank(self) -> int:
        """This process's data-parallel coordinate (CP peers share one dp_rank).

        With the canonical row-major layout and (ring, ulysses) innermost, CP peers are
        contiguous global ranks, so the dp coordinate is the integer quotient by cp_size —
        the same arithmetic accelerate's mesh-aware dataloader applies.
        """
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        return global_rank // self.cp_size

    def cp_mesh(self) -> None:
        """Private (ring, ulysses) mesh for the CP engine — reserved for the CP round.

        Raises:
            NotImplementedError: Always — context parallelism is not implemented yet.
        """
        raise NotImplementedError(
            "Context parallelism is not implemented yet; ParallelDims.cp_mesh is reserved for "
            "the CP engine round (a private mesh aligned with accelerate's cp block)."
        )
