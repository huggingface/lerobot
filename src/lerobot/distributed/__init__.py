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
"""Distributed-training runtime for LeRobot.

This package owns everything that turns the declarative topology in
:class:`lerobot.configs.parallelism.ParallelismConfig` into a running engine:
mesh math (:class:`~lerobot.distributed.parallel_dims.ParallelDims`), the
`Accelerator` factory (:func:`~lerobot.distributed.factory.make_accelerator`),
sharding-aware checkpoint helpers, and small rank utilities.

Setup-order contract (normative):
CP dispatch install -> activation checkpointing -> torch.compile ->
``fully_shard``/DDP (via ``accelerator.prepare``) -> optimizer rebind.
Only the last two steps are active today; CP/AC/compile are configured
placeholders wired in later rounds.
"""

from .factory import guard_against_env_interference, make_accelerator, set_fsdp_wrap_modules
from .parallel_dims import ParallelDims
from .utils import finalize_sharded_policy, is_main_process, strip_accelerate_cp_hooks

__all__ = [
    "ParallelDims",
    "finalize_sharded_policy",
    "guard_against_env_interference",
    "is_main_process",
    "make_accelerator",
    "set_fsdp_wrap_modules",
    "strip_accelerate_cp_hooks",
]
