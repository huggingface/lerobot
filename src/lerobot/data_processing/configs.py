# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Serializable processing controls, independent of training and scheduler flavors."""

from dataclasses import dataclass, field
from typing import Any, Literal

from .types import checked_name


@dataclass
class RuntimeConfig:
    backend: Literal["local", "slurm", "hf_jobs"] = "local"
    mode: Literal["run", "plan", "resume"] = "run"
    run_uri: str | None = None
    workers: int = 1
    batch_size: int = 16
    shard_size: int = 64
    max_retries: int = 2

    def __post_init__(self):
        if self.workers < 1 or self.batch_size < 1 or self.shard_size < 1 or self.max_retries < 0:
            raise ValueError("Invalid runtime limits")


@dataclass
class StageConfig:
    id: str
    factory: str
    config: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()

    def __post_init__(self):
        checked_name(self.id)
