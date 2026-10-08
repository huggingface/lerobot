# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Serializable processing controls, independent of training and scheduler flavors."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .types import checked_name


@dataclass
class SlurmConfig:
    partition: str | None = None
    account: str | None = None
    time_limit: str = "02:00:00"
    max_concurrent: int | None = None
    script_dir: Path = Path(".lerobot_slurm")
    working_directory: Path | None = None
    python_executable: str = sys.executable
    sbatch: str = "sbatch"
    squeue: str = "squeue"
    requeue: bool = True

    def __post_init__(self):
        if self.max_concurrent is not None and self.max_concurrent < 1:
            raise ValueError("Slurm concurrency must be positive")


@dataclass
class RuntimeConfig:
    backend: Literal["local", "slurm", "hf_jobs"] = "local"
    mode: Literal["run", "plan", "resume"] = "run"
    run_uri: str | None = None
    workers: int = 1
    batch_size: int = 16
    shard_size: int = 64
    max_retries: int = 2
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

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
