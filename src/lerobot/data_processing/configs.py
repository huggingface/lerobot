# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Serializable processing controls, independent of training and scheduler flavors."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .types import Resources, checked_name


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
class HFJobsConfig:
    """Pinned, operator-selected images; credentials are environment references only."""

    cpu_image: str | None = None
    gpu_image: str | None = None
    cpu_flavor: str = "cpu-upgrade"
    gpu_flavor: str | None = None
    code_revision: str | None = None
    code_repository: str = "https://github.com/huggingface/lerobot.git"
    namespace: str | None = None
    bootstrap_packages: tuple[str, ...] = ()
    timeout: str = "2h"
    max_parallel: int = 4
    detach: bool = False
    secret_env: tuple[str, ...] = ()

    def __post_init__(self):
        if self.max_parallel < 1:
            raise ValueError("HF Jobs concurrency must be positive")


@dataclass
class RuntimeConfig:
    backend: Literal["local", "slurm", "hf_jobs"] = "local"
    mode: Literal["run", "plan", "resume"] = "run"
    run_uri: str | None = None
    workers: int = 1
    batch_size: int = 16
    shard_size: int = 64
    max_retries: int = 2
    # Sequential by default. Parallel admission reserves every worker's resources.
    max_parallel_stages: int = 1
    resource_budget: Resources | None = None
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    hf_jobs: HFJobsConfig = field(default_factory=HFJobsConfig)

    def __post_init__(self):
        if (
            self.workers < 1
            or self.batch_size < 1
            or self.shard_size < 1
            or self.max_retries < 0
            or self.max_parallel_stages < 1
        ):
            raise ValueError("Invalid runtime limits")
        if self.max_parallel_stages > 1 and self.resource_budget is None:
            raise ValueError("Concurrent stages require an explicit resource_budget")


@dataclass
class StageConfig:
    id: str
    factory: str
    config: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    # A source attaches quality results to each item's metadata. Missing/non-bool
    # predicates are errors; false means a recorded mask, never dropped work.
    when: str | None = None
    skip_reason: str = "quality_condition_false"

    def __post_init__(self):
        checked_name(self.id)
        if self.when is not None and (not self.when or not self.depends_on or not self.skip_reason):
            raise ValueError("A conditional stage needs a predicate, quality dependency and skip reason")
