# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""CPU/GPU arrays, persistent submissions, and after-ok stage finalizers."""

import logging
import math
import os
import re
import shlex
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from filelock import FileLock

from lerobot.data_processing.configs import RuntimeConfig
from lerobot.data_processing.planner import load_module
from lerobot.data_processing.runtime import finalize_stage
from lerobot.data_processing.types import Resources, fingerprint
from lerobot.data_processing.worker import accepted_in_shard

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlurmScripts:
    worker: Path
    finalizer: Path
    groups: int
    resources: Resources
    job_args: tuple[str, ...]


def render_slurm(store, plan, runtime: RuntimeConfig) -> SlurmScripts:
    """Write operational scripts separately from immutable scientific identities."""
    cfg = runtime.slurm
    resources = load_module(plan.factory, plan.config).spec.resources
    groups = min(runtime.workers, plan.shards)
    run_uri = str(Path(store.root).resolve()) if store.is_local else store.uri
    working_directory = (cfg.working_directory or Path.cwd()).resolve()
    import draccus

    directory = (
        cfg.script_dir.resolve()
        / plan.plan_id
        / fingerprint({"runtime": draccus.encode(runtime), "cwd": str(working_directory), "run_uri": run_uri})
    )
    directory.mkdir(parents=True, exist_ok=True)
    base = [cfg.python_executable, "-m", "lerobot.jobs.processing_worker"]
    shared = ["--run-uri", run_uri, "--plan-id", plan.plan_id]
    worker = directory / "worker.sh"
    finalizer = directory / "finalize.sh"
    header = "#!/bin/sh\nset -eu\n" + "cd " + shlex.quote(str(working_directory)) + "\n"
    header += (
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"\nexport MKL_NUM_THREADS="$OMP_NUM_THREADS"\n'
    )
    worker_text = (
        header
        + "exec "
        + shlex.join(
            [
                *base,
                "worker",
                *shared,
                "--groups",
                str(max(1, groups)),
                "--batch-size",
                str(runtime.batch_size),
                "--max-retries",
                str(runtime.max_retries),
            ]
        )
        + "\n"
    )
    finalizer_text = header + "exec " + shlex.join([*base, "finalize", *shared]) + "\n"
    with FileLock(directory / ".render.lock"):
        for path, text in ((worker, worker_text), (finalizer, finalizer_text)):
            if path.exists() and path.read_text() != text:
                raise ValueError("Existing Slurm script differs from its operational config hash")
            if not path.exists():
                path.write_text(text)
    args = ["--parsable", "--ntasks=1", f"--time={cfg.time_limit}"]
    if cfg.partition:
        args.append(f"--partition={cfg.partition}")
    if cfg.account:
        args.append(f"--account={cfg.account}")
    return SlurmScripts(worker, finalizer, groups, resources, tuple(args))


def _job_id(output):
    value = output.strip().split(";", 1)[0]
    if not re.fullmatch(r"[0-9]+", value):
        raise RuntimeError("Scheduler returned an invalid parsable job ID")
    return value


def run_slurm(store, plan, runtime: RuntimeConfig):
    if store.is_local:
        with FileLock(store.path(f"{plan.prefix}/.controller.lock"), timeout=0):
            return _run_slurm_owned(store, plan, runtime)
    return _run_slurm_owned(store, plan, runtime)


def _run_slurm_owned(store, plan, runtime: RuntimeConfig):
    """Wait through one stage; resume discovers accepted work before scheduling.

    A controller advances the DAG only after real upstream finalization. It does
    not submit fictional dependent plans. Interrupting the controller leaves job
    IDs and accepted work persistent; it does not automatically cancel cluster jobs.
    """
    if all(
        len(accepted_in_shard(store, plan, shard)) == len(plan.read_shard(store, shard))
        for shard in range(plan.shards)
    ):
        return finalize_stage(store, plan)
    scripts = render_slurm(store, plan, runtime)
    cfg = runtime.slurm
    # Do not accidentally duplicate a still-running array on controller restart.
    for path in store.list(f"{plan.prefix}/slurm/*.json"):
        previous = store.read_json(path)
        if previous.get("array_job_id"):
            state = subprocess.run(
                [cfg.squeue, "--noheader", f"--jobs={previous['array_job_id']}", "--format=%i"],
                check=True,
                capture_output=True,
                text=True,
            )
            if state.stdout.strip():
                raise RuntimeError(
                    f"Slurm array {previous['array_job_id']} is still active; wait or cancel it explicitly before resuming"
                )
    concurrency = min(scripts.groups, cfg.max_concurrent or scripts.groups)
    args = [
        cfg.sbatch,
        *scripts.job_args,
        f"--array=0-{scripts.groups - 1}%{concurrency}",
        f"--cpus-per-task={scripts.resources.cpus}",
        f"--mem={math.ceil(scripts.resources.memory_gb * 1024)}M",
        f"--output={scripts.worker.parent}/worker-%A_%a.log",
    ]
    if scripts.resources.gpus:
        args.append(f"--gpus-per-task={scripts.resources.gpus}")
    if cfg.requeue:
        args.append("--requeue")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    submission = subprocess.run(
        [*args, str(scripts.worker)], check=True, capture_output=True, text=True, env=environment
    )
    job_id = _job_id(submission.stdout)
    audit = {
        "array_job_id": job_id,
        "groups": scripts.groups,
        "resources": asdict(scripts.resources),
        "worker_script": str(scripts.worker),
        "finalizer_script": str(scripts.finalizer),
    }
    logger.info("Processing stage %s: Slurm array %s", plan.plan_id, job_id)
    try:
        store.put_json(f"{plan.prefix}/slurm/{uuid.uuid4().hex}.json", audit)
    except Exception as exc:
        raise RuntimeError(
            f"Slurm array {job_id} was submitted but its ID could not be persisted; inspect/cancel it explicitly"
        ) from exc
    # Cancel an impossible dependency rather than leaving --wait stuck forever.
    finalizer = subprocess.run(
        [
            cfg.sbatch,
            *scripts.job_args,
            "--wait",
            f"--dependency=afterok:{job_id}",
            "--kill-on-invalid-dep=yes",
            "--cpus-per-task=1",
            "--mem=4096M",
            f"--output={scripts.finalizer.parent}/finalizer-%j.log",
            str(scripts.finalizer),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    _job_id(finalizer.stdout)
    return finalize_stage(store, plan)
