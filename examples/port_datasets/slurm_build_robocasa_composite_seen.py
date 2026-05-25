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

"""Build the merged RoboCasa composite_seen dataset on SLURM via datatrove.

Two-phase pipeline modeled after ``slurm_port_shards.py`` +
``slurm_aggregate_shards.py``:

* Phase 1 — DOWNLOAD (parallel, 16 tasks, 1 per worker):
  Each datatrove worker downloads one of the 16 RoboCasa composite_seen task
  archives (``v1.0/target/composite/<Task>/<date>/lerobot.tar``) via
  RoboCasa's own ``download_datasets`` helper. Idempotent — already-extracted
  tasks are skipped. Network-bound, CPU-light.

* Phase 2 — AGGREGATE (single worker, depends on phase 1):
  One worker calls ``aggregate_datasets`` over the 16 extracted directories,
  producing a single combined LeRobotDataset. Validates fps / robot_type /
  features, unifies task indices, concatenates videos + parquet, recomputes
  stats. CPU + disk heavy.

When run under SLURM, phase 2 is submitted with ``depends=phase_1_executor``
so the scheduler only releases it after every download task succeeds.

Local (``--slurm 0``) execution runs the two phases sequentially in the
current process — useful for debugging on a workstation.

Usage on SLURM::

    uv run python examples/port_datasets/slurm_build_robocasa_composite_seen.py \\
        --output-dir=/scratch/${USER}/robocasa_composite_seen \\
        --hub-repo-id=${HF_USER}/robocasa_composite_seen \\
        --logs-dir=/scratch/${USER}/logs/robocasa \\
        --partition=cpu \\
        --download-cpus=4 --download-mem=8G \\
        --aggregate-cpus=16 --aggregate-mem-per-cpu=4G \\
        --push-to-hub

Local debug (sequential, single process)::

    uv run python examples/port_datasets/slurm_build_robocasa_composite_seen.py \\
        --output-dir=/tmp/robocasa_composite_seen \\
        --slurm=0 \\
        --tasks=PrepareCoffee,KettleBoiling

Prereqs: ``robocasa`` + ``robosuite`` installed (see
``docs/source/benchmarks/robocasa.mdx``); ``datatrove`` installed via the
``annotations`` extra (``uv sync --extra annotations``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

# Reuse the per-task helpers + canonical task list from the single-machine
# script so both runners share one source of truth for "what does it mean to
# download a composite_seen task". The helpers are spelled with leading
# underscores there (module-private), but the slurm runner is a legitimate
# in-tree consumer, so we alias them to clean names here.
from lerobot.scripts.build_robocasa_composite_seen import (
    COMPOSITE_SEEN_TASKS,
    _download_task as download_task,
    _require_robocasa as require_robocasa,
    _resolve_task_root as resolve_task_root,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


class DownloadRoboCasaTask(PipelineStep):
    """Phase 1 — download the task assigned to this rank.

    Each datatrove worker is given a rank in ``[0, world_size)``. With
    ``tasks == len(task_names)`` each worker owns exactly one task; with
    fewer workers than tasks, datatrove load-balances tasks across workers
    using the standard ``rank::world_size`` slicing.
    """

    def __init__(self, task_names: list[str], *, overwrite: bool = False):
        super().__init__()
        self.task_names = list(task_names)
        self.overwrite = overwrite

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        from lerobot.utils.utils import init_logging  # noqa: PLC0415

        init_logging()
        require_robocasa()

        # Standard datatrove slicing: each rank owns the subset
        # ``task_names[rank::world_size]``. When ``world_size ==
        # len(task_names)`` this is exactly one task per rank.
        my_tasks = self.task_names[rank::world_size]
        if not my_tasks:
            logger.info("rank %d/%d: no tasks assigned", rank, world_size)
            return

        for task in my_tasks:
            logger.info("rank %d/%d: downloading %s", rank, world_size, task)
            root = download_task(task, overwrite=self.overwrite)
            logger.info("rank %d/%d: %s extracted at %s", rank, world_size, task, root)


class AggregateRoboCasaShards(PipelineStep):
    """Phase 2 — merge all 16 extracted directories into one LeRobotDataset.

    ``aggregate_datasets`` parallelizes internally; only rank 0 runs the
    merge (mirrors the DROID ``slurm_aggregate_shards.py`` convention).
    """

    def __init__(
        self,
        task_names: list[str],
        *,
        output_repo_id: str,
        output_dir: Path,
        push_to_hub: bool,
        private: bool,
    ):
        super().__init__()
        self.task_names = list(task_names)
        self.output_repo_id = output_repo_id
        self.output_dir = Path(output_dir)
        self.push_to_hub = push_to_hub
        self.private = private

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        from lerobot.utils.utils import init_logging  # noqa: PLC0415

        init_logging()

        if rank != 0:
            logger.info("rank %d: aggregation runs on rank 0 only — skipping", rank)
            return

        require_robocasa()
        from lerobot.datasets import aggregate_datasets  # noqa: PLC0415
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

        # Resolve each task's local extraction root. After phase 1 these
        # all exist on disk under robocasa.macros.DATASET_BASE_DIR; if any
        # are missing, fail loudly so the operator knows phase 1 didn't
        # cleanly complete for that task.
        roots: list[Path] = []
        missing: list[str] = []
        for task in self.task_names:
            root = resolve_task_root(task)
            if not root.exists():
                missing.append(f"{task} -> {root}")
            else:
                roots.append(root)
        if missing:
            raise RuntimeError(
                "Phase 1 did not produce extracted directories for: "
                + ", ".join(missing)
                + " — re-run the download phase before aggregating."
            )

        # ``aggregate_datasets`` uses ``repo_ids`` purely for logging /
        # the unified task table when ``roots=`` is supplied; the actual
        # data is loaded from each root directly.
        repo_ids = [f"robocasa/{task}_target_human" for task in self.task_names]

        logger.info(
            "Aggregating %d source datasets into %s at %s",
            len(roots),
            self.output_repo_id,
            self.output_dir,
        )
        aggregate_datasets(
            repo_ids=repo_ids,
            aggr_repo_id=self.output_repo_id,
            roots=roots,
            aggr_root=self.output_dir,
        )
        logger.info("Aggregation complete.")

        if self.push_to_hub:
            merged = LeRobotDataset(
                repo_id=self.output_repo_id,
                root=self.output_dir,
            )
            logger.info(
                "Pushing %s to the Hub (private=%s, %d episodes, %d frames)",
                self.output_repo_id,
                self.private,
                merged.num_episodes,
                merged.num_frames,
            )
            merged.push_to_hub(
                private=self.private,
                upload_large_folder=True,
                tags=["lerobot", "robocasa", "composite_seen", "manipulation"],
            )
            logger.info(
                "Push complete: https://huggingface.co/datasets/%s",
                self.output_repo_id,
            )


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------


def make_download_executor(
    *,
    task_names: list[str],
    overwrite: bool,
    job_name: str,
    logs_dir: Path,
    workers: int,
    partition: str | None,
    cpus_per_task: int,
    mem: str,
    time: str,
    slurm: bool,
):
    """Phase-1 executor: parallel downloads, one task per worker by default."""
    pipeline = [DownloadRoboCasaTask(task_names, overwrite=overwrite)]
    logging_dir = str(logs_dir / job_name)

    if slurm:
        return SlurmPipelineExecutor(
            pipeline=pipeline,
            logging_dir=logging_dir,
            job_name=job_name,
            tasks=len(task_names),  # one shard per RoboCasa task
            workers=workers,
            time=time,
            partition=partition,
            cpus_per_task=cpus_per_task,
            sbatch_args={"mem": mem},
        )
    return LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=logging_dir,
        tasks=len(task_names),
        workers=min(workers, len(task_names)),
    )


def make_aggregate_executor(
    *,
    task_names: list[str],
    output_repo_id: str,
    output_dir: Path,
    push_to_hub: bool,
    private: bool,
    job_name: str,
    logs_dir: Path,
    partition: str | None,
    cpus_per_task: int,
    mem_per_cpu: str,
    time: str,
    slurm: bool,
    depends: SlurmPipelineExecutor | None,
):
    """Phase-2 executor: single worker, aggregates the extracted shards."""
    pipeline = [
        AggregateRoboCasaShards(
            task_names,
            output_repo_id=output_repo_id,
            output_dir=output_dir,
            push_to_hub=push_to_hub,
            private=private,
        )
    ]
    logging_dir = str(logs_dir / job_name)

    if slurm:
        return SlurmPipelineExecutor(
            pipeline=pipeline,
            logging_dir=logging_dir,
            job_name=job_name,
            tasks=1,
            workers=1,
            time=time,
            partition=partition,
            cpus_per_task=cpus_per_task,
            sbatch_args={"mem-per-cpu": mem_per_cpu},
            depends=depends,  # SLURM job dependency on phase 1
        )
    return LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=logging_dir,
        tasks=1,
        workers=1,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the merged RoboCasa composite_seen LeRobotDataset on SLURM "
            "via datatrove (download in parallel, aggregate sequentially)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O.
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local directory for the merged dataset (will be created).",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help=(
            "Hub repo_id for the merged dataset (e.g. "
            "``yourname/robocasa_composite_seen``). Required for "
            "``--push-to-hub``; also becomes the merged dataset's "
            "canonical ``repo_id``."
        ),
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the merged dataset to the Hub after aggregation.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="When pushing, create the Hub repo as private.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("./logs/robocasa"),
        help="Path to datatrove logs directory (used for stdout/stderr and "
        "phase coordination).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names overriding the default 16 "
        "composite_seen list (debug / smoke-test).",
    )
    parser.add_argument(
        "--overwrite-download",
        action="store_true",
        help="Force re-download even if the local extraction looks complete.",
    )

    # SLURM controls.
    parser.add_argument(
        "--slurm",
        type=int,
        default=1,
        help="Launch over SLURM (``1``) or locally / sequentially (``0``).",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=None,
        help="SLURM partition. A CPU partition is sufficient — no GPU needed.",
    )

    # Phase-1 (download) sizing.
    parser.add_argument(
        "--download-workers",
        type=int,
        default=16,
        help="Number of parallel SLURM workers for the download phase. "
        "Default matches the number of composite_seen tasks (16).",
    )
    parser.add_argument(
        "--download-cpus",
        type=int,
        default=4,
        help="CPUs per download worker (the work is network- and "
        "tar-extraction-bound).",
    )
    parser.add_argument(
        "--download-mem",
        type=str,
        default="8G",
        help="Total memory per download worker.",
    )
    parser.add_argument(
        "--download-time",
        type=str,
        default="06:00:00",
        help="SLURM wall-clock limit per download worker (HH:MM:SS).",
    )

    # Phase-2 (aggregate) sizing.
    parser.add_argument(
        "--aggregate-cpus",
        type=int,
        default=16,
        help="CPUs for the aggregation worker (ffmpeg + parquet I/O parallelize).",
    )
    parser.add_argument(
        "--aggregate-mem-per-cpu",
        type=str,
        default="2G",
        help="SLURM mem-per-cpu for the aggregation worker.",
    )
    parser.add_argument(
        "--aggregate-time",
        type=str,
        default="12:00:00",
        help="SLURM wall-clock limit for aggregation (HH:MM:SS). Tens-of-GB "
        "merges can take several hours.",
    )

    # Job naming.
    parser.add_argument(
        "--download-job-name",
        type=str,
        default="robocasa_dl",
        help="SLURM job name for phase 1.",
    )
    parser.add_argument(
        "--aggregate-job-name",
        type=str,
        default="robocasa_agg",
        help="SLURM job name for phase 2.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    tasks = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks
        else list(COMPOSITE_SEEN_TASKS)
    )
    if not tasks:
        raise SystemExit("No tasks selected.")

    if args.push_to_hub and not args.hub_repo_id:
        raise SystemExit("--push-to-hub requires --hub-repo-id.")

    output_repo_id = args.hub_repo_id or "local/robocasa_composite_seen"
    slurm = args.slurm == 1

    logger.info(
        "Phase 1 (download) — %d tasks across %d workers (slurm=%s)",
        len(tasks),
        min(args.download_workers, len(tasks)),
        slurm,
    )
    download_executor = make_download_executor(
        task_names=tasks,
        overwrite=args.overwrite_download,
        job_name=args.download_job_name,
        logs_dir=args.logs_dir,
        workers=args.download_workers,
        partition=args.partition,
        cpus_per_task=args.download_cpus,
        mem=args.download_mem,
        time=args.download_time,
        slurm=slurm,
    )

    logger.info(
        "Phase 2 (aggregate) — single worker, output: %s (push_to_hub=%s)",
        output_repo_id,
        args.push_to_hub,
    )
    aggregate_executor = make_aggregate_executor(
        task_names=tasks,
        output_repo_id=output_repo_id,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        private=args.private,
        job_name=args.aggregate_job_name,
        logs_dir=args.logs_dir,
        partition=args.partition,
        cpus_per_task=args.aggregate_cpus,
        mem_per_cpu=args.aggregate_mem_per_cpu,
        time=args.aggregate_time,
        slurm=slurm,
        depends=download_executor if slurm else None,
    )

    if slurm:
        # Submitting the aggregate executor with ``depends=download_executor``
        # also submits the download executor — SlurmPipelineExecutor walks
        # the dependency chain and submits each job once with the right
        # ``--dependency=afterok:<jobid>`` arg.
        aggregate_executor.run()
    else:
        # Local sequential: run download to completion, then aggregate.
        download_executor.run()
        aggregate_executor.run()

    logger.info("Done. Merged dataset at %s.", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
