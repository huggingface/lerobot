#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
SLURM-distributed recomputation of a LeRobotDataset's ``meta/stats.json``.

Per-episode statistics are embarrassingly parallel, so we shard episodes across
workers, each computing stats for its subset, then a single worker aggregates all
shards (weighted by frame counts) and writes ``meta/stats.json``. This is mostly
useful when recomputing image/video stats (``--skip-image-video 0``), which decodes
frames and is far more expensive than the numeric-only path.

Requires: pip install 'lerobot[dataset]' datatrove

Two subcommands, each a separate SLURM submission:

  compute    – N workers, each writes per-episode stats for its episode shard
  aggregate  – 1 worker, merges shards into meta/stats.json (optionally push to hub)

The dataset is read-only during ``compute``. When ``--new-root`` is given, a
lightweight reference copy is made (large files symlinked, only meta/ copied) so a
read-only / mounted source dataset is never modified; stats land in ``--new-root``.

Usage:
    # Recompute image/video stats for a mounted, read-only dataset with 50 workers.
    python slurm_recompute_stats.py compute \\
        --repo-id someone-else/their-dataset \\
        --root /path/to/mounted/repo \\
        --new-root /local/writable/their-dataset_recomputed \\
        --skip-image-video 0 --workers 50 --partition cpu

    python slurm_recompute_stats.py aggregate \\
        --repo-id someone-else/their-dataset \\
        --new-root /local/writable/their-dataset_recomputed \\
        --partition cpu

    # Run locally without SLURM (single process); use pyav if torchcodec won't load.
    python slurm_recompute_stats.py compute \\
        --repo-id someone-else/their-dataset \\
        --new-root /local/writable/their-dataset_recomputed \\
        --skip-image-video 0 --video-backend pyav --slurm 0
"""

import argparse
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

SHARD_PATTERN = "episode_stats_{rank:05d}.pkl"
SHARD_GLOB = "episode_stats_*.pkl"


def _load_dataset(repo_id: str, root: str | None, new_root: str | None, video_backend: str | None = None):
    """Load the (possibly reference-copied) dataset used for stats.

    When ``new_root`` differs from the source, create a read-only-safe reference copy
    once (only the aggregator's rank 0 or the first compute worker needs to; here every
    rank just loads ``new_root`` if it already exists, else falls back to the source).
    """
    from lerobot.datasets import LeRobotDataset

    kwargs = {"video_backend": video_backend} if video_backend else {}
    if new_root and Path(new_root).exists():
        return LeRobotDataset(repo_id, root=new_root, **kwargs)
    return LeRobotDataset(repo_id, root=root, **kwargs)


class ComputeEpisodeStatsShards(PipelineStep):
    """Each worker computes per-episode stats for its ``episodes[rank::world_size]`` shard."""

    def __init__(self, repo_id, root, new_root, skip_image_video, shard_dir, video_backend=None):
        super().__init__()
        self.repo_id = repo_id
        self.root = root
        self.new_root = new_root
        self.skip_image_video = skip_image_video
        self.shard_dir = shard_dir
        self.video_backend = video_backend

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import logging
        import pickle

        from lerobot.datasets import compute_dataset_episode_stats
        from lerobot.utils.utils import init_logging

        init_logging()
        dataset = _load_dataset(self.repo_id, self.root, self.new_root, self.video_backend)

        my_episodes = list(range(dataset.meta.total_episodes))[rank::world_size]
        if not my_episodes:
            logging.info(f"Rank {rank}: no episodes assigned")
            return
        logging.info(f"Rank {rank}: {len(my_episodes)} / {dataset.meta.total_episodes} episodes")

        episode_stats = compute_dataset_episode_stats(
            dataset,
            episode_indices=my_episodes,
            skip_image_video=self.skip_image_video,
        )

        shard_dir = Path(self.shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
        out = shard_dir / SHARD_PATTERN.format(rank=rank)
        with open(out, "wb") as f:
            pickle.dump(episode_stats, f)
        logging.info(f"Rank {rank}: saved {len(episode_stats)} episode stats to {out}")


class AggregateEpisodeStats(PipelineStep):
    """Merge all per-episode stat shards into meta/stats.json."""

    def __init__(self, repo_id, root, new_root, shard_dir, push_to_hub=False, video_backend=None):
        super().__init__()
        self.repo_id = repo_id
        self.root = root
        self.new_root = new_root
        self.shard_dir = shard_dir
        self.push_to_hub = push_to_hub
        self.video_backend = video_backend

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import logging
        import pickle

        from lerobot.datasets import aggregate_episode_stats
        from lerobot.utils.utils import init_logging

        init_logging()
        if rank != 0:
            return

        shard_dir = Path(self.shard_dir)
        shards = sorted(shard_dir.glob(SHARD_GLOB))
        if not shards:
            raise FileNotFoundError(f"No episode stat shards found in {shard_dir}")

        all_episode_stats = []
        for shard in shards:
            with open(shard, "rb") as f:
                all_episode_stats.extend(pickle.load(f))
        logging.info(f"Aggregating {len(all_episode_stats)} episode stats from {len(shards)} shards")

        dataset = _load_dataset(self.repo_id, self.root, self.new_root, self.video_backend)

        # Aggregation is order-independent, so the only way sharding changes the result is a
        # gap (dropped shard) or an overlap (episode counted twice). Verify the shards cover
        # every episode exactly once before writing stats.json.
        expected_episodes = dataset.meta.total_episodes
        if len(all_episode_stats) != expected_episodes:
            raise ValueError(
                f"Expected {expected_episodes} per-episode stats (one per episode) but got "
                f"{len(all_episode_stats)} across {len(shards)} shards. A compute shard is likely "
                "missing or was written more than once; re-run the failed shards before aggregating."
            )

        # Frame-count check catches the case where a duplicate and a gap cancel out in the
        # episode count: summed per-episode frame counts must equal the dataset's total frames.
        numeric_key = next(
            (
                k
                for k, v in dataset.meta.features.items()
                if v["dtype"] not in ("image", "video", "string") and all_episode_stats and k in all_episode_stats[0]
            ),
            None,
        )
        if numeric_key is not None:
            total_frames = sum(int(s[numeric_key]["count"][0]) for s in all_episode_stats)
            if total_frames != dataset.meta.total_frames:
                raise ValueError(
                    f"Summed frame count from shards ({total_frames}) != dataset total_frames "
                    f"({dataset.meta.total_frames}); episodes are double-counted or missing."
                )

        new_stats = aggregate_episode_stats(dataset, all_episode_stats)
        if new_stats is None:
            raise RuntimeError("Aggregation produced no stats")
        logging.info(f"Wrote stats for features: {list(new_stats.keys())} to {dataset.root}")

        if self.push_to_hub:
            logging.info(f"Pushing {self.repo_id} to hub")
            dataset.push_to_hub()


def _make_executor(pipeline, logs_dir, job_name, slurm, workers, tasks, time, partition, cpus, mem):
    kwargs = {"pipeline": pipeline, "logging_dir": str(Path(logs_dir) / job_name)}
    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": tasks,
                "workers": workers,
                "time": time,
                "partition": partition,
                "cpus_per_task": cpus,
                "sbatch_args": {"mem-per-cpu": mem},
            }
        )
        return SlurmPipelineExecutor(**kwargs)
    kwargs.update({"tasks": tasks, "workers": 1})
    return LocalPipelineExecutor(**kwargs)


def _maybe_reference_copy(repo_id, root, new_root):
    """Create the read-only-safe reference copy once, before submitting workers."""
    if not new_root:
        return
    from lerobot.datasets import LeRobotDataset
    from lerobot.scripts.lerobot_edit_dataset import _reference_copy_dataset

    new_root_path = Path(new_root)
    if new_root_path.exists():
        return
    src = LeRobotDataset(repo_id, root=root)
    _reference_copy_dataset(src.root, new_root_path)


def _add_shared_args(p):
    p.add_argument("--repo-id", type=str, required=True, help="Dataset identifier, e.g. 'user/dataset'.")
    p.add_argument("--root", type=str, default=None, help="Source dataset root (e.g. a mount).")
    p.add_argument(
        "--new-root",
        type=str,
        default=None,
        help="Writable output root; a read-only-safe reference copy of --root. If omitted, stats "
        "are written in place at --root.",
    )
    p.add_argument("--shard-dir", type=Path, default=Path("stats_shards"), help="Per-rank shard dir.")
    p.add_argument("--logs-dir", type=Path, default=Path("logs"), help="datatrove logs dir.")
    p.add_argument("--job-name", type=str, default=None, help="SLURM job name.")
    p.add_argument("--slurm", type=int, default=1, help="1 = submit via SLURM; 0 = run locally.")
    p.add_argument("--partition", type=str, default=None, help="SLURM partition.")
    p.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per SLURM task.")
    p.add_argument("--mem-per-cpu", type=str, default="4G", help="Memory per CPU, e.g. '4G'.")
    p.add_argument(
        "--video-backend",
        type=str,
        default=None,
        help="Video decoding backend (e.g. 'pyav', 'torchcodec'). Defaults to the dataset's default; "
        "use 'pyav' if torchcodec fails to load locally.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="SLURM-distributed LeRobotDataset stats recomputation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cp = sub.add_parser("compute", help="Distribute per-episode stats across SLURM workers.")
    _add_shared_args(cp)
    cp.add_argument("--workers", type=int, default=50, help="Number of parallel SLURM tasks.")
    cp.add_argument(
        "--skip-image-video",
        type=int,
        default=1,
        help="1 = numeric features only (fast); 0 = also recompute image/video stats (decodes frames).",
    )

    ap = sub.add_parser("aggregate", help="Merge shards into meta/stats.json.")
    _add_shared_args(ap)
    ap.add_argument("--push-to-hub", action="store_true", help="Push the dataset after aggregation.")

    args = parser.parse_args()
    slurm = args.slurm == 1

    if args.command == "compute":
        # The reference copy (if any) is created once on the submitting node so workers
        # can all load --new-root without racing to build it.
        _maybe_reference_copy(args.repo_id, args.root, args.new_root)
        job_name = args.job_name or "recompute_stats_compute"
        executor = _make_executor(
            pipeline=[
                ComputeEpisodeStatsShards(
                    args.repo_id,
                    args.root,
                    args.new_root,
                    bool(args.skip_image_video),
                    str(args.shard_dir),
                    args.video_backend,
                )
            ],
            logs_dir=args.logs_dir,
            job_name=job_name,
            slurm=slurm,
            workers=args.workers,
            tasks=args.workers,
            time="24:00:00",
            partition=args.partition,
            cpus=args.cpus_per_task,
            mem=args.mem_per_cpu,
        )
    else:
        job_name = args.job_name or "recompute_stats_aggregate"
        executor = _make_executor(
            pipeline=[
                AggregateEpisodeStats(
                    args.repo_id,
                    args.root,
                    args.new_root,
                    str(args.shard_dir),
                    args.push_to_hub,
                    args.video_backend,
                )
            ],
            logs_dir=args.logs_dir,
            job_name=job_name,
            slurm=slurm,
            workers=1,
            tasks=1,
            time="02:00:00",
            partition=args.partition,
            cpus=args.cpus_per_task,
            mem=args.mem_per_cpu,
        )

    executor.run()


if __name__ == "__main__":
    main()
