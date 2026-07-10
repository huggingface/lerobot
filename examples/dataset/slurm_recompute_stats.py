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

This is a modified copy of lerobot's examples/dataset/slurm_recompute_stats.py
(feat/recompute-stats-readonly-and-visual branch) with three additions relevant
to a shared HPC cluster:

  1. --qos                : pass a SLURM QoS through to every worker's sbatch.
  2. per-worker hf-mount  : each worker mounts the read-only source dataset on
                            its OWN node's /scratch before loading it, injected
                            via datatrove's ``env_command`` hook. This keeps the
                            terabytes of reads node-local and lazy (nothing piles
                            up on /fsx) and keeps hub traffic on the CPU nodes.
  3. --chain-aggregate    : submit ``aggregate`` with an afterok dependency on
                            ``compute`` so it only runs once all shards exist
                            (no manual squeue-wait, no gap/overlap race).

IMPORTANT — how to run (do NOT sbatch this file):
  Run it as a normal python process on the LOGIN node. datatrove submits the
  workers for you. Because the reference copy (--new-root) walks the source tree
  on the login node, the source must also be mountable there — so mount once on
  the login node too, before launching (see the mount snippet below).

Requires: pip install 'lerobot[dataset]' datatrove

Example (single command, compute then dependent aggregate):

    # 0. Mount on the login node so the reference-copy walk can list the source.
    /fsx/$USER/bin/hf-mount-nfs-x86_64-linux \
        repo datasets/behavior-1k/2026-challenge-demos /scratch/$USER/behavior-demos \
        --cache-dir /scratch/$USER/hfmount-cache --cache-size 100000000000 &

    # 1. Launch. Each worker will mount the source on its own node via --mount-repo.
    python slurm_recompute_stats_patched.py compute \
        --repo-id behavior-1k/2026-challenge-demos \
        --new-root /fsx/$USER/behavior-1k_recomputed \
        --shard-dir /fsx/$USER/behavior-1k_recomputed/stats_shards \
        --logs-dir /fsx/$USER/logs/recompute \
        --skip-image-video 0 \
        --workers 250 \
        --partition hopper-cpu \
        --qos <your-cpu-qos> \
        --cpus-per-task 8 --mem-per-cpu 4G \
        --mount-repo datasets/behavior-1k/2026-challenge-demos \
        --hf-mount-bin /fsx/$USER/bin/hf-mount-nfs-x86_64-linux \
        --venv-path /fsx/$USER/venvs/lerobot/bin/activate \
        --chain-aggregate

REHEARSE FIRST with --workers 2 and inspect one worker's log under --logs-dir to
confirm the mount came up and video decoding ran (not a silent hub download).
"""

import argparse
import os
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


def _mem_gb(mem: str) -> int:
    """Parse '4G' / '4GB' / '4' into an int number of GB for datatrove's mem_per_cpu_gb."""
    s = str(mem).strip().lower().rstrip("b").rstrip("g")
    return int(float(s))


def _build_env_command(args) -> str | None:
    """Construct the per-worker shell snippet datatrove runs before the python step.

    Mounts the read-only source dataset on THIS worker's node-local /scratch, waits
    for it to come up, and fails LOUDLY (exit 1) if it doesn't — so a broken mount
    surfaces as a failed job instead of a silent fall-back to downloading the dataset.
    Also activates the venv. Returns None if --mount-repo was not requested (in which
    case you must supply --root yourself and datatrove uses --venv-path only).
    """
    if args.env_command:
        return args.env_command

    lines = []
    if args.venv_path:
        lines.append(f"source {args.venv_path}")

    if args.mount_repo:
        if not args.hf_mount_bin:
            raise SystemExit("--mount-repo requires --hf-mount-bin")
        mnt = args.mount_point
        cache = args.mount_cache_dir
        lines += [
            f'MNT="{mnt}"',
            f'CACHE="{cache}"',
            'mkdir -p "$MNT" "$CACHE"',
            f'{args.hf_mount_bin} \\',
            f'    repo {args.mount_repo} "$MNT" \\',
            f'    --cache-dir "$CACHE" --cache-size {args.mount_cache_size} &',
            'for i in $(seq 1 60); do [ -f "$MNT/meta/info.json" ] && break; sleep 2; done',
            '[ -f "$MNT/meta/info.json" ] || { echo "hf-mount failed to come up at $MNT" >&2; exit 1; }',
        ]

    return "\n".join(lines) if lines else None


def _make_executor(
    pipeline,
    logs_dir,
    job_name,
    slurm,
    workers,
    tasks,
    time,
    partition,
    cpus,
    mem,
    qos=None,
    env_command=None,
    depends=None,
):
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
                "mem_per_cpu_gb": _mem_gb(mem),  # datatrove's native field (int GB)
                "sbatch_args": {},
            }
        )
        if qos:
            kwargs["qos"] = qos  # -> "#SBATCH --qos=<qos>" on every worker
        if env_command:
            kwargs["env_command"] = env_command  # per-worker mount + venv, runs before python
        if depends is not None:
            kwargs["depends"] = depends  # chains --dependency=afterok:<compute jobid>
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


def _add_shared_args(p, user):
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
    p.add_argument("--partition", type=str, default=None, help="SLURM partition, e.g. 'hopper-cpu'.")
    p.add_argument("--qos", type=str, default=None, help="SLURM QoS, e.g. 'high'. Passed to every worker.")
    p.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per SLURM task.")
    p.add_argument("--mem-per-cpu", type=str, default="4G", help="Memory per CPU, e.g. '4G'.")
    p.add_argument(
        "--video-backend",
        type=str,
        default=None,
        help="Video decoding backend (e.g. 'pyav', 'torchcodec'). Defaults to the dataset's default; "
        "use 'pyav' if torchcodec fails to load locally.",
    )

    # --- per-worker mount options (patch) ---
    p.add_argument(
        "--env-command",
        type=str,
        default=None,
        help="Raw shell snippet injected into each worker's sbatch before the python step. "
        "Overrides the auto-generated mount snippet if given.",
    )
    p.add_argument(
        "--mount-repo",
        type=str,
        default=None,
        help="If set, each worker mounts this repo (e.g. 'datasets/user/name') on its own node "
        "via hf-mount before loading the dataset. Auto-sets --root to --mount-point if --root unset.",
    )
    p.add_argument("--hf-mount-bin", type=str, default=None, help="Path to the hf-mount NFS binary.")
    p.add_argument("--venv-path", type=str, default=None, help="Path to a venv activate script to source.")
    p.add_argument(
        "--mount-point",
        type=str,
        default=f"/scratch/{user}/behavior-demos",
        help="Node-local mount path (must be identical on every node).",
    )
    p.add_argument("--mount-cache-dir", type=str, default=f"/scratch/{user}/hfmount-cache")
    p.add_argument("--mount-cache-size", type=str, default="100000000000", help="hf-mount --cache-size bytes.")


def main():
    user = os.environ.get("USER", "user")

    parser = argparse.ArgumentParser(
        description="PATCHED SLURM-distributed LeRobotDataset stats recomputation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cp = sub.add_parser("compute", help="Distribute per-episode stats across SLURM workers.")
    _add_shared_args(cp, user)
    cp.add_argument("--workers", type=int, default=50, help="Number of parallel SLURM tasks.")
    cp.add_argument(
        "--skip-image-video",
        type=int,
        default=1,
        help="1 = numeric features only (fast); 0 = also recompute image/video stats (decodes frames).",
    )
    cp.add_argument(
        "--chain-aggregate",
        action="store_true",
        help="After building compute, submit aggregate with an afterok dependency (single command).",
    )
    cp.add_argument("--push-to-hub", action="store_true", help="For the chained aggregate: push after done.")

    ap = sub.add_parser("aggregate", help="Merge shards into meta/stats.json.")
    _add_shared_args(ap, user)
    ap.add_argument("--push-to-hub", action="store_true", help="Push the dataset after aggregation.")
    ap.add_argument(
        "--depends-job-id",
        type=str,
        default=None,
        help="Optional SLURM job id; aggregate waits for it (afterok) before running.",
    )

    args = parser.parse_args()
    slurm = args.slurm == 1

    # If a per-worker mount is requested and --root wasn't given, workers read from the mount.
    if args.mount_repo and not args.root:
        args.root = args.mount_point

    env_command = _build_env_command(args)

    if args.command == "compute":
        # The reference copy (if any) is created once on the submitting node so workers
        # can all load --new-root without racing to build it. NOTE: this walks the source
        # tree, so the source must be mountable on the login node too.
        _maybe_reference_copy(args.repo_id, args.root, args.new_root)

        compute_job_name = args.job_name or "recompute_stats_compute"
        compute_exec = _make_executor(
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
            job_name=compute_job_name,
            slurm=slurm,
            workers=args.workers,
            tasks=args.workers,
            time="24:00:00",
            partition=args.partition,
            cpus=args.cpus_per_task,
            mem=args.mem_per_cpu,
            qos=args.qos,
            env_command=env_command,
        )

        if args.chain_aggregate and slurm:
            # Build aggregate depending on compute. datatrove launches the dependency
            # (compute) first, then submits aggregate with --dependency=afterok:<jobid>.
            aggregate_exec = _make_executor(
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
                job_name="recompute_stats_aggregate",
                slurm=slurm,
                workers=1,
                tasks=1,
                time="02:00:00",
                partition=args.partition,
                cpus=args.cpus_per_task,
                mem=args.mem_per_cpu,
                qos=args.qos,
                env_command=env_command,  # aggregate also needs the mount to load the dataset
                depends=compute_exec,
            )
            aggregate_exec.run()
        else:
            compute_exec.run()
    else:
        aggregate_exec = _make_executor(
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
            job_name=args.job_name or "recompute_stats_aggregate",
            slurm=slurm,
            workers=1,
            tasks=1,
            time="02:00:00",
            partition=args.partition,
            cpus=args.cpus_per_task,
            mem=args.mem_per_cpu,
            qos=args.qos,
            env_command=env_command,
        )
        if args.depends_job_id is not None:
            aggregate_exec.depends_job_id = args.depends_job_id
        aggregate_exec.run()


if __name__ == "__main__":
    main()
