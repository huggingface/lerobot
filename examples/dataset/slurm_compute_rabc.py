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
SLURM-distributed SARM RA-BC annotation pipeline.

Computes SARM progress values for all frames in a dataset, distributed across
SLURM workers, then merges the shards into a single sarm_progress.parquet.

Two subcommands, each a separate SLURM submission:

  compute    – N workers, each computes progress for a subset of episodes
  aggregate  – 1 worker, merges N shards into sarm_progress.parquet, pushes to hub

Usage:
    python slurm_compute_rabc.py compute \\
        --repo-id user/dataset --reward-model-path user/sarm_model \\
        --stride 10 --device cpu --workers 50 --partition cpu

    python slurm_compute_rabc.py aggregate \\
        --repo-id user/dataset --reward-model-path user/sarm_model \\
        --partition cpu --push-to-hub
"""

import argparse
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep


class ComputeProgressShards(PipelineStep):
    """Each worker computes SARM progress for its assigned episodes."""

    def __init__(
        self, repo_id, reward_model_path, stride=1, head_mode="sparse", device="cpu", shard_dir="rabc_shards"
    ):
        super().__init__()
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.repo_id = repo_id
        self.reward_model_path = reward_model_path
        self.stride = stride
        self.head_mode = head_mode
        self.device = device
        self.shard_dir = shard_dir

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import logging
        from pathlib import Path

        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        import torch
        from tqdm import tqdm

        from lerobot.policies.sarm.compute_rabc_weights import (
            generate_all_frame_indices,
            interpolate_progress,
            load_sarm_resources,
        )
        from lerobot.utils.utils import init_logging

        init_logging()

        dataset, reward_model, preprocess = load_sarm_resources(
            self.repo_id,
            self.reward_model_path,
            self.device,
        )

        if hasattr(preprocess, "eval"):
            preprocess.eval()
        for step in preprocess.steps:
            if hasattr(step, "eval"):
                step.eval()

        image_key = reward_model.config.image_key
        state_key = reward_model.config.state_key
        frame_gap = reward_model.config.frame_gap
        center_idx = reward_model.config.n_obs_steps // 2

        dual_mode = reward_model.config.uses_dual_heads
        compute_sparse = self.head_mode in ("sparse", "both") or not dual_mode
        compute_dense = self.head_mode in ("dense", "both") and dual_mode

        my_episodes = list(range(dataset.num_episodes))[rank::world_size]
        if not my_episodes:
            logging.info(f"Rank {rank}: no episodes assigned")
            return
        logging.info(f"Rank {rank}: {len(my_episodes)} / {dataset.num_episodes} episodes")

        all_rows = []

        for ep_idx in tqdm(my_episodes, desc=f"Rank {rank}"):
            ep = dataset.meta.episodes[ep_idx]
            ep_start, ep_end = ep["dataset_from_index"], ep["dataset_to_index"]
            task = dataset[ep_start].get("task", "perform the task")

            all_ep_indices = generate_all_frame_indices(ep_start, ep_end, frame_gap)
            if self.stride > 1:
                compute_indices = [i for i in all_ep_indices if (i - ep_start) % self.stride == 0]
                if (ep_end - 1) not in compute_indices:
                    compute_indices.append(ep_end - 1)
                compute_indices = sorted(set(compute_indices))
            else:
                compute_indices = all_ep_indices

            frame_results = {}
            for qi in tqdm(compute_indices, desc=f"  Ep {ep_idx}", leave=False):
                try:
                    sample = dataset[qi]
                    batch = {
                        image_key: sample[image_key],
                        "task": task,
                        "index": qi,
                        "episode_index": ep_idx,
                    }
                    if state_key in sample:
                        batch[state_key] = sample[state_key]

                    with torch.no_grad():
                        processed = preprocess(batch)
                        vf = processed["video_features"].to(self.device)
                        tf = processed["text_features"].to(self.device)
                        sf = processed.get("state_features")
                        if sf is not None:
                            sf = sf.to(self.device)
                        lengths = processed.get("lengths")

                        sparse_val = dense_val = np.nan
                        if compute_sparse:
                            r = reward_model.calculate_rewards(
                                text_embeddings=tf,
                                video_embeddings=vf,
                                state_features=sf,
                                lengths=lengths,
                                return_all_frames=True,
                                head_mode="sparse",
                            )
                            sparse_val = float(r[0, center_idx] if r.ndim == 2 else r[center_idx])
                        if compute_dense:
                            r = reward_model.calculate_rewards(
                                text_embeddings=tf,
                                video_embeddings=vf,
                                state_features=sf,
                                lengths=lengths,
                                return_all_frames=True,
                                head_mode="dense",
                            )
                            dense_val = float(r[0, center_idx] if r.ndim == 2 else r[center_idx])

                        frame_results[qi] = (sparse_val, dense_val)
                except Exception as e:
                    logging.warning(f"Failed frame {qi}: {e}")

            if not frame_results:
                logging.warning(f"Episode {ep_idx}: all frames failed, skipping")
                continue

            # Interpolate to all frames in this episode
            computed_idx = np.array(sorted(frame_results.keys()))
            all_frame_arr = np.arange(ep_start, ep_end)

            sparse_vals = np.array([frame_results[i][0] for i in computed_idx]) if compute_sparse else None
            dense_vals = np.array([frame_results[i][1] for i in computed_idx]) if compute_dense else None

            if self.stride > 1 and len(computed_idx) > 1:
                if compute_sparse:
                    sparse_vals = interpolate_progress(computed_idx, sparse_vals, all_frame_arr)
                if compute_dense:
                    dense_vals = interpolate_progress(computed_idx, dense_vals, all_frame_arr)
                output_frames = all_frame_arr
            else:
                # Use only successfully computed frames to avoid indexing mismatch on failures
                output_frames = computed_idx

            for i, fi in enumerate(output_frames):
                row = {"index": int(fi), "episode_index": ep_idx, "frame_index": int(fi - ep_start)}
                if compute_sparse:
                    row["progress_sparse"] = float(sparse_vals[i])
                if compute_dense:
                    row["progress_dense"] = float(dense_vals[i])
                all_rows.append(row)

        if all_rows:
            import pandas as pd

            df = pd.DataFrame(all_rows).sort_values("index").reset_index(drop=True)
            table = pa.Table.from_pandas(df, preserve_index=False)
            table = table.replace_schema_metadata({b"reward_model_path": self.reward_model_path.encode()})
            shard_dir = Path(self.shard_dir)
            shard_dir.mkdir(parents=True, exist_ok=True)
            out = shard_dir / f"shard_{rank:05d}.parquet"
            pq.write_table(table, out)
            logging.info(f"Rank {rank}: saved {len(df)} rows to {out}")


class AggregateProgress(PipelineStep):
    """Merge all shard parquets into final sarm_progress.parquet."""

    def __init__(self, repo_id, reward_model_path, shard_dir="rabc_shards", push_to_hub=False):
        super().__init__()
        self.repo_id = repo_id
        self.reward_model_path = reward_model_path
        self.shard_dir = shard_dir
        self.push_to_hub = push_to_hub

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import datetime
        import logging
        import os
        from pathlib import Path

        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        from lerobot.datasets import LeRobotDataset
        from lerobot.utils.utils import init_logging

        init_logging()
        if rank != 0:
            return

        shard_dir = Path(self.shard_dir)
        shards = sorted(shard_dir.glob("shard_*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No shards found in {shard_dir}")

        # Log shard modification time range to help detect stale files
        mtimes = [os.path.getmtime(s) for s in shards]
        oldest = datetime.datetime.fromtimestamp(min(mtimes)).isoformat(timespec="seconds")
        newest = datetime.datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds")
        logging.info(f"Aggregating {len(shards)} shards (oldest: {oldest}, newest: {newest})")

        df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
        df = df.sort_values("index").reset_index(drop=True)

        table = pa.Table.from_pandas(df, preserve_index=False)
        table = table.replace_schema_metadata({b"reward_model_path": self.reward_model_path.encode()})

        temp_ds = LeRobotDataset(self.repo_id, download_videos=False)
        out_path = Path(temp_ds.root) / "sarm_progress.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)
        logging.info(f"Saved {len(df)} rows to {out_path}")

        for col in ["progress_sparse", "progress_dense"]:
            if col in df.columns:
                v = df[col].dropna()
                logging.info(
                    f"{col}: mean={v.mean():.4f} std={v.std():.4f} min={v.min():.4f} max={v.max():.4f}"
                )

        if self.push_to_hub:
            from huggingface_hub import HfApi

            api = HfApi()
            hub_path = "sarm_progress.parquet"
            logging.info(f"Uploading to {self.repo_id}/{hub_path}")
            api.upload_file(
                path_or_fileobj=str(out_path),
                path_in_repo=hub_path,
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            logging.info(f"Uploaded: https://huggingface.co/datasets/{self.repo_id}/blob/main/{hub_path}")


def make_compute_executor(
    repo_id,
    reward_model_path,
    stride,
    head_mode,
    device,
    shard_dir,
    logs_dir,
    job_name,
    slurm,
    workers,
    partition,
    cpus_per_task,
    mem_per_cpu,
):
    kwargs = {
        "pipeline": [
            ComputeProgressShards(repo_id, reward_model_path, stride, head_mode, device, str(shard_dir)),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": workers,
                "workers": workers,
                "time": "24:00:00",
                "partition": partition,
                "cpus_per_task": cpus_per_task,
                "sbatch_args": {"mem-per-cpu": mem_per_cpu},
            }
        )
        return SlurmPipelineExecutor(**kwargs)

    kwargs.update({"tasks": workers, "workers": 1})
    return LocalPipelineExecutor(**kwargs)


def make_aggregate_executor(
    repo_id,
    reward_model_path,
    shard_dir,
    logs_dir,
    job_name,
    slurm,
    partition,
    cpus_per_task,
    mem_per_cpu,
    push_to_hub,
):
    kwargs = {
        "pipeline": [
            AggregateProgress(repo_id, reward_model_path, str(shard_dir), push_to_hub),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": 1,
                "workers": 1,
                "time": "02:00:00",
                "partition": partition,
                "cpus_per_task": cpus_per_task,
                "sbatch_args": {"mem-per-cpu": mem_per_cpu},
            }
        )
        return SlurmPipelineExecutor(**kwargs)

    kwargs.update({"tasks": 1, "workers": 1})
    return LocalPipelineExecutor(**kwargs)


def _add_shared_args(p):
    p.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository identifier, e.g. 'user/dataset'.",
    )
    p.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("rabc_shards"),
        help="Directory to read/write per-rank parquet shards.",
    )
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for datatrove logs.",
    )
    p.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="SLURM job name (defaults to rabc_<subcommand>).",
    )
    p.add_argument(
        "--slurm",
        type=int,
        default=1,
        help="1 = submit via SLURM; 0 = run locally (useful for debugging).",
    )
    p.add_argument(
        "--partition",
        type=str,
        default=None,
        help="SLURM partition to submit to.",
    )
    p.add_argument(
        "--cpus-per-task",
        type=int,
        default=4,
        help="Number of CPUs per SLURM task.",
    )
    p.add_argument(
        "--mem-per-cpu",
        type=str,
        default="4G",
        help="Memory per CPU, e.g. '4G' or '1950M'.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="SLURM-distributed SARM RA-BC annotation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compute subcommand
    cp = sub.add_parser(
        "compute",
        help="Distribute progress computation across SLURM workers.",
    )
    _add_shared_args(cp)
    cp.add_argument(
        "--reward-model-path",
        type=str,
        required=True,
        help="Path or HF repo id of the SARM reward model.",
    )
    cp.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Compute every Nth frame; intermediate frames are interpolated (must be >= 1).",
    )
    cp.add_argument(
        "--head-mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "both"],
        help="Which reward head(s) to compute.",
    )
    cp.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for reward model inference, e.g. 'cpu' or 'cuda'.",
    )
    cp.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Number of parallel SLURM tasks (one shard per worker).",
    )

    # aggregate subcommand
    ap = sub.add_parser(
        "aggregate",
        help="Merge per-rank shards into a single sarm_progress.parquet.",
    )
    _add_shared_args(ap)
    ap.add_argument(
        "--reward-model-path",
        type=str,
        required=True,
        help="Path or HF repo id of the SARM reward model (stored in parquet metadata).",
    )
    ap.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload sarm_progress.parquet to the Hugging Face Hub after aggregation.",
    )

    args = parser.parse_args()
    job_name = args.job_name or f"rabc_{args.command}"
    kwargs = vars(args)
    kwargs["slurm"] = kwargs.pop("slurm") == 1
    kwargs["job_name"] = job_name
    command = kwargs.pop("command")

    executor = make_compute_executor(**kwargs) if command == "compute" else make_aggregate_executor(**kwargs)

    executor.run()


if __name__ == "__main__":
    main()
