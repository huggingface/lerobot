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
Mirror a bimanual robot dataset using SLURM for distributed video processing.

This script creates a mirrored version of a dataset where:
1. Left and right arm observations/actions are swapped
2. Joint values are inverted according to a mirroring mask
3. Video frames are horizontally flipped (parallelized via SLURM)

Example usage:
```shell
# SLURM execution
python examples/port_datasets/slurm_mirror_dataset.py \
    --repo-id pepijn/openarm_bimanual \
    --output-repo-id pepijn/openarm_bimanual_mirrored \
    --logs-dir /fsx/user/logs \
    --partition hopper-cpu

# Local execution (for debugging)
python examples/port_datasets/slurm_mirror_dataset.py \
    --repo-id pepijn/openarm_bimanual \
    --output-repo-id pepijn/openarm_bimanual_mirrored \
    --slurm 0 \
    --push-to-hub
```
"""

import argparse
import logging
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)

OPENARM_MIRRORING_MASK = {
    "joint_1": -1,
    "joint_2": -1,
    "joint_3": -1,
    "joint_4": 1,
    "joint_5": -1,
    "joint_6": -1,
    "joint_7": -1,
    "gripper": 1,
}


class MirrorVideos(PipelineStep):
    """Pipeline step that mirrors video files for assigned episodes."""

    def __init__(
        self,
        repo_id: str,
        output_repo_id: str,
        root: str | None = None,
        output_root: str | None = None,
        vcodec: str = "libsvtav1",
    ):
        super().__init__()
        self.repo_id = repo_id
        self.output_repo_id = output_repo_id
        self.root = root
        self.output_root = output_root
        self.vcodec = vcodec

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import subprocess
        from pathlib import Path

        from datasets.utils.tqdm import disable_progress_bars

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import HF_LEROBOT_HOME
        from lerobot.utils.utils import init_logging

        init_logging()
        disable_progress_bars()

        def swap_left_right_name(name: str) -> str:
            result = name.replace("left_", "LEFT_PLACEHOLDER_")
            result = result.replace("right_", "left_")
            result = result.replace("LEFT_PLACEHOLDER_", "right_")
            return result

        def flip_video_frames(input_path: Path, output_path: Path, fps: float, vcodec: str):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-vf", "hflip",
                "-c:v", vcodec,
                "-g", "2",
                "-crf", "30",
                "-r", str(int(fps)),
                "-pix_fmt", "yuv420p",
                "-loglevel", "error",
            ]
            if vcodec == "libsvtav1":
                cmd.extend(["-preset", "12"])
            cmd.append(str(output_path))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        def video_is_valid(path: Path) -> bool:
            if not path.exists():
                return False
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
                    capture_output=True, text=True, timeout=30
                )
                return result.returncode == 0 and result.stdout.strip().isdigit()
            except Exception:
                return False

        root = Path(self.root) if self.root else None
        output_root = Path(self.output_root) if self.output_root else None

        dataset = LeRobotDataset(self.repo_id, root=root)
        output_root = output_root or (HF_LEROBOT_HOME / self.output_repo_id)

        if not dataset.meta.video_keys:
            logger.info(f"Rank {rank}: No videos to process")
            return

        video_tasks = []
        for old_video_key in dataset.meta.video_keys:
            new_video_key = swap_left_right_name(old_video_key)
            for ep_idx in range(dataset.meta.total_episodes):
                try:
                    src_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, old_video_key)
                    dst_relative = dataset.meta.get_video_file_path(ep_idx, old_video_key)
                    dst_relative_str = str(dst_relative).replace(old_video_key, new_video_key)
                    dst_path = output_root / dst_relative_str
                    if src_path.exists():
                        video_tasks.append((src_path, dst_path, ep_idx, old_video_key))
                except KeyError:
                    continue

        my_tasks = [t for i, t in enumerate(video_tasks) if i % world_size == rank]
        logger.info(f"Rank {rank}/{world_size}: Processing {len(my_tasks)}/{len(video_tasks)} videos")

        for src_path, dst_path, ep_idx, video_key in my_tasks:
            if video_is_valid(dst_path):
                logger.info(f"Rank {rank}: Skipping {dst_path.name} (already done)")
                continue
            logger.info(f"Rank {rank}: Processing {src_path.name} -> {dst_path.name}")
            flip_video_frames(src_path, dst_path, dataset.meta.fps, self.vcodec)


class MirrorDataAndMetadata(PipelineStep):
    """Pipeline step that mirrors parquet data and metadata (runs once on rank 0)."""

    def __init__(
        self,
        repo_id: str,
        output_repo_id: str,
        root: str | None = None,
        output_root: str | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.output_repo_id = output_repo_id
        self.root = root
        self.output_root = output_root

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        if rank != 0:
            return

        from pathlib import Path

        import numpy as np
        import pandas as pd
        from datasets.utils.tqdm import disable_progress_bars

        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.utils import DATA_DIR, DEFAULT_DATA_PATH, write_info, write_stats, write_tasks
        from lerobot.utils.constants import HF_LEROBOT_HOME
        from lerobot.utils.utils import init_logging

        init_logging()
        disable_progress_bars()

        MIRRORING_MASK = {
            "joint_1": -1, "joint_2": -1, "joint_3": -1, "joint_4": 1,
            "joint_5": -1, "joint_6": -1, "joint_7": -1, "gripper": 1,
        }

        def get_mirroring_mask(robot_type: str) -> dict[str, int]:
            if robot_type in ["bi_openarm_follower", "openarm_follower", "bi_openarms_follower", "openarms_follower"]:
                return MIRRORING_MASK
            raise ValueError(f"Unknown robot type: {robot_type}. Add a mirroring mask for this robot.")

        def swap_left_right_name(name: str) -> str:
            result = name.replace("left_", "LEFT_PLACEHOLDER_")
            result = result.replace("right_", "left_")
            result = result.replace("LEFT_PLACEHOLDER_", "right_")
            return result

        def mirror_feature_names(names: list[str]) -> tuple[list[str], dict[int, int]]:
            mirrored_names = [swap_left_right_name(n) for n in names]
            old_to_new_idx = {}
            for old_idx, old_name in enumerate(names):
                new_name = swap_left_right_name(old_name)
                new_idx = mirrored_names.index(new_name)
                old_to_new_idx[old_idx] = new_idx
            return mirrored_names, old_to_new_idx

        def apply_mirroring_mask(value: float, feature_name: str, mirroring_mask: dict[str, int]) -> float:
            name_without_prefix = feature_name.split("_", 1)[1] if "_" in feature_name else feature_name
            joint_name = name_without_prefix.split(".")[0]
            if joint_name in mirroring_mask:
                return value * mirroring_mask[joint_name]
            return value

        def mirror_array(array: np.ndarray, names: list[str], mirroring_mask: dict[str, int]) -> np.ndarray:
            mirrored_names, idx_mapping = mirror_feature_names(names)
            result = np.zeros_like(array)
            for old_idx, new_idx in idx_mapping.items():
                new_name = mirrored_names[new_idx]
                value = array[old_idx]
                mirrored_value = apply_mirroring_mask(value, new_name, mirroring_mask)
                result[new_idx] = mirrored_value
            return result

        def mirror_stats(stats: dict) -> dict:
            mirrored = {}
            for key, value in stats.items():
                new_key = swap_left_right_name(key)
                if isinstance(value, dict):
                    mirrored[new_key] = mirror_stats(value)
                else:
                    mirrored[new_key] = value
            return mirrored

        root = Path(self.root) if self.root else None
        output_root = Path(self.output_root) if self.output_root else None

        dataset = LeRobotDataset(self.repo_id, root=root)
        output_root = output_root or (HF_LEROBOT_HOME / self.output_repo_id)

        done_marker = output_root / ".data_mirrored"
        if done_marker.exists():
            logger.info("Data and metadata already mirrored, skipping")
            return

        robot_type = dataset.meta.robot_type or "bi_openarms_follower"
        mirroring_mask = get_mirroring_mask(robot_type)

        mirrored_features = {}
        for key, feat in dataset.meta.features.items():
            new_key = swap_left_right_name(key)
            new_feat = feat.copy()
            if "names" in new_feat and new_feat["names"]:
                new_feat["names"] = [swap_left_right_name(n) for n in new_feat["names"]]
            mirrored_features[new_key] = new_feat

        new_meta = LeRobotDatasetMetadata.create(
            repo_id=self.output_repo_id,
            fps=dataset.meta.fps,
            features=mirrored_features,
            robot_type=dataset.meta.robot_type,
            root=output_root,
            use_videos=len(dataset.meta.video_keys) > 0,
        )

        if dataset.meta.tasks is not None:
            write_tasks(dataset.meta.tasks, new_meta.root)

        data_dir = dataset.root / DATA_DIR
        parquet_files = sorted(data_dir.glob("*/*.parquet"))
        action_names = dataset.meta.features.get("action", {}).get("names", [])
        state_names = dataset.meta.features.get("observation.state", {}).get("names", [])

        for src_path in parquet_files:
            df = pd.read_parquet(src_path).reset_index(drop=True)
            relative_path = src_path.relative_to(dataset.root)
            chunk_dir = relative_path.parts[1]
            file_name = relative_path.parts[2]
            chunk_idx = int(chunk_dir.split("-")[1])
            file_idx = int(file_name.split("-")[1].split(".")[0])

            if "action" in df.columns and action_names:
                actions = np.stack(df["action"].values)
                mirrored_actions = np.array([mirror_array(row, action_names, mirroring_mask) for row in actions])
                df["action"] = list(mirrored_actions)

            if "observation.state" in df.columns and state_names:
                states = np.stack(df["observation.state"].values)
                mirrored_states = np.array([mirror_array(row, state_names, mirroring_mask) for row in states])
                df["observation.state"] = list(mirrored_states)

            dst_path = new_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst_path, index=False)

        episodes_dir = dataset.root / "meta/episodes"
        dst_episodes_dir = new_meta.root / "meta/episodes"
        if episodes_dir.exists():
            dst_episodes_dir.mkdir(parents=True, exist_ok=True)
            for src_parquet in episodes_dir.glob("*/*.parquet"):
                df = pd.read_parquet(src_parquet)
                columns_to_rename = {}
                for col in df.columns:
                    if col.startswith("videos/"):
                        parts = col.split("/")
                        if len(parts) >= 2:
                            video_key = parts[1]
                            new_video_key = swap_left_right_name(video_key)
                            new_col = col.replace(f"videos/{video_key}/", f"videos/{new_video_key}/")
                            columns_to_rename[col] = new_col
                if columns_to_rename:
                    df = df.rename(columns=columns_to_rename)
                dst_parquet = dst_episodes_dir / src_parquet.relative_to(episodes_dir)
                dst_parquet.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(dst_parquet, index=False)

        new_meta.info.update({
            "total_episodes": dataset.meta.total_episodes,
            "total_frames": dataset.meta.total_frames,
            "total_tasks": dataset.meta.total_tasks,
            "total_videos": dataset.meta.total_videos,
            "total_chunks": dataset.meta.total_chunks,
        })
        write_info(new_meta.info, new_meta.root)

        if dataset.meta.stats is not None:
            mirrored_stats = mirror_stats(dataset.meta.stats)
            write_stats(mirrored_stats, new_meta.root)

        done_marker.touch()
        logger.info(f"Data and metadata mirrored to {output_root}")


def swap_left_right_name(name: str) -> str:
    result = name.replace("left_", "LEFT_PLACEHOLDER_")
    result = result.replace("right_", "left_")
    result = result.replace("LEFT_PLACEHOLDER_", "right_")
    return result


def get_num_video_tasks(repo_id: str, root: str | None = None) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root_path = Path(root) if root else None
    dataset = LeRobotDataset(repo_id, root=root_path)
    count = 0
    for video_key in dataset.meta.video_keys:
        for ep_idx in range(dataset.meta.total_episodes):
            try:
                src_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, video_key)
                if src_path.exists():
                    count += 1
            except KeyError:
                continue
    return count


def make_mirror_executor(
    repo_id: str,
    output_repo_id: str,
    root: str | None,
    output_root: str | None,
    vcodec: str,
    job_name: str,
    logs_dir: Path,
    workers: int,
    partition: str,
    cpus_per_task: int,
    mem_per_cpu: str,
    time_limit: str,
    slurm: bool = True,
):
    num_tasks = get_num_video_tasks(repo_id, root) if slurm else 1
    num_tasks = max(1, num_tasks)

    kwargs = {
        "pipeline": [
            MirrorDataAndMetadata(repo_id, output_repo_id, root, output_root),
            MirrorVideos(repo_id, output_repo_id, root, output_root, vcodec),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update({
            "job_name": job_name,
            "tasks": num_tasks,
            "workers": min(workers, num_tasks),
            "time": time_limit,
            "partition": partition,
            "cpus_per_task": cpus_per_task,
            "sbatch_args": {
                "mem-per-cpu": mem_per_cpu,
                "requeue": True,
                "signal": "USR1@30",
            },
        })
        return SlurmPipelineExecutor(**kwargs)
    else:
        kwargs.update({"tasks": 1, "workers": 1})
        return LocalPipelineExecutor(**kwargs)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Mirror a bimanual robot dataset using SLURM")
    parser.add_argument("--repo-id", type=str, required=True, help="Source dataset repo_id")
    parser.add_argument("--output-repo-id", type=str, required=True, help="Output dataset repo_id")
    parser.add_argument("--root", type=str, default=None, help="Source dataset root directory")
    parser.add_argument("--output-root", type=str, default=None, help="Output dataset root directory")
    parser.add_argument("--vcodec", type=str, default="libsvtav1", help="Video codec")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"), help="Directory for datatrove logs")
    parser.add_argument("--job-name", type=str, default="mirror_dataset", help="SLURM job name")
    parser.add_argument("--slurm", type=int, default=1, help="Use SLURM (1) or local (0)")
    parser.add_argument("--workers", type=int, default=64, help="Number of SLURM workers")
    parser.add_argument("--partition", type=str, default="hopper-cpu", help="SLURM partition")
    parser.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per task")
    parser.add_argument("--mem-per-cpu", type=str, default="2G", help="Memory per CPU")
    parser.add_argument("--time-limit", type=str, default="04:00:00", help="SLURM time limit")
    parser.add_argument("--push-to-hub", action="store_true", help="Push mirrored dataset to HuggingFace Hub")

    args = parser.parse_args()

    executor = make_mirror_executor(
        repo_id=args.repo_id,
        output_repo_id=args.output_repo_id,
        root=args.root,
        output_root=args.output_root,
        vcodec=args.vcodec,
        job_name=args.job_name,
        logs_dir=args.logs_dir,
        workers=args.workers,
        partition=args.partition,
        cpus_per_task=args.cpus_per_task,
        mem_per_cpu=args.mem_per_cpu,
        time_limit=args.time_limit,
        slurm=args.slurm == 1,
    )
    executor.run()

    if args.push_to_hub:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import HF_LEROBOT_HOME
        output_root = Path(args.output_root) if args.output_root else HF_LEROBOT_HOME / args.output_repo_id
        logger.info(f"Pushing dataset to HuggingFace Hub: {args.output_repo_id}")
        dataset = LeRobotDataset(args.output_repo_id, root=output_root)
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
