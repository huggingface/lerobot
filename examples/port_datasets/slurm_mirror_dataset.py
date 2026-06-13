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
Mirror a bimanual dataset in parallel with DataTrove + SLURM, then double it.

Workflow:
1) Split source episodes across `num_shards` ranks and mirror each shard in parallel.
2) Aggregate mirrored shards into one mirrored dataset.
3) Aggregate [original, mirrored] into a final doubled dataset.

Example:
python examples/port_datasets/slurm_mirror_dataset.py \
  --repo-id=pepijn/openarm_bimanual \
  --output-repo-id=pepijn/openarm_bimanual_doubled \
  --partition=hopper-cpu \
  --num-shards=256 \
  --workers=64 \
  --cpus-per-task=8 \
  --mem-per-cpu=4G
"""

import argparse
import copy
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging

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


def get_mirroring_mask(robot_type: str | None) -> dict[str, int]:
    if robot_type in ["bi_openarm_follower", "openarm_follower", "bi_openarms_follower", "openarms_follower"]:
        return OPENARM_MIRRORING_MASK
    raise ValueError(f"Unknown robot type: {robot_type}. Add a mirroring mask for this robot.")


def swap_left_right_name(name: str) -> str:
    value = name.replace("left_", "LEFT_PLACEHOLDER_")
    value = value.replace("right_", "left_")
    value = value.replace("LEFT_PLACEHOLDER_", "right_")
    return value


def mirror_feature_names(names: list[str]) -> tuple[list[str], dict[int, int]]:
    mirrored_names = [swap_left_right_name(n) for n in names]
    old_to_new_idx = {}
    for old_idx, old_name in enumerate(names):
        new_name = swap_left_right_name(old_name)
        new_idx = mirrored_names.index(new_name)
        old_to_new_idx[old_idx] = new_idx
    return mirrored_names, old_to_new_idx


def _get_axis_names(feature: dict[str, Any]) -> list[str] | None:
    names = feature.get("names")
    if isinstance(names, list):
        return names
    if isinstance(names, dict):
        axes = names.get("axes")
        if isinstance(axes, list):
            return axes
    return None


def _to_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def apply_mirroring_mask(value: float, axis_name: str, mirroring_mask: dict[str, int]) -> float:
    if axis_name.startswith("left_") or axis_name.startswith("right_"):
        axis_name = axis_name.split("_", 1)[1]
    joint_name = axis_name.split(".")[0]
    return value * mirroring_mask.get(joint_name, 1)


def mirror_vector_feature(
    value: Any,
    feature: dict[str, Any],
    mirroring_mask: dict[str, int],
) -> Any:
    array = _to_numpy(value)
    if not isinstance(array, np.ndarray) or array.ndim != 1:
        return array

    names = _get_axis_names(feature)
    if names is None or len(names) != len(array):
        return array

    mirrored_names, index_mapping = mirror_feature_names(names)
    mirrored = np.zeros_like(array)
    for old_idx, new_idx in index_mapping.items():
        mirrored[new_idx] = apply_mirroring_mask(array[old_idx], mirrored_names[new_idx], mirroring_mask)
    return mirrored


def flip_horizontal(value: Any, expected_shape: list[int] | tuple[int, ...]) -> Any:
    array = _to_numpy(value)
    if not isinstance(array, np.ndarray) or array.ndim != 3:
        return array

    expected_shape = tuple(expected_shape)
    if array.shape == expected_shape:
        return np.flip(array, axis=1).copy()  # HWC

    if len(expected_shape) == 3:
        c, h, w = expected_shape
        if array.shape == (c, h, w):
            return np.flip(array, axis=2).copy()  # CHW

    # Conservative fallback for unexpected layouts.
    return np.flip(array, axis=-1).copy()


def build_mirrored_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mirrored = {}
    for key, feature in features.items():
        new_key = swap_left_right_name(key)
        new_feature = copy.deepcopy(feature)
        names = new_feature.get("names")
        if isinstance(names, list):
            new_feature["names"] = [swap_left_right_name(name) for name in names]
        elif isinstance(names, dict) and isinstance(names.get("axes"), list):
            new_feature["names"]["axes"] = [swap_left_right_name(name) for name in names["axes"]]
        mirrored[new_key] = new_feature
    return mirrored


def build_mirrored_frame(
    item: dict[str, Any],
    source_features: dict[str, dict[str, Any]],
    mirroring_mask: dict[str, int],
) -> dict[str, Any]:
    frame = {}
    for key, feature in source_features.items():
        if key in DEFAULT_FEATURES:
            continue

        value = item[key]
        if key in {"action", "observation.state"}:
            value = mirror_vector_feature(value, feature, mirroring_mask)
        elif feature["dtype"] in {"video", "image"}:
            value = flip_horizontal(value, feature["shape"])
        else:
            value = _to_numpy(value)

        frame[swap_left_right_name(key)] = value

    frame["task"] = item["task"]
    if "timestamp" in item:
        ts = _to_numpy(item["timestamp"])
        frame["timestamp"] = float(ts.item() if hasattr(ts, "item") else ts)
    return frame


def _resolve_source_root(repo_id: str, root: Path | None) -> Path:
    source_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    return source_meta.root


def _get_work_dir(output_repo_id: str, work_dir: Path | None) -> Path:
    if work_dir is not None:
        return work_dir
    safe_name = output_repo_id.replace("/", "__")
    return HF_LEROBOT_HOME / "_mirror_work" / safe_name


def _get_shard_root(work_dir: Path, world_size: int, rank: int) -> Path:
    return work_dir / "mirrored_shards" / f"world_{world_size}_rank_{rank}"


def _is_valid_dataset_root(root: Path) -> bool:
    return (root / "meta" / "info.json").exists()


def mirror_shard(
    repo_id: str,
    source_root: Path,
    mirrored_repo_id: str,
    shard_root: Path,
    rank: int,
    world_size: int,
    vcodec: str,
    overwrite: bool,
) -> None:
    source_dataset = LeRobotDataset(repo_id=repo_id, root=source_root)
    selected_episodes = list(range(rank, source_dataset.meta.total_episodes, world_size))

    if len(selected_episodes) == 0:
        logger.info("Rank %s has no episodes assigned. Skipping.", rank)
        return

    if shard_root.exists():
        if overwrite:
            shutil.rmtree(shard_root)
        elif _is_valid_dataset_root(shard_root):
            logger.info("Rank %s shard already exists at %s. Skipping.", rank, shard_root)
            return
        else:
            raise RuntimeError(
                f"Shard root {shard_root} exists but is not a valid dataset. Use --overwrite to recreate."
            )

    mirroring_mask = get_mirroring_mask(source_dataset.meta.robot_type)
    mirrored_features = build_mirrored_features(source_dataset.meta.features)

    shard_repo_name = f"{mirrored_repo_id}_world_{world_size}_rank_{rank}"
    mirrored_dataset = LeRobotDataset.create(
        repo_id=shard_repo_name,
        root=shard_root,
        fps=source_dataset.meta.fps,
        features=mirrored_features,
        robot_type=source_dataset.meta.robot_type,
        use_videos=len(source_dataset.meta.video_keys) > 0,
        vcodec=vcodec,
    )
    mirrored_dataset.meta.update_chunk_settings(
        chunks_size=source_dataset.meta.chunks_size,
        data_files_size_in_mb=source_dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=source_dataset.meta.video_files_size_in_mb,
    )

    logger.info(
        "Rank %s processing %s episodes into shard %s",
        rank,
        len(selected_episodes),
        shard_root,
    )
    for source_ep_idx in selected_episodes:
        episode = source_dataset.meta.episodes[source_ep_idx]
        start_idx = int(episode["dataset_from_index"])
        end_idx = int(episode["dataset_to_index"])

        for frame_idx in range(start_idx, end_idx):
            item = source_dataset[frame_idx]
            mirrored_frame = build_mirrored_frame(
                item=item,
                source_features=source_dataset.meta.features,
                mirroring_mask=mirroring_mask,
            )
            mirrored_dataset.add_frame(mirrored_frame)

        mirrored_dataset.save_episode()

    mirrored_dataset.finalize()


class MirrorDatasetShards(PipelineStep):
    def __init__(
        self,
        repo_id: str,
        source_root: Path,
        mirrored_repo_id: str,
        work_dir: Path,
        vcodec: str,
        overwrite: bool,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.source_root = source_root
        self.mirrored_repo_id = mirrored_repo_id
        self.work_dir = work_dir
        self.vcodec = vcodec
        self.overwrite = overwrite

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        init_logging()
        shard_root = _get_shard_root(self.work_dir, world_size, rank)
        mirror_shard(
            repo_id=self.repo_id,
            source_root=self.source_root,
            mirrored_repo_id=self.mirrored_repo_id,
            shard_root=shard_root,
            rank=rank,
            world_size=world_size,
            vcodec=self.vcodec,
            overwrite=self.overwrite,
        )


def make_mirror_executor(
    repo_id: str,
    source_root: Path,
    mirrored_repo_id: str,
    work_dir: Path,
    logs_dir: Path,
    job_name: str,
    num_shards: int,
    workers: int,
    partition: str,
    cpus_per_task: int,
    mem_per_cpu: str,
    time_limit: str,
    vcodec: str,
    overwrite: bool,
    slurm: bool,
):
    kwargs = {
        "pipeline": [
            MirrorDatasetShards(
                repo_id=repo_id,
                source_root=source_root,
                mirrored_repo_id=mirrored_repo_id,
                work_dir=work_dir,
                vcodec=vcodec,
                overwrite=overwrite,
            ),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        if partition is None:
            raise ValueError("`--partition` is required when `--slurm 1`.")
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": num_shards,
                "workers": workers,
                "time": time_limit,
                "partition": partition,
                "cpus_per_task": cpus_per_task,
                "sbatch_args": {"mem-per-cpu": mem_per_cpu},
            }
        )
        return SlurmPipelineExecutor(**kwargs)

    kwargs.update({"tasks": num_shards, "workers": 1})
    return LocalPipelineExecutor(**kwargs)


class AggregateMirroredShardsStep(PipelineStep):
    def __init__(
        self,
        mirrored_repo_id: str,
        mirrored_root: Path,
        work_dir: Path,
        num_shards: int,
        overwrite: bool,
    ):
        super().__init__()
        self.mirrored_repo_id = mirrored_repo_id
        self.mirrored_root = mirrored_root
        self.work_dir = work_dir
        self.num_shards = num_shards
        self.overwrite = overwrite

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        init_logging()
        if rank != 0:
            logger.info("Skipping rank %s for aggregate mirrored step", rank)
            return
        aggregate_mirrored_shards(
            mirrored_repo_id=self.mirrored_repo_id,
            mirrored_root=self.mirrored_root,
            work_dir=self.work_dir,
            num_shards=self.num_shards,
            overwrite=self.overwrite,
        )


class BuildDoubledDatasetStep(PipelineStep):
    def __init__(
        self,
        source_repo_id: str,
        source_root: Path,
        mirrored_repo_id: str,
        mirrored_root: Path,
        output_repo_id: str,
        output_root: Path,
        overwrite: bool,
    ):
        super().__init__()
        self.source_repo_id = source_repo_id
        self.source_root = source_root
        self.mirrored_repo_id = mirrored_repo_id
        self.mirrored_root = mirrored_root
        self.output_repo_id = output_repo_id
        self.output_root = output_root
        self.overwrite = overwrite

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        init_logging()
        if rank != 0:
            logger.info("Skipping rank %s for build doubled step", rank)
            return
        build_doubled_dataset(
            source_repo_id=self.source_repo_id,
            source_root=self.source_root,
            mirrored_repo_id=self.mirrored_repo_id,
            mirrored_root=self.mirrored_root,
            output_repo_id=self.output_repo_id,
            output_root=self.output_root,
            overwrite=self.overwrite,
        )


class PushDoubledDatasetStep(PipelineStep):
    def __init__(
        self,
        output_repo_id: str,
        output_root: Path,
    ):
        super().__init__()
        self.output_repo_id = output_repo_id
        self.output_root = output_root

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        init_logging()
        if rank != 0:
            logger.info("Skipping rank %s for push step", rank)
            return
        logger.info("Pushing doubled dataset to hub: %s", self.output_repo_id)
        LeRobotDataset(self.output_repo_id, root=self.output_root).push_to_hub()


def make_single_task_executor(
    step: PipelineStep,
    logs_dir: Path,
    job_name: str,
    partition: str | None,
    cpus_per_task: int,
    mem_per_cpu: str,
    time_limit: str,
    slurm: bool,
    depends: SlurmPipelineExecutor | None = None,
):
    kwargs = {"pipeline": [step], "logging_dir": str(logs_dir / job_name)}
    if slurm:
        if partition is None:
            raise ValueError("`--partition` is required when `--slurm 1`.")
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": 1,
                "workers": 1,
                "time": time_limit,
                "partition": partition,
                "cpus_per_task": cpus_per_task,
                "sbatch_args": {"mem-per-cpu": mem_per_cpu},
                "depends": depends,
            }
        )
        return SlurmPipelineExecutor(**kwargs)

    kwargs.update({"tasks": 1, "workers": 1})
    return LocalPipelineExecutor(**kwargs)


def aggregate_mirrored_shards(
    mirrored_repo_id: str,
    mirrored_root: Path,
    work_dir: Path,
    num_shards: int,
    overwrite: bool,
):
    if mirrored_root.exists():
        if overwrite:
            shutil.rmtree(mirrored_root)
        elif _is_valid_dataset_root(mirrored_root):
            logger.info("Mirrored dataset already exists at %s. Skipping aggregation.", mirrored_root)
            return
        else:
            raise RuntimeError(
                f"Mirrored root {mirrored_root} exists but is not a valid dataset. Use --overwrite to recreate."
            )

    shard_repo_ids = []
    shard_roots = []
    for rank in range(num_shards):
        shard_root = _get_shard_root(work_dir, num_shards, rank)
        if _is_valid_dataset_root(shard_root):
            shard_repo_ids.append(f"{mirrored_repo_id}_world_{num_shards}_rank_{rank}")
            shard_roots.append(shard_root)

    if len(shard_repo_ids) == 0:
        raise RuntimeError("No mirrored shards were produced. Nothing to aggregate.")

    logger.info("Aggregating %s mirrored shards into %s", len(shard_repo_ids), mirrored_root)
    aggregate_datasets(
        repo_ids=shard_repo_ids,
        roots=shard_roots,
        aggr_repo_id=mirrored_repo_id,
        aggr_root=mirrored_root,
    )


def build_doubled_dataset(
    source_repo_id: str,
    source_root: Path,
    mirrored_repo_id: str,
    mirrored_root: Path,
    output_repo_id: str,
    output_root: Path,
    overwrite: bool,
):
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        elif _is_valid_dataset_root(output_root):
            logger.info("Doubled dataset already exists at %s. Skipping final aggregation.", output_root)
            return
        else:
            raise RuntimeError(
                f"Output root {output_root} exists but is not a valid dataset. Use --overwrite to recreate."
            )

    logger.info("Aggregating source + mirrored into doubled dataset at %s", output_root)
    aggregate_datasets(
        repo_ids=[source_repo_id, mirrored_repo_id],
        roots=[source_root, mirrored_root],
        aggr_repo_id=output_repo_id,
        aggr_root=output_root,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="Source dataset repo id.")
    parser.add_argument("--output-repo-id", type=str, required=True, help="Final doubled dataset repo id.")
    parser.add_argument("--root", type=Path, default=None, help="Root path of source dataset.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root path where final doubled dataset is written.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Intermediate directory for mirrored shards and mirrored aggregate dataset.",
    )
    parser.add_argument("--logs-dir", type=Path, required=True, help="DataTrove logs path.")
    parser.add_argument("--job-name", type=str, default="mirror_dataset", help="SLURM job name.")
    parser.add_argument("--num-shards", type=int, default=256, help="Number of DataTrove tasks/ranks.")
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Max concurrent DataTrove workers on SLURM.",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition (e.g. hopper-cpu).")
    parser.add_argument("--cpus-per-task", type=int, default=8, help="CPU count per SLURM task.")
    parser.add_argument("--mem-per-cpu", type=str, default="4G", help="Memory per CPU for SLURM task.")
    parser.add_argument("--time", type=str, default="24:00:00", help="SLURM time limit.")
    parser.add_argument("--vcodec", type=str, default="libsvtav1", help="Video codec for output videos.")
    parser.add_argument(
        "--slurm",
        type=int,
        default=1,
        help="Use SLURM executor. Set 0 for local sequential debugging.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete existing intermediate/final outputs.")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push final doubled dataset to Hugging Face Hub after completion.",
    )
    args = parser.parse_args()

    init_logging()
    slurm = args.slurm == 1

    source_root = _resolve_source_root(args.repo_id, args.root)
    output_root = args.output_root if args.output_root is not None else HF_LEROBOT_HOME / args.output_repo_id
    work_dir = _get_work_dir(args.output_repo_id, args.work_dir)
    mirrored_repo_id = f"{args.output_repo_id}_mirrored"
    mirrored_root = work_dir / "mirrored_aggregate"

    work_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    mirror_executor = make_mirror_executor(
        repo_id=args.repo_id,
        source_root=source_root,
        mirrored_repo_id=mirrored_repo_id,
        work_dir=work_dir,
        logs_dir=args.logs_dir,
        job_name=args.job_name,
        num_shards=args.num_shards,
        workers=args.workers,
        partition=args.partition,
        cpus_per_task=args.cpus_per_task,
        mem_per_cpu=args.mem_per_cpu,
        time_limit=args.time,
        vcodec=args.vcodec,
        overwrite=args.overwrite,
        slurm=slurm,
    )
    if slurm:
        aggregate_executor = make_single_task_executor(
            step=AggregateMirroredShardsStep(
                mirrored_repo_id=mirrored_repo_id,
                mirrored_root=mirrored_root,
                work_dir=work_dir,
                num_shards=args.num_shards,
                overwrite=args.overwrite,
            ),
            logs_dir=args.logs_dir,
            job_name=f"{args.job_name}_aggregate_mirrored",
            partition=args.partition,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu=args.mem_per_cpu,
            time_limit=args.time,
            slurm=True,
            depends=mirror_executor,
        )
        build_executor = make_single_task_executor(
            step=BuildDoubledDatasetStep(
                source_repo_id=args.repo_id,
                source_root=source_root,
                mirrored_repo_id=mirrored_repo_id,
                mirrored_root=mirrored_root,
                output_repo_id=args.output_repo_id,
                output_root=output_root,
                overwrite=args.overwrite,
            ),
            logs_dir=args.logs_dir,
            job_name=f"{args.job_name}_build_doubled",
            partition=args.partition,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu=args.mem_per_cpu,
            time_limit=args.time,
            slurm=True,
            depends=aggregate_executor,
        )

        final_executor: SlurmPipelineExecutor | LocalPipelineExecutor = build_executor
        push_executor = None
        if args.push_to_hub:
            push_executor = make_single_task_executor(
                step=PushDoubledDatasetStep(
                    output_repo_id=args.output_repo_id,
                    output_root=output_root,
                ),
                logs_dir=args.logs_dir,
                job_name=f"{args.job_name}_push",
                partition=args.partition,
                cpus_per_task=args.cpus_per_task,
                mem_per_cpu=args.mem_per_cpu,
                time_limit=args.time,
                slurm=True,
                depends=build_executor,
            )
            final_executor = push_executor

        final_executor.run()
        logger.info(
            "Submitted SLURM chain. job_ids: mirror=%s aggregate=%s doubled=%s push=%s",
            mirror_executor.job_id,
            aggregate_executor.job_id,
            build_executor.job_id,
            push_executor.job_id if push_executor is not None else None,
        )
        return

    mirror_executor.run()
    aggregate_mirrored_shards(
        mirrored_repo_id=mirrored_repo_id,
        mirrored_root=mirrored_root,
        work_dir=work_dir,
        num_shards=args.num_shards,
        overwrite=args.overwrite,
    )
    build_doubled_dataset(
        source_repo_id=args.repo_id,
        source_root=source_root,
        mirrored_repo_id=mirrored_repo_id,
        mirrored_root=mirrored_root,
        output_repo_id=args.output_repo_id,
        output_root=output_root,
        overwrite=args.overwrite,
    )
    if args.push_to_hub:
        logger.info("Pushing doubled dataset to hub: %s", args.output_repo_id)
        LeRobotDataset(args.output_repo_id, root=output_root).push_to_hub()


if __name__ == "__main__":
    main()
