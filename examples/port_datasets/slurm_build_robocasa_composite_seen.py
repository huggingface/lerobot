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

"""Rebuild RoboCasa tarballs into one unified LeRobot v3 dataset.

Discovers tasks from RoboCasa's ``box_links_ds.json`` for a given ``--split``
(``target`` or ``pretrain``) and ``--source`` (``human`` / ``mimicgen``), then
filters to a chosen ``--task-set`` (``composite_seen``, ``composite_unseen``,
``composite_all``, ``atomic``, ``composite_atomic``, ``all``) or an explicit
``--tasks`` list. Same code path produces the 16-task ``composite_seen`` slice,
the full 50-task target benchmark, the 300-task ``Human300`` pretraining
slice, or any 2-task smoke set.

Per-rank, each datatrove worker:

1. Downloads the assigned task tarball(s) directly from Box (resolved via the
   ``box_links_ds.json`` bundled with the local ``robocasa`` clone).
2. Converts the extracted LeRobot v2.1 dataset to v3.0 in place.
3. Rewrites the per-task data into a per-rank shard with:
   - the canonical RoboCasa task name in ``task``
   - standardized camera keys under ``observation.images.robot0_*``
   - a guaranteed flat ``observation.state`` (concatenation of base / EE /
     gripper sub-keys when the source dataset stores them separately)
   - a standardized ``action`` key

A single aggregate worker then merges all shards into one unified dataset.

Heavy lifting is parallelized via Datatrove + SLURM on CPU nodes. With
``--workers=16 --cpus-per-task=8`` on ``hopper-cpu`` you get 128 CPUs total
across the prepare phase (one task per worker, 8 CPUs each for ffmpeg /
parquet) and the aggregate phase reuses the same CPU budget on a single node.

Typical hopper-cpu invocation::

    uv run python examples/port_datasets/slurm_build_robocasa_composite_seen.py \\
        --repo-id=${HF_USER}/robocasa_composite_seen_v3 \\
        --work-dir=/fsx/${USER}/robocasa/datasets/v1.0 \\
        --robocasa-root=/fsx/${USER}/robocasa \\
        --split=target \\
        --source=human \\
        --partition=hopper-cpu \\
        --workers=16 \\
        --cpus-per-task=8 \\
        --mem-per-cpu=4G \\
        --time=24:00:00 \\
        --logs-dir=/fsx/${USER}/logs/robocasa

Local debug (no SLURM)::

    uv run python examples/port_datasets/slurm_build_robocasa_composite_seen.py \\
        --repo-id=local/robocasa_composite_seen_v3_smoke \\
        --work-dir=/tmp/robocasa_smoke \\
        --robocasa-root=$HOME/robocasa \\
        --slurm=0 --workers=1 \\
        --tasks PrepareCoffee

If ``robocasa`` is already importable in the runtime environment, you can omit
``--robocasa-root``; the box-links manifest will be located from the package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

DEFAULT_SPLIT = "target"
DEFAULT_SOURCE = "human"
DEFAULT_ROBOT_TYPE = "robocasa"

# RoboCasa365 target benchmark task groupings. Order matches the official docs.
COMPOSITE_SEEN_TASKS: list[str] = [
    "DeliverStraw",
    "GetToastedBread",
    "KettleBoiling",
    "LoadDishwasher",
    "PackIdenticalLunches",
    "PreSoakPan",
    "PrepareCoffee",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SetUpCuttingStation",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "WashLettuce",
]

COMPOSITE_UNSEEN_TASKS: list[str] = [
    "ArrangeBreadBasket",
    "ArrangeTea",
    "BreadSelection",
    "CategorizeCondiments",
    "CuttingToolSelection",
    "GarnishPancake",
    "GatherTableware",
    "HeatKebabSandwich",
    "MakeIceLemonade",
    "PanTransfer",
    "PortionHotDogs",
    "RecycleBottlesByType",
    "SeparateFreezerRack",
    "WaffleReheat",
    "WashFruitColander",
    "WeighIngredients",
]

ATOMIC_TASKS: list[str] = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]

TASK_SETS: dict[str, list[str]] = {
    "composite_seen": COMPOSITE_SEEN_TASKS,
    "composite_unseen": COMPOSITE_UNSEEN_TASKS,
    "composite_all": COMPOSITE_SEEN_TASKS + COMPOSITE_UNSEEN_TASKS,
    "atomic": ATOMIC_TASKS,
    "composite_atomic": COMPOSITE_SEEN_TASKS + COMPOSITE_UNSEEN_TASKS + ATOMIC_TASKS,
    "all": [],  # sentinel — no filter
}


def _task_name_from_tar_key(tar_key: str) -> str:
    parts = tar_key.split("/")
    if len(parts) < 3:
        raise ValueError(f"Unexpected RoboCasa tar key: {tar_key}")
    return parts[2].removesuffix(".tar")


def _resolve_box_links_json(
    box_links_json: Path | None,
    robocasa_root: Path | None,
) -> Path:
    if box_links_json is not None:
        if not box_links_json.exists():
            raise FileNotFoundError(f"--box-links-json does not exist: {box_links_json}")
        return box_links_json

    if robocasa_root is not None:
        candidates = [
            robocasa_root / "models" / "assets" / "box_links" / "box_links_ds.json",
            robocasa_root / "robocasa" / "models" / "assets" / "box_links" / "box_links_ds.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find box_links_ds.json under --robocasa-root={robocasa_root}"
        )

    try:
        import robocasa  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(
            "Could not resolve RoboCasa box links. Pass --robocasa-root or --box-links-json, "
            "or run in an environment where `robocasa` is importable."
        ) from exc

    candidate = Path(robocasa.__path__[0]) / "models" / "assets" / "box_links" / "box_links_ds.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Resolved RoboCasa package, but box links file is missing: {candidate}")
    return candidate


def _discover_tasks(
    box_links_json: Path,
    split: str = DEFAULT_SPLIT,
    source: str | None = DEFAULT_SOURCE,
) -> list[dict[str, str]]:
    with open(box_links_json) as f:
        box_links: dict[str, str] = json.load(f)

    tasks: list[dict[str, str]] = []
    for tar_key in sorted(box_links):
        parts = tar_key.split("/")
        if len(parts) < 3 or parts[0] != split:
            continue

        # RoboCasa registries can appear in two layouts:
        #   new: split/<atomic|composite>/<task>/<date>/lerobot.tar
        #   old: split/<human|mimicgen>/<task>.tar
        if parts[1] in {"human", "mimicgen"}:
            tar_source = parts[1]
        else:
            tar_source = "human"

        if source is not None and tar_source != source:
            continue

        tasks.append(
            {
                "task_name": _task_name_from_tar_key(tar_key),
                "tar_key": tar_key,
                "source": tar_source,
                "rel_path": tar_key.removesuffix(".tar"),
                "shared_url": box_links[tar_key],
            }
        )
    return tasks


class PrepareRoboCasaUnifiedShards(PipelineStep):
    """Build per-rank unified shards from RoboCasa task tarballs."""

    def __init__(
        self,
        tasks: list[dict[str, str]],
        output_repo_id: str,
        work_dir: str,
        split: str,
        robot_type: str,
        overwrite: bool = False,
        cleanup_temp: bool = False,
        max_episodes_per_task: int | None = None,
        vcodec: str = "libsvtav1",
    ):
        super().__init__()
        self.tasks = tasks
        self.output_repo_id = output_repo_id
        self.work_dir = Path(work_dir)
        self.split = split
        self.robot_type = robot_type
        self.overwrite = overwrite
        self.cleanup_temp = cleanup_temp
        self.max_episodes_per_task = max_episodes_per_task
        self.vcodec = vcodec

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import copy
        import json
        import logging
        import shutil
        import tarfile
        import urllib.request

        import numpy as np
        from PIL import Image

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import DEFAULT_DATA_FILE_SIZE_IN_MB, DEFAULT_VIDEO_FILE_SIZE_IN_MB
        from lerobot.scripts.convert_dataset_v21_to_v30 import (
            convert_data,
            convert_episodes_metadata,
            convert_info,
            convert_tasks,
            convert_videos,
            validate_local_dataset_version,
        )
        from lerobot.utils.utils import init_logging

        init_logging()

        target_image_keys = {
            "observation.images.robot0_agentview_left": [
                "observation.images.robot0_agentview_left",
                "left_image",
                "observation.images.left_image",
            ],
            "observation.images.robot0_agentview_right": [
                "observation.images.robot0_agentview_right",
                "right_image",
                "observation.images.right_image",
            ],
            "observation.images.robot0_eye_in_hand": [
                "observation.images.robot0_eye_in_hand",
                "wrist_image",
                "observation.images.wrist_image",
            ],
        }
        direct_state_keys = [
            "observation.state",
            "state",
        ]
        explicit_state_groups = [
            [
                "observation.state.base_position",
                "observation.state.base_rotation",
                "observation.state.end_effector_position_relative",
                "observation.state.end_effector_rotation_relative",
                "observation.state.gripper_qpos",
            ],
            [
                "state.base_position",
                "state.base_rotation",
                "state.end_effector_position_relative",
                "state.end_effector_rotation_relative",
                "state.gripper_qpos",
            ],
        ]

        my_tasks = self.tasks[rank::world_size]
        logging.info(
            "Rank %s/%s: rebuilding %s of %s tasks",
            rank,
            world_size,
            len(my_tasks),
            len(self.tasks),
        )
        if not my_tasks:
            return

        shard_repo_id = f"{self.output_repo_id}_world_{world_size}_rank_{rank}"
        shard_root = (
            self.work_dir
            / "shards"
            / self.output_repo_id.replace("/", "__")
            / f"world_{world_size}"
            / f"rank_{rank}"
        )

        def shard_is_complete(root: Path) -> bool:
            info_path = root / "meta" / "info.json"
            tasks_path = root / "meta" / "tasks.parquet"
            stats_path = root / "meta" / "stats.json"
            if not (info_path.exists() and tasks_path.exists() and stats_path.exists()):
                return False

            episodes_dir = root / "meta" / "episodes"
            data_dir = root / "data"
            videos_dir = root / "videos"
            if not episodes_dir.exists() or not data_dir.exists() or not videos_dir.exists():
                return False
            if not any(episodes_dir.rglob("*.parquet")):
                return False
            if not any(data_dir.rglob("*.parquet")):
                return False
            if not any(videos_dir.rglob("*.mp4")):
                return False

            with open(info_path) as f:
                info = json.load(f)
            return info.get("total_episodes", 0) > 0 and info.get("total_frames", 0) > 0

        if shard_is_complete(shard_root) and not self.overwrite:
            logging.info("Shard already complete, skipping rank %s: %s", rank, shard_root)
            return
        if shard_root.exists():
            if self.overwrite:
                logging.warning("Removing existing shard root (--overwrite): %s", shard_root)
            else:
                logging.warning("Removing incomplete shard root before rebuild: %s", shard_root)
            shutil.rmtree(shard_root)

        def direct_download_url(shared_url: str) -> str:
            shared_id = shared_url.rstrip("/").split("/")[-1]
            base = shared_url.split("/s/")[0]
            return f"{base}/shared/static/{shared_id}.tar"

        def restore_v21_root_if_needed(dataset_root: Path) -> None:
            old_root = dataset_root.parent / f"{dataset_root.name}_old"
            if not dataset_root.exists() and old_root.exists():
                shutil.move(str(old_root), str(dataset_root))

        def download_and_extract(shared_url: str, destination: Path) -> None:
            url = direct_download_url(shared_url)
            extract_dir = destination.parent
            extract_dir.mkdir(parents=True, exist_ok=True)
            tar_path = extract_dir / f"{destination.name}.tar"

            if destination.exists() and (destination / "meta" / "info.json").exists():
                logging.info("  Already extracted: %s", destination)
                return

            for attempt in range(3):
                try:
                    logging.info("  Downloading (attempt %s) -> %s", attempt + 1, tar_path)
                    urllib.request.urlretrieve(url, str(tar_path))
                    break
                except Exception as exc:
                    logging.warning("  Download attempt %s failed: %s", attempt + 1, exc)
                    if tar_path.exists():
                        tar_path.unlink()
            else:
                raise RuntimeError(f"Failed to download {url} after 3 attempts")

            logging.info("  Extracting to %s", extract_dir)
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=extract_dir)
            tar_path.unlink()

        def is_v30(dataset_root: Path) -> bool:
            info_path = dataset_root / "meta" / "info.json"
            if not info_path.exists():
                return False
            with open(info_path) as f:
                info = json.load(f)
            return info.get("codebase_version") == "v3.0"

        def convert_v21_to_v30(dataset_root: Path) -> None:
            data_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
            video_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB

            validate_local_dataset_version(dataset_root)

            new_root = dataset_root.parent / f"{dataset_root.name}_v30"
            if new_root.exists():
                shutil.rmtree(new_root)

            convert_info(dataset_root, new_root, data_mb, video_mb)
            convert_tasks(dataset_root, new_root)
            episodes_metadata = convert_data(dataset_root, new_root, data_mb)
            episodes_video_metadata = convert_videos(dataset_root, new_root, video_mb)
            convert_episodes_metadata(
                dataset_root,
                new_root,
                episodes_metadata,
                episodes_video_metadata,
            )

            old_root = dataset_root.parent / f"{dataset_root.name}_old"
            if old_root.exists():
                shutil.rmtree(old_root)
            shutil.move(str(dataset_root), str(old_root))
            shutil.move(str(new_root), str(dataset_root))
            logging.info("  Conversion complete: %s", dataset_root)

        def as_float32_vector(value) -> np.ndarray:
            if value.__class__.__module__.startswith("torch"):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)
            return arr.astype(np.float32).reshape(-1)

        def to_pil_image(value) -> Image.Image:
            if isinstance(value, Image.Image):
                return value
            if value.__class__.__module__.startswith("torch"):
                arr = value.detach().cpu()
                if arr.ndim != 3:
                    raise ValueError(f"Expected rank-3 image tensor, got shape {tuple(arr.shape)}")
                if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = arr.permute(1, 2, 0)
                if getattr(arr.dtype, "is_floating_point", False):
                    if float(arr.max()) <= 1.0:
                        arr = arr * 255.0
                    arr = arr.clamp(0, 255).byte()
                else:
                    arr = arr.byte()
                return Image.fromarray(arr.numpy())

            arr = np.asarray(value)
            if arr.ndim != 3:
                raise ValueError(f"Expected rank-3 image array, got shape {arr.shape}")
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if np.issubdtype(arr.dtype, np.floating):
                if float(arr.max()) <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr)

        def normalize_name(name: str) -> str:
            return name.replace("/", ".").replace("_", ".").lower()

        def choose_one(available_keys: list[str], aliases: list[str], label: str) -> str:
            for alias in aliases:
                if alias in available_keys:
                    return alias
            raise ValueError(f"Could not resolve {label}. Available keys: {available_keys}")

        def resolve_image_key_map(available_keys: list[str]) -> dict[str, str]:
            return {
                target_key: choose_one(available_keys, aliases, target_key)
                for target_key, aliases in target_image_keys.items()
            }

        def resolve_action_key(available_keys: list[str]) -> str:
            return choose_one(available_keys, ["action", "actions"], "action")

        def state_sort_key(name: str) -> tuple[int, str]:
            normalized = normalize_name(name)
            if "base.position" in normalized:
                return (0, normalized)
            if "base.rotation" in normalized or "base.quat" in normalized:
                return (1, normalized)
            if "end.effector.position" in normalized or "eef.pos" in normalized:
                return (2, normalized)
            if "end.effector.rotation" in normalized or "eef.quat" in normalized or "eef.rot" in normalized:
                return (3, normalized)
            if "gripper" in normalized:
                return (4, normalized)
            return (5, normalized)

        def resolve_state_keys(available_keys: list[str]) -> list[str]:
            for key in direct_state_keys:
                if key in available_keys:
                    return [key]

            for group in explicit_state_groups:
                if all(key in available_keys for key in group):
                    return group

            prefix_keys = [
                key
                for key in available_keys
                if key.startswith("observation.state.") or key.startswith("state.")
            ]
            if prefix_keys:
                return sorted(prefix_keys, key=state_sort_key)

            proprio_like = [
                key
                for key in available_keys
                if any(
                    token in normalize_name(key)
                    for token in ["base.position", "base.rotation", "end.effector", "eef", "gripper"]
                )
            ]
            if proprio_like:
                return sorted(set(proprio_like), key=state_sort_key)

            raise ValueError(f"Could not resolve RoboCasa proprioception keys. Available keys: {available_keys}")

        def build_state(item: dict, state_keys: list[str]) -> np.ndarray:
            if len(state_keys) == 1:
                return as_float32_vector(item[state_keys[0]])
            parts = [as_float32_vector(item[key]) for key in state_keys]
            return np.concatenate(parts, axis=0).astype(np.float32)

        def infer_target_features(
            src_dataset: LeRobotDataset,
            image_key_map: dict[str, str],
            action_key: str,
            state_dim: int,
        ) -> tuple[dict, bool]:
            features = {}
            use_videos = False

            for target_key, source_key in image_key_map.items():
                feature_info = copy.deepcopy(src_dataset.meta.features[source_key])
                if "fps" not in feature_info and feature_info.get("dtype") != "video":
                    feature_info["fps"] = int(src_dataset.meta.fps)
                use_videos = use_videos or feature_info.get("dtype") == "video"
                features[target_key] = feature_info

            action_info = copy.deepcopy(src_dataset.meta.features[action_key])
            action_info["dtype"] = "float32"
            action_info["fps"] = int(src_dataset.meta.fps)
            features["action"] = action_info

            features["observation.state"] = {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": [f"state_{i}" for i in range(state_dim)],
                "fps": int(src_dataset.meta.fps),
            }

            return features, use_videos

        def task_root(task_meta: dict[str, str]) -> Path:
            return self.work_dir / task_meta["rel_path"]

        def cleanup_task_root(dataset_root: Path) -> None:
            old_root = dataset_root.parent / f"{dataset_root.name}_old"
            if dataset_root.exists():
                shutil.rmtree(dataset_root)
            if old_root.exists():
                shutil.rmtree(old_root)

        shard_dataset = None
        shard_meta: dict[str, int | tuple[int, ...]] | None = None

        for task_meta in my_tasks:
            task_name = task_meta["task_name"]
            dataset_root = task_root(task_meta)

            logging.info("--- %s (%s) ---", task_name, task_meta["tar_key"])
            restore_v21_root_if_needed(dataset_root)

            download_and_extract(task_meta["shared_url"], dataset_root)
            if not is_v30(dataset_root):
                convert_v21_to_v30(dataset_root)

            src_dataset = LeRobotDataset(repo_id=task_name, root=dataset_root)
            available_keys = list(src_dataset.meta.features.keys())
            image_key_map = resolve_image_key_map(available_keys)
            action_key = resolve_action_key(available_keys)
            state_keys = resolve_state_keys(available_keys)

            if len(src_dataset) == 0:
                raise ValueError(f"Task dataset is empty: {dataset_root}")

            first_item = src_dataset[0]
            first_state = build_state(first_item, state_keys)
            first_action = as_float32_vector(first_item[action_key])

            if shard_dataset is None:
                target_features, use_videos = infer_target_features(
                    src_dataset=src_dataset,
                    image_key_map=image_key_map,
                    action_key=action_key,
                    state_dim=int(first_state.size),
                )
                shard_dataset = LeRobotDataset.create(
                    repo_id=shard_repo_id,
                    root=shard_root,
                    fps=int(src_dataset.meta.fps),
                    robot_type=self.robot_type,
                    features=target_features,
                    use_videos=use_videos,
                    vcodec=self.vcodec,
                    batch_encoding_size=1,
                )
                shard_meta = {
                    "fps": int(src_dataset.meta.fps),
                    "state_dim": int(first_state.size),
                    "action_shape": tuple(first_action.shape),
                }
            else:
                assert shard_meta is not None
                if int(src_dataset.meta.fps) != shard_meta["fps"]:
                    raise ValueError(
                        f"FPS mismatch for {task_name}: {src_dataset.meta.fps} != {shard_meta['fps']}"
                    )
                if int(first_state.size) != shard_meta["state_dim"]:
                    raise ValueError(
                        f"State dim mismatch for {task_name}: {first_state.size} != {shard_meta['state_dim']}"
                    )
                if tuple(first_action.shape) != shard_meta["action_shape"]:
                    raise ValueError(
                        f"Action shape mismatch for {task_name}: {tuple(first_action.shape)} != "
                        f"{shard_meta['action_shape']}"
                    )

            num_episodes = src_dataset.num_episodes
            if self.max_episodes_per_task is not None:
                num_episodes = min(num_episodes, self.max_episodes_per_task)

            logging.info("  Appending %s episodes into shard %s", num_episodes, shard_root)
            for episode_idx in range(num_episodes):
                start = int(src_dataset.meta.episodes["dataset_from_index"][episode_idx])
                end = int(src_dataset.meta.episodes["dataset_to_index"][episode_idx])

                for frame_idx in range(start, end):
                    item = src_dataset[frame_idx]
                    frame = {
                        "task": task_name,
                        "observation.state": build_state(item, state_keys),
                        "action": as_float32_vector(item[action_key]),
                    }
                    for target_key, source_key in image_key_map.items():
                        frame[target_key] = to_pil_image(item[source_key])
                    shard_dataset.add_frame(frame)

                shard_dataset.save_episode()

            if self.cleanup_temp:
                cleanup_task_root(dataset_root)

        if shard_dataset is None:
            logging.warning("Rank %s produced no shard dataset", rank)
            return

        shard_dataset.finalize()
        logging.info("Rank %s finalized shard at %s", rank, shard_root)


class AggregateRoboCasaUnifiedShards(PipelineStep):
    """Aggregate repaired shard datasets into one final RoboCasa dataset."""

    def __init__(
        self,
        output_repo_id: str,
        shard_roots: list[str],
        output_root: str,
        push: bool = True,
        overwrite: bool = False,
        hub_tags: list[str] | None = None,
    ):
        super().__init__()
        self.output_repo_id = output_repo_id
        self.shard_roots = [Path(root) for root in shard_roots]
        self.output_root = Path(output_root)
        self.push = push
        self.overwrite = overwrite
        self.hub_tags = hub_tags or ["lerobot", "robocasa", "unified"]

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import json
        import logging
        import shutil

        from lerobot.datasets.aggregate import aggregate_datasets
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.utils import init_logging

        init_logging()

        if rank != 0:
            logging.info("Rank %s: only rank 0 aggregates", rank)
            return

        def shard_is_complete(root: Path) -> bool:
            info_path = root / "meta" / "info.json"
            tasks_path = root / "meta" / "tasks.parquet"
            stats_path = root / "meta" / "stats.json"
            if not (info_path.exists() and tasks_path.exists() and stats_path.exists()):
                return False

            episodes_dir = root / "meta" / "episodes"
            data_dir = root / "data"
            videos_dir = root / "videos"
            if not episodes_dir.exists() or not data_dir.exists() or not videos_dir.exists():
                return False
            if not any(episodes_dir.rglob("*.parquet")):
                return False
            if not any(data_dir.rglob("*.parquet")):
                return False
            if not any(videos_dir.rglob("*.mp4")):
                return False

            with open(info_path) as f:
                info = json.load(f)
            return info.get("total_episodes", 0) > 0 and info.get("total_frames", 0) > 0

        missing = [root for root in self.shard_roots if not shard_is_complete(root)]
        if missing:
            raise FileNotFoundError(f"Missing shard datasets: {missing}")

        if self.output_root.exists() and self.overwrite:
            logging.warning("Removing existing unified output (--overwrite): %s", self.output_root)
            shutil.rmtree(self.output_root)

        shard_repo_ids = [f"{self.output_repo_id}_shard_{idx}" for idx in range(len(self.shard_roots))]
        logging.info("Aggregating %s shards into %s", len(self.shard_roots), self.output_root)
        aggregate_datasets(
            repo_ids=shard_repo_ids,
            roots=self.shard_roots,
            aggr_repo_id=self.output_repo_id,
            aggr_root=self.output_root,
        )

        if self.push:
            dataset = LeRobotDataset(repo_id=self.output_repo_id, root=self.output_root)
            dataset.push_to_hub(
                tags=self.hub_tags,
                private=False,
            )
            logging.info("Pushed to https://huggingface.co/datasets/%s", self.output_repo_id)


def make_prepare_executor(
    *,
    tasks: list[dict[str, str]],
    output_repo_id: str,
    work_dir: Path,
    split: str,
    robot_type: str,
    overwrite: bool,
    cleanup_temp: bool,
    max_episodes_per_task: int | None,
    vcodec: str,
    job_name: str,
    logs_dir: Path,
    workers: int,
    partition: str,
    cpus_per_task: int,
    mem_per_cpu: str,
    time_limit: str,
    slurm: bool,
):
    kwargs = {
        "pipeline": [
            PrepareRoboCasaUnifiedShards(
                tasks=tasks,
                output_repo_id=output_repo_id,
                work_dir=str(work_dir),
                split=split,
                robot_type=robot_type,
                overwrite=overwrite,
                cleanup_temp=cleanup_temp,
                max_episodes_per_task=max_episodes_per_task,
                vcodec=vcodec,
            )
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": workers,
                "workers": workers,
                "time": time_limit,
                "partition": partition,
                "cpus_per_task": cpus_per_task,
                "sbatch_args": {"mem-per-cpu": mem_per_cpu},
            }
        )
        return SlurmPipelineExecutor(**kwargs)

    kwargs.update({"tasks": workers, "workers": 1})
    return LocalPipelineExecutor(**kwargs)


def make_aggregate_executor(
    *,
    output_repo_id: str,
    shard_roots: list[Path],
    output_root: Path,
    push: bool,
    overwrite: bool,
    job_name: str,
    logs_dir: Path,
    partition: str,
    cpus_per_task: int,
    mem_per_cpu: str,
    time_limit: str,
    slurm: bool,
    hub_tags: list[str] | None = None,
    depends: SlurmPipelineExecutor | None = None,
):
    kwargs = {
        "pipeline": [
            AggregateRoboCasaUnifiedShards(
                output_repo_id=output_repo_id,
                shard_roots=[str(root) for root in shard_roots],
                output_root=str(output_root),
                push=push,
                overwrite=overwrite,
                hub_tags=hub_tags,
            )
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
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


def resolve_repo_id(args: argparse.Namespace) -> str:
    if args.repo_id:
        return args.repo_id
    if args.hf_user:
        return f"{args.hf_user}/robocasa_composite_seen_{args.split}_{args.source}_unified_v3"
    raise ValueError("Pass either --repo-id or --hf-user.")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild the 16 RoboCasa composite_seen tarballs into one unified LeRobot v3 dataset."
    )
    parser.add_argument("--repo-id", type=str, default=None, help="Final unified dataset repo id.")
    parser.add_argument(
        "--hf-user",
        type=str,
        default=None,
        help="Optional shorthand. If set and --repo-id is omitted, derive "
        "<hf_user>/robocasa_composite_seen_<split>_<source>_unified_v3.",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["target", "pretrain"])
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "prepare", "aggregate"],
        help="prepare = build shards, aggregate = merge existing shards, all = do both.",
    )
    parser.add_argument(
        "--task-set",
        type=str,
        default="composite_seen",
        choices=sorted(TASK_SETS.keys()),
        help="Predefined task set to restrict discovery to. Default "
        "``composite_seen`` (the 16 multi-step composite_seen tasks). Use "
        "``all`` to keep every discovered task in the split/source slice. "
        "``--tasks`` overrides this when provided.",
    )
    parser.add_argument("--robocasa-root", type=Path, default=None)
    parser.add_argument("--box-links-json", type=Path, default=None)
    parser.add_argument("--robot-type", type=str, default=DEFAULT_ROBOT_TYPE)
    parser.add_argument("--vcodec", type=str, default="libsvtav1")
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--cleanup-temp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--job-name", type=str, default="port_robocasa_composite_seen")
    parser.add_argument("--slurm", type=int, default=1, help="1 = Slurm executor, 0 = local debug.")
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of SLURM workers. Default 16 = one per composite_seen task.",
    )
    parser.add_argument("--partition", type=str, default="hopper-cpu")
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=8,
        help="CPUs per worker. 16 workers × 8 cpus = 128 cpus total on hopper-cpu.",
    )
    parser.add_argument("--mem-per-cpu", type=str, default="4G")
    parser.add_argument("--time", type=str, default="24:00:00")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Explicit task names. Overrides --task-set when provided.",
    )
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()

    box_links_json = _resolve_box_links_json(args.box_links_json, args.robocasa_root)
    all_tasks = _discover_tasks(box_links_json, split=args.split, source=args.source)

    # Filter: explicit --tasks wins; otherwise apply --task-set.
    if args.tasks:
        selected = {task.lower() for task in args.tasks}
        all_tasks = [task for task in all_tasks if task["task_name"].lower() in selected]
    elif args.task_set != "all":
        wanted = {t.lower() for t in TASK_SETS[args.task_set]}
        all_tasks = [task for task in all_tasks if task["task_name"].lower() in wanted]

    if not all_tasks:
        raise ValueError(
            f"No RoboCasa tasks selected for split={args.split!r}, source={args.source!r}, "
            f"task_set={args.task_set!r}, tasks={args.tasks!r}"
        )

    print(f"Tasks to rebuild ({len(all_tasks)}):")
    for task in all_tasks:
        print(f"  {task['task_name']}  ({task['tar_key']})")
    if args.dryrun:
        return

    output_repo_id = resolve_repo_id(args)
    output_root = args.work_dir / "unified" / output_repo_id
    active_ranks = [rank for rank in range(args.workers) if all_tasks[rank::args.workers]]
    shard_roots = [
        args.work_dir
        / "shards"
        / output_repo_id.replace("/", "__")
        / f"world_{args.workers}"
        / f"rank_{rank}"
        for rank in active_ranks
    ]

    prepare_executor = None
    if args.mode in {"all", "prepare"}:
        prepare_executor = make_prepare_executor(
            tasks=all_tasks,
            output_repo_id=output_repo_id,
            work_dir=args.work_dir,
            split=args.split,
            robot_type=args.robot_type,
            overwrite=args.overwrite,
            cleanup_temp=args.cleanup_temp,
            max_episodes_per_task=args.max_episodes_per_task,
            vcodec=args.vcodec,
            job_name=args.job_name,
            logs_dir=args.logs_dir,
            workers=args.workers,
            partition=args.partition,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu=args.mem_per_cpu,
            time_limit=args.time,
            slurm=args.slurm == 1,
        )
        if args.mode == "prepare":
            prepare_executor.run()

    if args.mode in {"all", "aggregate"}:
        hub_tags = ["lerobot", "robocasa", "unified", args.split, args.source]
        if not args.tasks and args.task_set != "all":
            hub_tags.append(args.task_set)
        aggregate_executor = make_aggregate_executor(
            output_repo_id=output_repo_id,
            shard_roots=shard_roots,
            output_root=output_root,
            push=not args.no_push,
            overwrite=args.overwrite,
            job_name=f"{args.job_name}_aggregate",
            logs_dir=args.logs_dir,
            partition=args.partition,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu=args.mem_per_cpu,
            time_limit=args.time,
            slurm=args.slurm == 1,
            hub_tags=hub_tags,
            depends=prepare_executor if args.mode == "all" and args.slurm == 1 else None,
        )
        if args.mode == "all" and args.slurm == 1:
            # SLURM: submitting the aggregate executor with depends=prepare_executor
            # transitively submits prepare too, with the right --dependency=afterok.
            aggregate_executor.run()
        elif args.mode == "all":
            # Local: run sequentially.
            assert prepare_executor is not None
            prepare_executor.run()
            aggregate_executor.run()
        else:
            aggregate_executor.run()


if __name__ == "__main__":
    main()
