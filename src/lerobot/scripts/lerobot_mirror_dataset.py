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
Mirror a bimanual robot dataset by swapping left/right arms and inverting joint values.

This script creates a mirrored version of a dataset where:
1. Left and right arm observations/actions are swapped
2. Joint values are inverted according to a mirroring mask
3. Video frames are horizontally flipped

Example usage:
```shell
python -m lerobot.scripts.lerobot_mirror_dataset \
    --repo_id=pepijn/openarm_bimanual \
    --output_repo_id=pepijn/openarm_bimanual_mirrored
```
"""

import argparse
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_DATA_PATH,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)

OPENARM_MIRRORING_MASK = {
    "joint_1": -1,  # Pan - invert
    "joint_2": -1,  # Lift - invert
    "joint_3": -1,  # Roll - invert
    "joint_4": 1,   # Elbow - no invert
    "joint_5": -1,  # W-Roll - invert
    "joint_6": -1,  # W-Pitch - invert
    "joint_7": -1,  # W-Yaw - invert
    "gripper": 1,   # Gripper - no invert
}


def get_mirroring_mask(robot_type: str) -> dict[str, int]:
    """Get the mirroring mask for a given robot type."""
    if robot_type in ["bi_openarm_follower", "openarm_follower", "bi_openarms_follower", "openarms_follower"]:
        return OPENARM_MIRRORING_MASK
    raise ValueError(f"Unknown robot type: {robot_type}. Add a mirroring mask for this robot.")


def swap_left_right_name(name: str) -> str:
    """Swap 'left' and 'right' in a feature name."""
    if name.startswith("left_"):
        return "right_" + name[5:]
    elif name.startswith("right_"):
        return "left_" + name[6:]
    return name


def mirror_feature_names(names: list[str]) -> tuple[list[str], dict[int, int]]:
    """Mirror feature names by swapping left/right and return the new names and index mapping."""
    mirrored_names = [swap_left_right_name(n) for n in names]
    old_to_new_idx = {}
    for old_idx, old_name in enumerate(names):
        new_name = swap_left_right_name(old_name)
        new_idx = mirrored_names.index(new_name)
        old_to_new_idx[old_idx] = new_idx
    return mirrored_names, old_to_new_idx


def apply_mirroring_mask(
    value: float,
    feature_name: str,
    mirroring_mask: dict[str, int],
) -> float:
    """Apply mirroring mask to a joint value."""
    name_without_prefix = feature_name.split("_", 1)[1] if "_" in feature_name else feature_name
    joint_name = name_without_prefix.split(".")[0]
    if joint_name in mirroring_mask:
        return value * mirroring_mask[joint_name]
    return value


def mirror_array(
    array: np.ndarray,
    names: list[str],
    mirroring_mask: dict[str, int],
) -> np.ndarray:
    """Mirror an array of values (action or state) by swapping left/right and applying mask."""
    mirrored_names, idx_mapping = mirror_feature_names(names)
    result = np.zeros_like(array)
    for old_idx, new_idx in idx_mapping.items():
        old_name = names[old_idx]
        new_name = mirrored_names[new_idx]
        value = array[old_idx]
        mirrored_value = apply_mirroring_mask(value, new_name, mirroring_mask)
        result[new_idx] = mirrored_value
    return result


def flip_video_frames(
    input_path: Path,
    output_path: Path,
    fps: float,
    vcodec: str = "libsvtav1",
):
    """Flip video frames horizontally using FFmpeg with same settings as encode_video_frames."""
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


def mirror_dataset(
    repo_id: str,
    output_repo_id: str,
    root: str | Path | None = None,
    output_root: str | Path | None = None,
    mirroring_mask: dict[str, int] | None = None,
    vcodec: str = "libsvtav1",
    num_workers: int | None = None,
) -> LeRobotDataset:
    """Mirror a bimanual robot dataset."""
    logger.info(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id, root=root)

    if mirroring_mask is None:
        robot_type = dataset.meta.robot_type or "bi_openarms_follower"
        mirroring_mask = get_mirroring_mask(robot_type)
        logger.info(f"Using mirroring mask for robot type: {robot_type}")

    output_root = Path(output_root) if output_root else HF_LEROBOT_HOME / output_repo_id

    mirrored_features = {}
    for key, feat in dataset.meta.features.items():
        new_key = swap_left_right_name(key)
        new_feat = feat.copy()
        if "names" in new_feat and new_feat["names"]:
            new_feat["names"] = [swap_left_right_name(n) for n in new_feat["names"]]
        mirrored_features[new_key] = new_feat

    logger.info("Creating mirrored dataset metadata...")
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=output_repo_id,
        fps=dataset.meta.fps,
        features=mirrored_features,
        robot_type=dataset.meta.robot_type,
        root=output_root,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    if dataset.meta.tasks is not None:
        write_tasks(dataset.meta.tasks, new_meta.root)
        new_meta.tasks = dataset.meta.tasks.copy()

    _mirror_data(dataset, new_meta, mirroring_mask)
    _mirror_videos(dataset, new_meta, vcodec, num_workers)
    _copy_episodes_metadata(dataset, new_meta)

    logger.info(f"Mirrored dataset saved to: {output_root}")
    return LeRobotDataset(output_repo_id, root=output_root)


def _mirror_data(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    mirroring_mask: dict[str, int],
) -> None:
    """Mirror parquet data files."""
    data_dir = src_dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    action_names = src_dataset.meta.features.get("action", {}).get("names", [])
    state_names = src_dataset.meta.features.get("observation.state", {}).get("names", [])

    for src_path in tqdm(parquet_files, desc="Mirroring data files"):
        df = pd.read_parquet(src_path).reset_index(drop=True)
        relative_path = src_path.relative_to(src_dataset.root)
        chunk_dir = relative_path.parts[1]
        file_name = relative_path.parts[2]
        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])

        if "action" in df.columns and action_names:
            actions = np.stack(df["action"].values)
            mirrored_actions = np.array([
                mirror_array(row, action_names, mirroring_mask) for row in actions
            ])
            df["action"] = list(mirrored_actions)

        if "observation.state" in df.columns and state_names:
            states = np.stack(df["observation.state"].values)
            mirrored_states = np.array([
                mirror_array(row, state_names, mirroring_mask) for row in states
            ])
            df["observation.state"] = list(mirrored_states)

        dst_path = dst_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_path, index=False)


def _mirror_videos(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    vcodec: str,
    num_workers: int | None = None,
) -> None:
    """Mirror video files by flipping horizontally and swapping left/right names."""
    if not src_dataset.meta.video_keys:
        return

    video_tasks = []
    for old_video_key in src_dataset.meta.video_keys:
        new_video_key = swap_left_right_name(old_video_key)
        for ep_idx in range(src_dataset.meta.total_episodes):
            try:
                src_path = src_dataset.root / src_dataset.meta.get_video_file_path(ep_idx, old_video_key)
                dst_relative = src_dataset.meta.get_video_file_path(ep_idx, old_video_key)
                dst_relative_str = str(dst_relative).replace(old_video_key, new_video_key)
                dst_path = dst_meta.root / dst_relative_str
                if src_path.exists():
                    video_tasks.append((src_path, dst_path))
            except KeyError:
                continue

    def process_video(task, pbar):
        src_path, dst_path = task
        pbar.set_postfix_str(src_path.name)
        flip_video_frames(src_path, dst_path, src_dataset.meta.fps, vcodec)
        return src_path

    if num_workers is None:
        num_workers = os.cpu_count() or 16
    num_workers = min(len(video_tasks), num_workers)
    logger.info(f"Processing {len(video_tasks)} videos with {num_workers} workers")
    with tqdm(total=len(video_tasks), desc="Mirroring videos") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_video, t, pbar): t for t in video_tasks}
            for future in as_completed(futures):
                task = futures[future]
                future.result()
                pbar.set_postfix_str(f"done: {task[0].name}")
                pbar.update(1)


def _copy_episodes_metadata(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
) -> None:
    """Copy episodes metadata with swapped video keys."""
    episodes_dir = src_dataset.root / "meta/episodes"
    dst_episodes_dir = dst_meta.root / "meta/episodes"

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

    dst_meta.info.update({
        "total_episodes": src_dataset.meta.total_episodes,
        "total_frames": src_dataset.meta.total_frames,
        "total_tasks": src_dataset.meta.total_tasks,
        "total_videos": src_dataset.meta.total_videos,
        "total_chunks": src_dataset.meta.total_chunks,
    })
    write_info(dst_meta.info, dst_meta.root)

    if src_dataset.meta.stats is not None:
        mirrored_stats = _mirror_stats(src_dataset.meta.stats)
        write_stats(mirrored_stats, dst_meta.root)


def _mirror_stats(stats: dict) -> dict:
    """Mirror stats by swapping left/right in feature names."""
    mirrored = {}
    for key, value in stats.items():
        new_key = swap_left_right_name(key)
        if isinstance(value, dict):
            mirrored[new_key] = _mirror_stats(value)
        else:
            mirrored[new_key] = value
    return mirrored


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Mirror a bimanual robot dataset")
    parser.add_argument("--repo_id", type=str, required=True, help="Source dataset repo_id")
    parser.add_argument("--output_repo_id", type=str, required=True, help="Output dataset repo_id")
    parser.add_argument("--root", type=str, default=None, help="Source dataset root directory")
    parser.add_argument("--output_root", type=str, default=None, help="Output dataset root directory")
    parser.add_argument("--vcodec", type=str, default="libsvtav1", help="Video codec (libsvtav1, h264, hevc)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers for video processing")
    args = parser.parse_args()

    mirror_dataset(
        repo_id=args.repo_id,
        output_repo_id=args.output_repo_id,
        root=args.root,
        output_root=args.output_root,
        vcodec=args.vcodec,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

