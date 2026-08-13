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

"""Evaluate RynnValue on causal prefixes of a LeRobot dataset.

The primary outputs are the model's remaining-time prediction and its
higher-is-better potential. Optionally, ``--max-remaining-time-s`` adds a
bounded ``progress_sparse`` column compatible with the existing RA-BC loader.

Example:
    uv run python -m lerobot.rewards.rynnvalue.compute_rabc_weights \
        --dataset-repo-id lilkm/stackblocks_recap_all_for_vf_v2 \
        --reward-model-path outputs/rynnvalue-4b \
        --output-path outputs/rynnvalue/stackblocks_top.parquet \
        --image-key observation.images.top \
        --robot-description "an SO-100 single-arm robot" \
        --camera-description "a fixed third-person top camera" \
        --inference-fps 1 \
        --episodes 0 1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.configs.rewards import RewardModelConfig
from lerobot.datasets import LeRobotDataset
from lerobot.rewards.rynnvalue.configuration_rynnvalue import (
    RYNNVALUE_FEATURE_PREFIX,
    RynnValueConfig,
)
from lerobot.rewards.rynnvalue.modeling_rynnvalue import RynnValueRewardModel
from lerobot.rewards.rynnvalue.processor_rynnvalue import (
    RynnValueEncoderProcessorStep,
    _video_to_pil,
)

DEFAULT_OUTPUT_FILENAME = "rynnvalue_values.parquet"


def _select_anchor_indices(num_frames: int, dataset_fps: float, inference_fps: float) -> np.ndarray:
    """Select approximately evenly timed anchors, always including both boundaries."""
    if num_frames < 1:
        return np.empty(0, dtype=np.int64)
    if dataset_fps <= 0:
        raise ValueError(f"dataset_fps must be positive, got {dataset_fps}")
    if inference_fps <= 0 or inference_fps > dataset_fps:
        raise ValueError(f"inference_fps must be in (0, dataset_fps], got {inference_fps} for {dataset_fps=}")

    stride = dataset_fps / inference_fps
    anchors = np.rint(np.arange(0, num_frames, stride)).astype(np.int64)
    anchors = np.unique(np.clip(anchors, 0, num_frames - 1))
    if anchors[-1] != num_frames - 1:
        anchors = np.append(anchors, num_frames - 1)
    return anchors


def _select_prefix_indices(anchor_index: int, max_frames: int | None) -> np.ndarray:
    """Select a chronological causal prefix ending exactly at ``anchor_index``."""
    if anchor_index < 0:
        raise ValueError(f"anchor_index must be non-negative, got {anchor_index}")
    prefix_length = anchor_index + 1
    if max_frames is None or prefix_length <= max_frames:
        return np.arange(prefix_length, dtype=np.int64)
    if max_frames < 1:
        raise ValueError(f"max_frames must be positive or None, got {max_frames}")
    if max_frames == 1:
        return np.asarray([anchor_index], dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, anchor_index, max_frames)).astype(np.int64))


def _interpolate_anchor_values(
    num_frames: int, anchor_indices: np.ndarray, anchor_values: np.ndarray
) -> np.ndarray:
    """Linearly interpolate anchor predictions to every frame in one episode."""
    if num_frames < 1:
        return np.empty(0, dtype=np.float32)
    if anchor_indices.ndim != 1 or anchor_values.ndim != 1:
        raise ValueError("anchor_indices and anchor_values must be one-dimensional")
    if len(anchor_indices) != len(anchor_values) or not len(anchor_indices):
        raise ValueError("anchor_indices and anchor_values must have the same non-zero length")
    if np.any(np.diff(anchor_indices) <= 0):
        raise ValueError("anchor_indices must be strictly increasing")
    if anchor_indices[0] != 0 or anchor_indices[-1] != num_frames - 1:
        raise ValueError("anchor_indices must include the first and last frame")
    return np.interp(np.arange(num_frames), anchor_indices, anchor_values).astype(np.float32)


def _remaining_time_to_progress(remaining_time_s: np.ndarray, max_remaining_time_s: float) -> np.ndarray:
    """Map physical remaining time to bounded RA-BC-compatible progress."""
    if max_remaining_time_s <= 0:
        raise ValueError(f"max_remaining_time_s must be positive, got {max_remaining_time_s}")
    return np.clip(1.0 - remaining_time_s / max_remaining_time_s, 0.0, 1.0).astype(np.float32)


def _build_episode_table(
    *,
    global_start: int,
    episode_index: int,
    remaining_time_s: np.ndarray,
    anchor_indices: np.ndarray,
    max_remaining_time_s: float | None,
) -> pa.Table:
    """Build the evaluation schema for one densely interpolated episode."""
    num_frames = len(remaining_time_s)
    is_inference_frame = np.zeros(num_frames, dtype=np.bool_)
    is_inference_frame[anchor_indices] = True
    data: dict[str, Any] = {
        "index": np.arange(global_start, global_start + num_frames, dtype=np.int64),
        "episode_index": np.full(num_frames, episode_index, dtype=np.int64),
        "frame_index": np.arange(num_frames, dtype=np.int64),
        "remaining_time_s": remaining_time_s.astype(np.float32),
        "potential": (-remaining_time_s).astype(np.float32),
        "is_inference_frame": is_inference_frame,
    }
    if max_remaining_time_s is not None:
        data["progress_sparse"] = _remaining_time_to_progress(remaining_time_s, max_remaining_time_s)
    return pa.table(data)


def _resolve_task(sample: dict[str, Any], default_task: str | None) -> str:
    task = sample.get("task")
    if isinstance(task, str) and task:
        return task
    if default_task:
        return default_task
    raise KeyError("Dataset sample has no task string; pass --default-task explicitly")


def compute_rynnvalue_values(
    dataset_repo_id: str,
    reward_model_path: str,
    *,
    output_path: str | Path | None = None,
    device: str = "cuda",
    batch_size: int = 2,
    inference_fps: float = 1.0,
    max_frames: int | None = 8,
    episodes: list[int] | None = None,
    image_key: str | None = None,
    default_task: str | None = None,
    robot_description: str | None = None,
    camera_description: str | None = None,
    max_remaining_time_s: float | None = None,
) -> Path:
    """Evaluate RynnValue on dataset prefixes and write dense per-frame values."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_frames is not None and max_frames < 1:
        raise ValueError(f"max_frames must be positive or None, got {max_frames}")
    if max_remaining_time_s is not None and max_remaining_time_s <= 0:
        raise ValueError(f"max_remaining_time_s must be positive or None, got {max_remaining_time_s}")

    config = RewardModelConfig.from_pretrained(reward_model_path)
    if not isinstance(config, RynnValueConfig):
        raise TypeError(f"Expected a RynnValue checkpoint, got {type(config).__name__}")
    config.device = device
    config.reward_output = "remaining_time"
    config.max_frames = max_frames
    if image_key is not None:
        config.image_key = image_key
    if default_task is not None:
        config.default_task = default_task
    if robot_description is not None:
        config.robot_description = robot_description
    if camera_description is not None:
        config.camera_description = camera_description
    if config.use_meta is not False and not (config.robot_description or config.camera_description):
        raise ValueError("RynnValue metadata prompting requires --robot-description or --camera-description")

    logging.info("Loading RynnValue checkpoint: %s", reward_model_path)
    model = RynnValueRewardModel.from_pretrained(reward_model_path, config=config)
    model.eval()
    encoder = RynnValueEncoderProcessorStep(
        model_id=reward_model_path,
        model_revision=config.pretrained_revision,
        image_key=config.image_key,
        task_key=config.task_key,
        default_task=config.default_task,
        max_frames=None,
        robot_description=config.robot_description,
        camera_description=config.camera_description,
        use_meta=config.use_meta,
    )

    logging.info("Loading dataset: %s", dataset_repo_id)
    dataset = LeRobotDataset(dataset_repo_id, download_videos=True)
    episode_indices = list(range(dataset.num_episodes)) if episodes is None else episodes
    invalid_episodes = [index for index in episode_indices if not 0 <= index < dataset.num_episodes]
    if invalid_episodes:
        raise IndexError(
            f"Episode indices out of range for {dataset.num_episodes} episodes: {invalid_episodes}"
        )

    tables: list[pa.Table] = []
    for episode_index in tqdm(episode_indices, desc="Episodes"):
        episode = dataset.meta.episodes[episode_index]
        global_start = int(episode["dataset_from_index"])
        global_end = int(episode["dataset_to_index"])
        num_frames = global_end - global_start
        if num_frames < 1:
            continue

        task = _resolve_task(dataset[global_start], config.default_task)
        anchor_indices = _select_anchor_indices(num_frames, dataset.fps, inference_fps)
        anchor_values: list[float] = []

        for batch_start in tqdm(
            range(0, len(anchor_indices), batch_size),
            desc=f"  Episode {episode_index}",
            leave=False,
        ):
            batch_anchors = anchor_indices[batch_start : batch_start + batch_size]
            samples = []
            for anchor_index in batch_anchors:
                prefix_indices = _select_prefix_indices(int(anchor_index), max_frames)
                frames = torch.stack(
                    [dataset[global_start + int(index)][config.image_key] for index in prefix_indices]
                )
                samples.append((_video_to_pil(frames, max_frames=None), task))

            encoded = encoder.encode_samples(samples)
            model_batch = {f"{RYNNVALUE_FEATURE_PREFIX}{key}": value for key, value in encoded.items()}
            remaining_time = model.compute_reward(model_batch)
            anchor_values.extend(remaining_time.detach().float().cpu().tolist())

        dense_remaining_time = _interpolate_anchor_values(
            num_frames,
            anchor_indices,
            np.asarray(anchor_values, dtype=np.float32),
        )
        tables.append(
            _build_episode_table(
                global_start=global_start,
                episode_index=episode_index,
                remaining_time_s=dense_remaining_time,
                anchor_indices=anchor_indices,
                max_remaining_time_s=max_remaining_time_s,
            )
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if not tables:
        raise ValueError("No non-empty episodes were selected")
    table = pa.concat_tables(tables)
    metadata = {
        "adapter": "rynnvalue_dataset_evaluation",
        "dataset_repo_id": dataset_repo_id,
        "reward_model_path": reward_model_path,
        "image_key": config.image_key,
        "dataset_fps": dataset.fps,
        "inference_fps": inference_fps,
        "max_frames": max_frames,
        "robot_description": config.robot_description,
        "camera_description": config.camera_description,
        "max_remaining_time_s": max_remaining_time_s,
    }
    table = table.replace_schema_metadata(
        {
            b"rynnvalue_evaluation": json.dumps(metadata, sort_keys=True).encode(),
            b"reward_model_path": reward_model_path.encode(),
        }
    )

    destination = Path(dataset.root) / DEFAULT_OUTPUT_FILENAME if output_path is None else Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination)
    logging.info("Saved %d frame values to %s", len(table), destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--reward-model-path", required=True)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--inference-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--image-key")
    parser.add_argument("--default-task")
    parser.add_argument("--robot-description")
    parser.add_argument("--camera-description")
    parser.add_argument(
        "--max-remaining-time-s",
        type=float,
        help="Optional task horizon used to add RA-BC-compatible progress_sparse.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    destination = compute_rynnvalue_values(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=args.reward_model_path,
        output_path=args.output_path,
        device=args.device,
        batch_size=args.batch_size,
        inference_fps=args.inference_fps,
        max_frames=args.max_frames,
        episodes=args.episodes,
        image_key=args.image_key,
        default_task=args.default_task,
        robot_description=args.robot_description,
        camera_description=args.camera_description,
        max_remaining_time_s=args.max_remaining_time_s,
    )
    print(f"RynnValue evaluation saved to: {destination}")
    if args.max_remaining_time_s is not None:
        print("RA-BC compatibility column written: progress_sparse")


if __name__ == "__main__":
    main()
