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

"""Compute per-frame SOLE-R1 progress for a LeRobot dataset.

Each episode is evaluated autoregressively using SOLE-R1. When
``num_samples`` is set, only uniformly spaced anchor frames are evaluated
and the model interpolates the predictions back to every original frame.

The output parquet uses the same schema as the SARM and TOPReward
RA-BC utilities.
"""

from __future__ import annotations

import argparse
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
from lerobot.rewards.soler1.configuration_soler1 import SOLER1Config
from lerobot.rewards.soler1.modeling_soler1 import SOLER1RewardModel
from lerobot.rewards.soler1.processor_soler1 import make_soler1_pre_post_processors

DEFAULT_OUTPUT_FILENAME = "soler1_progress.parquet"


def _resolve_task(sample: dict[str, Any], task_key: str, default_task: str | None) -> str:
    """Extract a non-empty task description from an episode sample."""
    task = sample.get(task_key, default_task)
    if not isinstance(task, str) or not task:
        raise KeyError(
            f"SOLE-R1 expected a non-empty task description under {task_key!r}. "
            "Set --default-task if the dataset has no task column."
        )
    return task


def _load_config(
    *,
    reward_model_path: str | None,
    model_name: str | None,
    device: str,
    external_image_key: str,
    wrist_image_key: str | None,
    task_key: str,
    default_task: str | None,
    num_samples: int | None,
) -> SOLER1Config:
    """Load or construct a SOLE-R1 configuration without loading the VLM twice."""
    if reward_model_path is None:
        config = SOLER1Config()
    else:
        loaded_config = RewardModelConfig.from_pretrained(pretrained_name_or_path=reward_model_path)
        if not isinstance(loaded_config, SOLER1Config):
            raise TypeError(
                f"Expected SOLER1Config at {reward_model_path!r}, got {type(loaded_config).__name__}"
            )
        config = loaded_config

    config.device = device
    config.external_image_key = external_image_key
    config.wrist_image_key = wrist_image_key
    config.task_key = task_key
    config.default_task = default_task
    config.num_samples = num_samples
    config.reward_output = "progress"

    if model_name is not None:
        config.model_name = model_name

    # Re-run validation after applying CLI overrides.
    config.__post_init__()
    return config


def compute_episode_progress(
    *,
    model: SOLER1RewardModel,
    preprocessor: Any,
    dataset: LeRobotDataset,
    config: SOLER1Config,
    start: int,
    end: int,
    task: str,
) -> np.ndarray:
    """Compute one dense progress value per frame in an episode."""
    samples = [dataset[index] for index in range(start, end)]

    batch: dict[str, Any] = {
        config.external_image_key: torch.stack([sample[config.external_image_key] for sample in samples]),
        config.task_key: task,
    }

    if config.wrist_image_key is not None:
        batch[config.wrist_image_key] = torch.stack([sample[config.wrist_image_key] for sample in samples])

    processed = preprocessor(batch)

    with torch.no_grad():
        progress = model.compute_progress(processed)

    return progress.squeeze(0).detach().cpu().numpy().astype(np.float32)


def compute_soler1_progress(
    dataset_repo_id: str,
    *,
    reward_model_path: str | None = None,
    model_name: str | None = None,
    output_path: str | None = None,
    device: str = "cuda",
    external_image_key: str = "observation.images.top",
    wrist_image_key: str | None = None,
    task_key: str = "task",
    default_task: str | None = None,
    num_samples: int | None = 10,
    episodes: list[int] | None = None,
) -> Path:
    """Run SOLE-R1 over selected dataset episodes and write a parquet file."""
    if num_samples is not None and num_samples < 1:
        raise ValueError(f"num_samples must be >= 1 or None, got {num_samples}")

    config = _load_config(
        reward_model_path=reward_model_path,
        model_name=model_name,
        device=device,
        external_image_key=external_image_key,
        wrist_image_key=wrist_image_key,
        task_key=task_key,
        default_task=default_task,
        num_samples=num_samples,
    )

    logging.info("Loading SOLE-R1 model: %s", config.model_name)
    model = SOLER1RewardModel(config).to(device).eval()
    preprocessor, _ = make_soler1_pre_post_processors(config)

    logging.info("Loading dataset: %s", dataset_repo_id)
    dataset = LeRobotDataset(dataset_repo_id, download_videos=True)

    episode_indices = list(range(dataset.num_episodes)) if episodes is None else episodes
    invalid_episodes = [index for index in episode_indices if index < 0 or index >= dataset.num_episodes]
    if invalid_episodes:
        raise ValueError(
            f"Invalid episode indices {invalid_episodes}; dataset has {dataset.num_episodes} episodes"
        )

    all_indices: list[int] = []
    all_episode_indices: list[int] = []
    all_frame_indices: list[int] = []
    all_progress: list[float] = []

    for episode_index in tqdm(episode_indices, desc="Episodes"):
        episode = dataset.meta.episodes[episode_index]
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])

        if end <= start:
            logging.warning("Skipping empty episode %d", episode_index)
            continue

        task = _resolve_task(
            dataset[start],
            task_key=config.task_key,
            default_task=config.default_task,
        )
        progress = compute_episode_progress(
            model=model,
            preprocessor=preprocessor,
            dataset=dataset,
            config=config,
            start=start,
            end=end,
            task=task,
        )

        expected_length = end - start
        if progress.shape != (expected_length,):
            raise RuntimeError(
                f"SOLE-R1 returned shape {progress.shape} for episode "
                f"{episode_index}; expected {(expected_length,)}"
            )

        for frame_index, value in enumerate(progress):
            all_indices.append(start + frame_index)
            all_episode_indices.append(episode_index)
            all_frame_indices.append(frame_index)
            all_progress.append(float(value))

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    table = pa.table(
        {
            "index": np.asarray(all_indices, dtype=np.int64),
            "episode_index": np.asarray(all_episode_indices, dtype=np.int64),
            "frame_index": np.asarray(all_frame_indices, dtype=np.int64),
            "progress_sparse": np.asarray(all_progress, dtype=np.float32),
        }
    ).replace_schema_metadata(
        {
            b"reward_model_type": b"sole-r1",
            b"model_name": config.model_name.encode(),
        }
    )

    output = Path(dataset.root) / DEFAULT_OUTPUT_FILENAME if output_path is None else Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output)

    logging.info("Saved %d frame values to %s", len(table), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-frame SOLE-R1 progress for a LeRobot dataset.")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--reward-model-path")
    parser.add_argument("--model-name")
    parser.add_argument("--output-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--external-image-key",
        default="observation.images.top",
    )
    parser.add_argument("--wrist-image-key")
    parser.add_argument("--task-key", default="task")
    parser.add_argument("--default-task")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Anchor frames per episode. Use 0 to evaluate every frame.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="Episode indices to process, for example --episodes 0 3 5.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    num_samples = None if args.num_samples == 0 else args.num_samples

    output = compute_soler1_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=args.reward_model_path,
        model_name=args.model_name,
        output_path=args.output_path,
        device=args.device,
        external_image_key=args.external_image_key,
        wrist_image_key=args.wrist_image_key,
        task_key=args.task_key,
        default_task=args.default_task,
        num_samples=num_samples,
        episodes=args.episodes,
    )

    print(f"\nSOLE-R1 progress saved to: {output}")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(output),
            path_in_repo=DEFAULT_OUTPUT_FILENAME,
            repo_id=args.dataset_repo_id,
            repo_type="dataset",
        )
        print(
            "Uploaded to "
            f"https://huggingface.co/datasets/{args.dataset_repo_id}/"
            f"blob/main/{DEFAULT_OUTPUT_FILENAME}"
        )


if __name__ == "__main__":
    main()
