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
Automatic Skill Annotation for LeRobot Datasets.

This script performs automatic subtask/skill labeling for ANY LeRobot dataset using
Vision-Language Models (VLMs). It segments each robot demonstration into short atomic
skills (1-3 seconds each) and creates a new dataset with subtask annotations.

The pipeline:
1. Loads a LeRobot dataset (local or from HuggingFace Hub)
2. For each episode, extracts video frames
3. Uses a VLM to identify skill boundaries and labels
4. Creates a subtasks.parquet file with unique subtasks
5. Adds a subtask_index feature to the dataset

Supported VLMs (modular design): Qwen2-VL, Qwen3-VL, Qwen3.5-VL (see vlm_annotations.py).

Usage:
  lerobot-dataset-subtask-annotate --repo_id=user/dataset --video_key=observation.images.base ...
  lerobot-dataset-subtask-annotate --data_dir=/path/to/dataset --video_key=observation.images.base ...
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from lerobot.configs import parser
from lerobot.data_processing.data_annotations.subtask_annotations import (
    SkillAnnotator,
    save_skill_annotations,
)
from lerobot.data_processing.data_annotations.vlm_annotations import get_vlm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class SubtaskAnnotateConfig:
    """Configuration for automatic subtask/skill annotation with VLMs."""

    # Data source: provide exactly one of data_dir (local) or repo_id (Hub)
    data_dir: str | None = None
    repo_id: str | None = None
    # Video observation key (e.g. observation.images.base)
    video_key: str = "observation.images.base"
    # VLM model name (default: Qwen/Qwen2-VL-7B-Instruct)
    model: str = "Qwen/Qwen2-VL-7B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    batch_size: int = 8
    # Episode selection (default: all)
    episodes: list[int] | None = None
    skip_existing: bool = False
    # Output
    output_dir: str | None = None
    output_repo_id: str | None = None
    push_to_hub: bool = False
    # Closed vocabulary: model must choose only from these labels
    subtask_labels: list[str] | None = None
    # Disable timer overlay on video (by default a timer is drawn for the VLM)
    no_timer_overlay: bool = False


@parser.wrap()
def subtask_annotate(cfg: SubtaskAnnotateConfig):
    """
    Run automatic skill annotation on a LeRobot dataset using a VLM.

    Args:
        cfg: SubtaskAnnotateConfig with data source, model, and output options.
    """
    if (cfg.data_dir is None) == (cfg.repo_id is None):
        raise ValueError("Provide exactly one of --data_dir or --repo_id")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[cfg.dtype]

    print("Loading dataset...")
    if cfg.data_dir:
        dataset = LeRobotDataset(
            repo_id="local/dataset", root=cfg.data_dir, download_videos=False
        )
    else:
        dataset = LeRobotDataset(repo_id=cfg.repo_id, download_videos=True)

    print(f" Loaded dataset with {dataset.meta.total_episodes} episodes")

    if cfg.video_key not in dataset.meta.video_keys:
        available = ", ".join(dataset.meta.video_keys)
        raise ValueError(
            f"Video key '{cfg.video_key}' not found. Available: {available}"
        )

    print(f"Initializing VLM: {cfg.model}...")
    vlm = get_vlm(cfg.model, cfg.device, torch_dtype)

    add_timer_overlay = not cfg.no_timer_overlay
    annotator = SkillAnnotator(
        vlm=vlm,
        batch_size=cfg.batch_size,
        add_timer_overlay=add_timer_overlay,
    )
    print(f"Processing with batch size: {cfg.batch_size}")
    annotations = annotator.annotate_dataset(
        dataset=dataset,
        video_key=cfg.video_key,
        episodes=cfg.episodes,
        skip_existing=cfg.skip_existing,
        subtask_labels=cfg.subtask_labels,
    )

    output_dir = Path(cfg.output_dir) if cfg.output_dir else None
    output_repo_id = cfg.output_repo_id
    new_dataset = save_skill_annotations(
        dataset, annotations, output_dir, output_repo_id
    )

    total_skills = sum(len(ann.skills) for ann in annotations.values())
    print("\nAnnotation complete!")
    print(f"Episodes annotated: {len(annotations)}")
    print(f"Total subtasks identified: {total_skills}")
    print(f"Dataset with subtask_index saved to: {new_dataset.root}")

    if cfg.push_to_hub:
        if cfg.data_dir:
            print("Warning: --push_to_hub requires --repo_id, skipping...")
        else:
            print("Pushing to HuggingFace Hub...")
            try:
                new_dataset.push_to_hub(branch="subtasks")
                print(f" Pushed to {output_repo_id or cfg.repo_id}")
            except Exception as e:
                print(f"Push failed: {e}")


def main():
    """CLI entry point that parses config and runs subtask annotation."""
    subtask_annotate()


if __name__ == "__main__":
    main()
