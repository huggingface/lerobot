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
Test script to verify BEHAVIOR-1K dataset loading with v3.0 wrapper.
"""

import argparse
import logging

from behavior_lerobot_dataset_v3 import BehaviorLeRobotDatasetV3

from lerobot.utils.utils import init_logging

init_logging()


def load_behavior1k_dataset(repo_id, root):
    """Test basic dataset loading."""
    logging.info("=" * 80)
    logging.info("Testing BEHAVIOR-1K dataset loading")
    logging.info("=" * 80)

    logging.info(f"\n1. Loading dataset with repo_id: {repo_id}")
    dataset = BehaviorLeRobotDatasetV3(
        repo_id=repo_id,
        root=root,
        modalities=["rgb"],
        cameras=["head"],
        chunk_streaming_using_keyframe=False,
        check_timestamp_sync=False,
    )

    logging.info("\n2. Dataset loaded successfully!")
    logging.info(f"   - Number of episodes: {dataset.num_episodes}")
    logging.info(f"   - Number of frames: {dataset.num_frames}")
    logging.info(f"   - FPS: {dataset.fps}")
    logging.info(f"   - Features: {list(dataset.features)}")

    return dataset


def load_behavior1k_dataset_with_multiple_modalities(repo_id, root):
    """Test loading multiple modalities and cameras."""
    logging.info("\n" + "=" * 80)
    logging.info("Testing multi-modality loading with repo_id: {repo_id}")
    logging.info("=" * 80)

    logging.info(f"\n1. Loading dataset with RGB + Depth with repo_id: {repo_id}")
    dataset = BehaviorLeRobotDatasetV3(
        repo_id=repo_id,
        root=root,
        modalities=["rgb", "depth"],
        cameras=["head", "left_wrist", "right_wrist"],
        chunk_streaming_using_keyframe=False,
        check_timestamp_sync=False,
        video_backend="pyav",
    )

    logging.info(f"\n2. Dataset loaded with modalities: {list(dataset.features)}")
    logging.info(f"   - Total features: {len(dataset.features)}")

    rgb_keys = [k for k in dataset.features if "rgb" in k]
    depth_keys = [k for k in dataset.features if "depth" in k]
    logging.info(f"   - RGB features: {rgb_keys}")
    logging.info(f"   - Depth features: {depth_keys}")

    logging.info("\n3. SUCCESS! Multi-modality loading works.")

    return dataset


def stream_behavior1k_dataset(repo_id, root):
    """Test chunk streaming mode."""
    logging.info("\n" + "=" * 80)
    logging.info("Testing chunk streaming mode")
    logging.info("=" * 80)

    logging.info("\n1. Loading dataset with chunk streaming...")
    dataset = BehaviorLeRobotDatasetV3(
        repo_id=repo_id,
        root=root,
        modalities=["rgb"],
        cameras=["head"],
        chunk_streaming_using_keyframe=True,
        shuffle=True,
        seed=42,
        check_timestamp_sync=False,
    )

    logging.info("\n2. Dataset loaded in streaming mode")
    logging.info(f"   - Number of chunks: {len(dataset.chunks)}")
    logging.info(f"   - First chunk range: {dataset.chunks[0]}")

    logging.info("\n3. Testing frame access in streaming mode...")
    for i in range(min(3, len(dataset))):
        frame = dataset[i]
        logging.info(
            f"   - Frame {i}: episode_index={frame['episode_index'].item()}, "
            f"task_index={frame['task_index'].item()}"
        )

    logging.info("\n4. SUCCESS! Chunk streaming works.")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--root", type=str, default=None)

    args = parser.parse_args()

    load_behavior1k_dataset(args.repo_id, args.root)
    load_behavior1k_dataset_with_multiple_modalities(args.repo_id, args.root)
    stream_behavior1k_dataset(args.repo_id, args.root)
