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

"""Compute per-frame TOPReward progress curves for a LeRobot dataset.

For each episode, scores trajectory prefixes of increasing length using
the TOPReward reward model, min-max normalises the raw log-prob rewards per episode,
and writes a parquet file with one row per frame.

The parquet uses the same schema as SARM's :mod:`lerobot.rewards.sarm.compute_rabc_weights`.

Usage:
    # Sparse-dense mode (15 anchors per episode, matches upstream)
    python -m lerobot.rewards.topreward.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image \\
        --num-samples 15

    # Use a different VLM backbone
    python -m lerobot.rewards.topreward.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image \\
        --vlm-name Qwen/Qwen3-VL-4B-Instruct
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

from lerobot.datasets import LeRobotDataset
from lerobot.rewards.topreward.configuration_topreward import TOPRewardConfig
from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel
from lerobot.rewards.topreward.processor_topreward import TOPRewardEncoderProcessorStep
from lerobot.types import TransitionKey

DEFAULT_OUTPUT_FILENAME = "topreward_progress.parquet"


def get_reward_model_path_from_parquet(parquet_path: Path) -> str | None:
    """Read ``reward_model_path`` from parquet metadata if available."""
    if not parquet_path.exists():
        return None
    try:
        metadata = pq.read_metadata(parquet_path).schema.to_arrow_schema().metadata
        if metadata and b"reward_model_path" in metadata:
            return metadata[b"reward_model_path"].decode()
    except Exception:  # nosec B110
        return None
    return None


def _resolve_task(sample: dict[str, Any], default: str) -> str:
    """Best-effort task extraction from a dataset sample."""
    task = sample.get("task")
    if isinstance(task, str) and task:
        return task
    return default


def normalize_rewards(rewards: list[float] | np.ndarray) -> np.ndarray:
    """Min-max normalise raw log-prob rewards into ``[0, 1]``."""
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    if rewards_arr.size == 0:
        return rewards_arr.astype(np.float32)
    if rewards_arr.size == 1:
        return np.array([1.0], dtype=np.float32)
    r_min, r_max = rewards_arr.min(), rewards_arr.max()
    if r_max == r_min:
        return np.ones_like(rewards_arr, dtype=np.float32)
    return ((rewards_arr - r_min) / (r_max - r_min)).astype(np.float32)


def compute_instruction_rewards_for_prefixes(
    model: TOPRewardModel,
    encoder: TOPRewardEncoderProcessorStep,
    dataset: LeRobotDataset,
    ep_start: int,
    num_frames: int,
    task: str,
    image_key: str,
    num_samples: int | None,
    device: str,
) -> np.ndarray:
    """Score an episode via prefix sweep and return a per-frame normalised curve."""
    if num_samples is None or num_samples >= num_frames:
        prefix_lengths = np.arange(1, num_frames + 1, dtype=np.int64)
    else:
        prefix_lengths = np.unique(np.linspace(1, num_frames, num_samples).round().astype(np.int64))

    episode_frames = torch.stack([dataset[ep_start + i][image_key] for i in range(num_frames)])
    rewards: list[float] = []
    for length in prefix_lengths:
        frames = episode_frames[: int(length)].unsqueeze(0)  # (1, T, C, H, W)

        transition = {
            TransitionKey.OBSERVATION: {image_key: frames},
            TransitionKey.COMPLEMENTARY_DATA: {"task": task},
        }
        encoded = encoder(transition)
        obs = encoded[TransitionKey.OBSERVATION]
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in obs.items()
        }

        with torch.no_grad():
            reward = model.compute_reward(batch)
        rewards.append(float(reward.item()))

    normalized_rewards = normalize_rewards(rewards)

    if prefix_lengths.shape[0] == num_frames:
        return normalized_rewards

    return np.interp(
        np.arange(1, num_frames + 1, dtype=np.float64),
        prefix_lengths.astype(np.float64),
        normalized_rewards.astype(np.float64),
    ).astype(np.float32)


def compute_topreward_progress(
    dataset_repo_id: str,
    reward_model_path: str | None = None,
    vlm_name: str | None = None,
    output_path: str | None = None,
    device: str = "cuda",
    num_samples: int | None = None,
    fps: float | None = None,
    episodes: list[int] | None = None,
) -> Path:
    """Run TOPReward over a dataset and write per-frame progress."""
    if reward_model_path is not None:
        logging.info(f"Loading TOPReward config from: {reward_model_path}")
        model = TOPRewardModel.from_pretrained(reward_model_path)
        config = model.config
        config.device = device
        if vlm_name is not None and vlm_name != config.vlm_name:
            logging.info(f"Overriding vlm_name from config: {config.vlm_name} -> {vlm_name}")
            config.vlm_name = vlm_name
            model = TOPRewardModel(config)
    else:
        config_kwargs: dict[str, Any] = {"device": device}
        if vlm_name is not None:
            config_kwargs["vlm_name"] = vlm_name
        if fps is not None:
            config_kwargs["fps"] = fps
        config = TOPRewardConfig(**config_kwargs)
        logging.info(f"Constructing TOPReward with VLM: {config.vlm_name}")
        model = TOPRewardModel(config)

    model.to(device).eval()

    encoder = TOPRewardEncoderProcessorStep(
        vlm_name=config.vlm_name,
        image_key=config.image_key,
        task_key=config.task_key,
        default_task=config.default_task,
        max_frames=None,  # no tail-crop: we control prefix length explicitly
        fps=config.fps,
        prompt_prefix=config.prompt_prefix,
        prompt_suffix_template=config.prompt_suffix_template,
        add_chat_template=config.add_chat_template,
        max_length=config.max_input_length,
    )

    image_key = config.image_key

    logging.info(f"Loading dataset: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id, download_videos=True)
    logging.info(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    episode_indices = list(range(dataset.num_episodes)) if episodes is None else episodes
    logging.info(f"Processing {len(episode_indices)} episode(s)")

    all_index: list[int] = []
    all_episode: list[int] = []
    all_frame: list[int] = []
    all_progress: list[float] = []

    for episode_idx in tqdm(episode_indices, desc="Episodes"):
        ep = dataset.meta.episodes[episode_idx]
        ep_start = int(ep["dataset_from_index"])
        ep_end = int(ep["dataset_to_index"])
        num_frames = ep_end - ep_start
        if num_frames <= 0:
            continue

        first_sample = dataset[ep_start]
        task = _resolve_task(first_sample, default=config.default_task or "perform the task")

        per_frame = compute_instruction_rewards_for_prefixes(
            model=model,
            encoder=encoder,
            dataset=dataset,
            ep_start=ep_start,
            num_frames=num_frames,
            task=task,
            image_key=image_key,
            num_samples=num_samples,
            device=device,
        )

        for local in range(num_frames):
            all_index.append(ep_start + local)
            all_episode.append(episode_idx)
            all_frame.append(local)
            all_progress.append(float(per_frame[local]))

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    table = pa.table(
        {
            "index": np.asarray(all_index, dtype=np.int64),
            "episode_index": np.asarray(all_episode, dtype=np.int64),
            "frame_index": np.asarray(all_frame, dtype=np.int64),
            "progress_sparse": np.asarray(all_progress, dtype=np.float32),
        }
    )

    schema_metadata: dict[bytes, bytes] = {b"vlm_name": config.vlm_name.encode()}
    if reward_model_path is not None:
        schema_metadata[b"reward_model_path"] = reward_model_path.encode()
    table = table.replace_schema_metadata(schema_metadata)

    out = Path(dataset.root) / DEFAULT_OUTPUT_FILENAME if output_path is None else Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    logging.info(f"Saved {len(table)} frame values to {out}")

    progress_arr = np.asarray(all_progress, dtype=np.float32)
    if progress_arr.size:
        logging.info(
            f"Progress: mean={float(progress_arr.mean()):.4f}, "
            f"std={float(progress_arr.std()):.4f}, "
            f"min={float(progress_arr.min()):.4f}, "
            f"max={float(progress_arr.max()):.4f}"
        )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-frame TOPReward progress curves for RA-BC weighting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Sparse-dense mode (matches upstream TOPReward num_samples=15)
    python -m lerobot.rewards.topreward.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image \\
        --num-samples 15

    # Use a smaller VLM
    python -m lerobot.rewards.topreward.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image \\
        --vlm-name Qwen/Qwen3-VL-4B-Instruct
        """,
    )
    parser.add_argument(
        "--dataset-repo-id", type=str, required=True, help="HuggingFace dataset repo id or local path."
    )
    parser.add_argument(
        "--reward-model-path", type=str, default=None, help="Optional TOPReward LeRobot config."
    )
    parser.add_argument("--vlm-name", type=str, default=None, help="Override the VLM backbone (HF Hub id).")
    parser.add_argument("--output-path", type=str, default=None, help="Output parquet path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda).")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Anchor prefix samples per episode. None = dense. 15 matches upstream.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Process only these episode indices (e.g. --episodes 0 or --episodes 0 5 10).",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override TOPRewardConfig.fps.")
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Upload to the dataset repo on HuggingFace Hub."
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_path = compute_topreward_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=args.reward_model_path,
        vlm_name=args.vlm_name,
        output_path=args.output_path,
        device=args.device,
        num_samples=args.num_samples,
        fps=args.fps,
        episodes=args.episodes,
    )

    print(f"\nTOPReward progress saved to: {output_path}")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        hub_path = DEFAULT_OUTPUT_FILENAME

        print(f"\nUploading to Hub: {args.dataset_repo_id}/{hub_path}")
        api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo=hub_path,
            repo_id=args.dataset_repo_id,
            repo_type="dataset",
        )
        print(
            "Successfully uploaded to: "
            f"https://huggingface.co/datasets/{args.dataset_repo_id}/blob/main/{hub_path}"
        )

        print("\nTo use in training, add to your config:")
        print("  use_rabc: true")
        print(f"  rabc_progress_path: hf://datasets/{args.dataset_repo_id}/{hub_path}")
        print("  rabc_head_mode: sparse")
    else:
        print("\nTo use in training, add to your config:")
        print("  use_rabc: true")
        print(f"  rabc_progress_path: {output_path}")
        print("  rabc_head_mode: sparse")


if __name__ == "__main__":
    main()
