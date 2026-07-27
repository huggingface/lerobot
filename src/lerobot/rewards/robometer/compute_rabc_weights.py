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

"""Compute per-frame Robometer progress and success curves for a LeRobot dataset.

For each episode, builds per-frame sub-samples using the frame-steps
strategy from the Robometer eval server: for each original frame ``t``,
linspace-subsample ``[0, t]`` into ``K`` frames (default 4, matching
``NUM_SUBSAMPLED_FRAMES`` in the eval server), run one forward through
the Robometer processor + model, and keep the last-frame progress value.
All sub-samples are the same size ``K`` so they batch cleanly.

The parquet uses the same schema as SARM's
:mod:`lerobot.rewards.sarm.compute_rabc_weights` so existing consumers —
:class:`lerobot.rewards.sarm.rabc.RABCWeights` (which reads
``progress_sparse``) and the progress-overlay script in
``examples/dataset/create_progress_videos.py`` — work without modification.

Usage:
    # Dense per-frame progress for one episode
    python -m lerobot.rewards.robometer.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image \\
        --reward-model-path lerobot/Robometer-4B \\
        --episodes 0

    # All episodes with batching
    python -m lerobot.rewards.robometer.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image \\
        --reward-model-path lerobot/Robometer-4B \\
        --batch-size 16
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
from lerobot.rewards.robometer.configuration_robometer import RobometerConfig
from lerobot.rewards.robometer.modeling_robometer import RobometerRewardModel
from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep
from lerobot.types import TransitionKey

DEFAULT_OUTPUT_FILENAME = "robometer_progress.parquet"

# Upstream Robometer eval server uses K=4 for frame-steps sub-samples.
DEFAULT_NUM_SUBSAMPLED_FRAMES = 4


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


def _build_subsample_indices(num_frames: int, num_subsampled_frames: int) -> list[np.ndarray]:
    """Frame-steps linspace expansion.

    For each ``t in [0, num_frames - 1]`` returns ``num_subsampled_frames``
    indices from ``np.linspace(0, t, num_subsampled_frames)`` — the first
    and last frames are always included. Each entry is a fixed-size array
    so the model can batch them.
    """
    return [np.linspace(0, t, num_subsampled_frames).round().astype(np.int64) for t in range(num_frames)]


def compute_robometer_progress(
    dataset_repo_id: str,
    reward_model_path: str,
    output_path: str | None = None,
    device: str = "cuda",
    batch_size: int = 32,
    num_subsampled_frames: int = DEFAULT_NUM_SUBSAMPLED_FRAMES,
    episodes: list[int] | None = None,
    image_key: str | None = None,
) -> Path:
    """Run Robometer over a dataset and write per-frame progress + success."""
    logging.info(f"Loading Robometer: {reward_model_path}")
    config = RobometerConfig(pretrained_path=reward_model_path, device=device)
    if image_key is not None:
        config.image_key = image_key
    model = RobometerRewardModel.from_pretrained(reward_model_path, config=config)
    model.to(device).eval()

    encoder = RobometerEncoderProcessorStep(
        base_model_id=config.base_model_id,
        image_key=config.image_key,
        task_key=config.task_key,
        default_task=config.default_task,
        max_frames=num_subsampled_frames,
        use_multi_image=config.use_multi_image,
        use_per_frame_progress_token=config.use_per_frame_progress_token,
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

        ep_frames = torch.stack([dataset[ep_start + i][image_key] for i in range(num_frames)])

        sub_indices = _build_subsample_indices(num_frames, num_subsampled_frames)

        progress_per_frame = np.zeros(num_frames, dtype=np.float32)

        for start in tqdm(range(0, num_frames, batch_size), desc=f"  Ep {episode_idx}", leave=False):
            end = min(start + batch_size, num_frames)
            frames_batch = torch.stack([ep_frames[sub_indices[i]] for i in range(start, end)])

            transition = {
                TransitionKey.OBSERVATION: {image_key: frames_batch},
                TransitionKey.COMPLEMENTARY_DATA: {"task": task},
            }
            encoded = encoder(transition)
            obs = encoded[TransitionKey.OBSERVATION]
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in obs.items()
            }

            with torch.no_grad():
                rewards = model.compute_reward(batch)
            progress_per_frame[start:end] = rewards.cpu().numpy()

        for local in range(num_frames):
            all_index.append(ep_start + local)
            all_episode.append(episode_idx)
            all_frame.append(local)
            all_progress.append(float(progress_per_frame[local]))

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    table = pa.table(
        {
            "index": np.asarray(all_index, dtype=np.int64),
            "episode_index": np.asarray(all_episode, dtype=np.int64),
            "frame_index": np.asarray(all_frame, dtype=np.int64),
            "progress_sparse": np.asarray(all_progress, dtype=np.float32),
        }
    ).replace_schema_metadata({b"reward_model_path": reward_model_path.encode()})

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
        description="Compute per-frame Robometer progress curves for RA-BC weighting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dense per-frame progress for one episode
    python -m lerobot.rewards.robometer.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image \\
        --reward-model-path lerobot/Robometer-4B \\
        --episodes 0

    # All episodes, smaller batches for memory-constrained GPUs
    python -m lerobot.rewards.robometer.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image \\
        --reward-model-path lerobot/Robometer-4B \\
        --batch-size 16
        """,
    )
    parser.add_argument(
        "--dataset-repo-id", type=str, required=True, help="HuggingFace dataset repo id or local path."
    )
    parser.add_argument(
        "--reward-model-path", type=str, default=None, help="Robometer checkpoint repo id or local path."
    )
    parser.add_argument("--output-path", type=str, default=None, help="Output parquet path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda).")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Sub-samples per Qwen forward (default: 32)."
    )
    parser.add_argument(
        "--num-subsampled-frames",
        type=int,
        default=DEFAULT_NUM_SUBSAMPLED_FRAMES,
        help=f"Frames per sub-sample (default: {DEFAULT_NUM_SUBSAMPLED_FRAMES}, matches eval server).",
    )
    parser.add_argument(
        "--episodes", type=int, nargs="+", default=None, help="Process only these episode indices."
    )
    parser.add_argument(
        "--image-key", type=str, default=None, help="Image observation key (default: from config)."
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Upload to the dataset repo on HuggingFace Hub."
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    reward_model_path = args.reward_model_path
    if reward_model_path is None:
        temp_dataset = LeRobotDataset(args.dataset_repo_id, download_videos=False)
        parquet_path = Path(temp_dataset.root) / DEFAULT_OUTPUT_FILENAME
        reward_model_path = get_reward_model_path_from_parquet(parquet_path)
        if reward_model_path:
            logging.info(f"Using reward model from parquet metadata: {reward_model_path}")
        else:
            raise ValueError(
                "--reward-model-path is required (no existing parquet with model metadata found)."
            )

    output_path = compute_robometer_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=reward_model_path,
        output_path=args.output_path,
        device=args.device,
        batch_size=args.batch_size,
        num_subsampled_frames=args.num_subsampled_frames,
        episodes=args.episodes,
        image_key=args.image_key,
    )

    print(f"\nRobometer progress saved to: {output_path}")

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
