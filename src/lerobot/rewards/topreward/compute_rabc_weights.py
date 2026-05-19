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

This mirrors :mod:`lerobot.rewards.sarm.compute_rabc_weights` (and the
ROBOMETER equivalent): it walks every episode in a dataset, runs the
TOPReward zero-shot reward model, and writes a parquet file with one row
per frame. The output uses the same schema SARM produces so existing
consumers — :class:`lerobot.rewards.sarm.rabc.RABCWeights` (which reads
``progress_sparse``) and the SARM-style overlay script in
``examples/dataset/create_progress_videos.py`` — work without modification.

TOPReward is zero-shot: there is no fine-tuned checkpoint to load. The
``--reward-model-path`` argument is therefore optional and only used when
you want to load a TOPReward LeRobot config (e.g. one published on the Hub
that pins ``vlm_name`` and prompt knobs). Otherwise the default
:class:`TOPRewardConfig` is used, which points at
``Qwen/Qwen3-VL-8B-Instruct`` — the VLM is re-downloaded from the HF Hub
on every run unless cached.

Parquet schema:
    +--------------------+---------+----------------------------------------+
    | column             | dtype   | meaning                                |
    +====================+=========+========================================+
    | ``index``          | int64   | global frame index                     |
    | ``episode_index``  | int64   | episode id                             |
    | ``frame_index``    | int64   | local within-episode index             |
    | ``progress_sparse``| float32 | per-frame TOPReward progress in [0, 1] |
    |                    |         | (RA-BC + overlay read this column)     |
    +--------------------+---------+----------------------------------------+

Usage:
    # Full computation (one VLM forward per frame, slowest but most accurate)
    python -m lerobot.rewards.topreward.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image

    # Sparse-dense mode: 15 anchor prefixes per episode, interpolated to
    # per-frame resolution. Matches upstream TOPReward ``num_samples=15``.
    python -m lerobot.rewards.topreward.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image \\
        --num-prefixes 15

    # Use a different VLM backbone
    python -m lerobot.rewards.topreward.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image \\
        --vlm-name Qwen/Qwen3-VL-4B-Instruct

The output is written to the dataset's local cache directory as
``topreward_progress.parquet`` (or to ``--output-path`` if provided).
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
from lerobot.rewards.topreward.processor_topreward import TOPREWARD_FEATURE_PREFIX

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


def _frames_to_uint8_hwc(video: torch.Tensor) -> np.ndarray:
    """Convert a ``(T, C, H, W)`` or ``(T, H, W, C)`` tensor to ``(T, H, W, C) uint8``.

    Inlined here (rather than reusing the processor) so the labeling script
    can side-step the ``max_frames`` tail-crop and feed full trajectories
    to :meth:`TOPRewardModel.predict_curves`.
    """
    if video.shape[1] in (1, 3):
        video = video.permute(0, 2, 3, 1)
    elif video.shape[-1] not in (1, 3):
        raise ValueError(f"Expected channel dim of size 1 or 3, got shape {tuple(video.shape)}")

    array = video.detach().cpu().numpy()
    if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def compute_topreward_progress(
    dataset_repo_id: str,
    reward_model_path: str | None = None,
    vlm_name: str | None = None,
    output_path: str | None = None,
    device: str = "cuda",
    num_prefixes: int | None = None,
    fps: float | None = None,
    reduction: str | None = None,
    use_video_description: bool = False,
) -> Path:
    """Run TOPReward over a dataset and write per-frame progress.

    Args:
        dataset_repo_id: Hugging Face dataset repo id or local path.
        reward_model_path: Optional TOPReward LeRobot config repo / dir to
            load (a tiny ``config.json``). When ``None`` (default), a
            fresh :class:`TOPRewardConfig` is constructed from the CLI
            overrides.
        vlm_name: Override the VLM backbone (HF Hub id).
        output_path: Where to write the parquet. Defaults to
            ``<dataset_root>/topreward_progress.parquet``.
        device: Device for the VLM.
        num_prefixes: Number of evenly-spaced anchor prefixes per episode.
            ``None`` (default) = fully dense (one VLM forward per frame).
            Set to ``15`` to match upstream TOPReward ``num_samples=15``.
        fps: Override the config's ``fps``.
        reduction: Override the config's ``reduction`` (``"mean"`` / ``"sum"``).
        use_video_description: Override the config's ``use_video_description``.

    Returns:
        Path to the written parquet file.
    """
    if reward_model_path is not None:
        logging.info(f"Loading TOPReward config from: {reward_model_path}")
        model = TOPRewardModel.from_pretrained(reward_model_path)
        config = model.config
        # Apply CLI overrides on top of the loaded config.
        if vlm_name is not None and vlm_name != config.vlm_name:
            logging.info(f"Overriding vlm_name from config: {config.vlm_name} -> {vlm_name}")
            # vlm_name affects the loaded weights; reload from scratch.
            config.vlm_name = vlm_name
            config.device = device
            model = TOPRewardModel(config)
    else:
        config_kwargs: dict[str, Any] = {"device": device}
        if vlm_name is not None:
            config_kwargs["vlm_name"] = vlm_name
        if fps is not None:
            config_kwargs["fps"] = fps
        if reduction is not None:
            config_kwargs["reduction"] = reduction
        if use_video_description:
            config_kwargs["use_video_description"] = True
        config = TOPRewardConfig(**config_kwargs)
        logging.info(f"Constructing TOPReward with VLM: {config.vlm_name}")
        model = TOPRewardModel(config)

    model.to(device).eval()

    image_key = config.image_key
    frames_key = f"{TOPREWARD_FEATURE_PREFIX}frames"
    task_batch_key = f"{TOPREWARD_FEATURE_PREFIX}task"

    logging.info(f"Loading dataset: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id, download_videos=True)
    logging.info(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    all_index: list[int] = []
    all_episode: list[int] = []
    all_frame: list[int] = []
    all_progress: list[float] = []

    for episode_idx in tqdm(range(dataset.num_episodes), desc="Episodes"):
        ep = dataset.meta.episodes[episode_idx]
        ep_start = int(ep["dataset_from_index"])
        ep_end = int(ep["dataset_to_index"])
        num_frames = ep_end - ep_start
        if num_frames <= 0:
            continue

        first_sample = dataset[ep_start]
        task = _resolve_task(first_sample, default=config.default_task or "perform the task")

        # Read the whole episode into one (N, C, H, W) tensor and convert
        # to (N, H, W, C) uint8 — same format ``TOPREWARD_FEATURE_PREFIX.frames``
        # expects. We deliberately bypass the encoder step here so its
        # ``max_frames`` tail-crop doesn't clip the prefix sweep.
        ep_video = torch.stack([dataset[ep_start + i][image_key] for i in range(num_frames)])
        ep_frames_uint8 = _frames_to_uint8_hwc(ep_video)

        batch = {frames_key: [ep_frames_uint8], task_batch_key: [task]}
        out = model.predict_curves(batch, num_prefixes=num_prefixes)
        per_frame = out["progress"][0, :num_frames].cpu().numpy()

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
            # Same column name SARM uses so RABCWeights + the overlay
            # script read TOPReward's output without per-model branching.
            "progress_sparse": np.asarray(all_progress, dtype=np.float32),
        }
    )

    # Persist provenance metadata: the LeRobot path (if any) and the VLM id.
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
    # Full RA-BC computation with the default Qwen3-VL-8B-Instruct backbone
    python -m lerobot.rewards.topreward.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image

    # Sparse-dense mode (matches upstream TOPReward num_samples=15)
    python -m lerobot.rewards.topreward.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image \\
        --num-prefixes 15

    # Use a smaller VLM
    python -m lerobot.rewards.topreward.compute_rabc_weights \\
        --dataset-repo-id lerobot/libero_10_image \\
        --vlm-name Qwen/Qwen3-VL-4B-Instruct
        """,
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repo id or local path.",
    )
    parser.add_argument(
        "--reward-model-path",
        type=str,
        default=None,
        help="Optional TOPReward LeRobot config (repo id or local dir). "
        "Falls back to a fresh TOPRewardConfig if unset.",
    )
    parser.add_argument(
        "--vlm-name",
        type=str,
        default=None,
        help="Override the VLM backbone (HF Hub id).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output parquet path. Defaults to <dataset_root>/topreward_progress.parquet.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda).",
    )
    parser.add_argument(
        "--num-prefixes",
        type=int,
        default=None,
        help="Evenly-spaced anchor prefixes per episode. None = fully dense "
        "(one VLM forward per frame). 15 matches upstream TOPReward.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override TOPRewardConfig.fps (frames per second for the Qwen video processor).",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default=None,
        choices=["mean", "sum"],
        help="Override TOPRewardConfig.reduction.",
    )
    parser.add_argument(
        "--use-video-description",
        action="store_true",
        help="Generate an instruction-agnostic video description and prepend "
        "it as context before scoring (doubles VLM calls per prefix).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the progress file to the dataset repo on HuggingFace Hub.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_path = compute_topreward_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=args.reward_model_path,
        vlm_name=args.vlm_name,
        output_path=args.output_path,
        device=args.device,
        num_prefixes=args.num_prefixes,
        fps=args.fps,
        reduction=args.reduction,
        use_video_description=args.use_video_description,
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
