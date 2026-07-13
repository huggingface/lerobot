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

"""Compute per-frame ``is_terminal`` and ``mc_return`` for a LeRobot dataset.

Implements the sparse reward function from pi*0.6 / RECAP (Eq. 5):

    r_t = -1          for non-terminal steps
    r_T = 0           for terminal success
    r_T = -C_fail     for terminal failure

Monte Carlo returns are the cumulative sum from each step to the end of
the episode, normalized by ``max_episode_length`` so that values are bounded
to approximately (-1, 0).

The columns are written directly into the dataset's parquet data shards as
flat per-frame scalars. These serve as training targets for the distributional
value function.

Usage:
    # Compute returns using the default "next.success" column (from lerobot-eval/rollout)
    lerobot-compute-returns \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human_image

    # Override: treat all episodes as successful (demo-only datasets)
    lerobot-compute-returns \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human_image \\
        --default-success true

    # Custom success key, failure penalty, and discount
    lerobot-compute-returns \\
        --dataset-repo-id my_org/my_dataset \\
        --success-key episode_success \\
        --c-fail 100 \\
        --gamma 0.99
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logger = logging.getLogger(__name__)

IS_TERMINAL_COL = "is_terminal"
MC_RETURN_COL = "mc_return"


@dataclass
class ComputeReturnsConfig:
    """Configuration for the returns computation script."""

    dataset_repo_id: str = ""
    root: str | None = None
    success_key: str = "next.success"
    default_success: bool | None = None
    max_episode_length: int | None = None
    c_fail: float = 50.0
    gamma: float = 1.0
    episodes: list[int] = field(default_factory=list)
    force: bool = False
    push_to_hub: bool = False


def _get_episode_success(
    episode_table: pa.Table,
    success_key: str,
    default_success: bool | None,
) -> bool:
    """Determine whether an episode was successful.

    Priority:
    1. If ``default_success`` is set, use it unconditionally.
    2. Look for ``success_key`` in the parquet columns and reduce with any().
    3. Fall back to True (assume success for demo datasets).
    """
    if default_success is not None:
        return default_success

    if success_key in episode_table.column_names:
        col = episode_table.column(success_key)
        for val in col:
            py_val = val.as_py()
            if isinstance(py_val, bool) and py_val:
                return True
            if isinstance(py_val, (int, float)) and py_val:
                return True
        return False

    return True


def compute_episode_returns(
    num_frames: int,
    success: bool,
    c_fail: float,
    gamma: float,
    max_episode_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute is_terminal and mc_return arrays for a single episode.

    Args:
        num_frames: Number of frames in the episode.
        success: Whether the episode ended successfully.
        c_fail: Failure penalty constant.
        gamma: Discount factor (1.0 = undiscounted).
        max_episode_length: Normalization horizon H.

    Returns:
        Tuple of (is_terminal, mc_return) arrays, each of length num_frames.
    """
    horizon = max_episode_length

    rewards = np.full(num_frames, -1.0 / horizon, dtype=np.float64)

    if success:
        rewards[-1] = 0.0
    else:
        rewards[-1] = -c_fail / horizon

    is_terminal = np.zeros(num_frames, dtype=bool)
    is_terminal[-1] = True

    if gamma == 1.0:
        # Reverse cumulative sum
        mc_return = np.cumsum(rewards[::-1])[::-1].astype(np.float32)
    else:
        mc_return = np.zeros(num_frames, dtype=np.float64)
        mc_return[-1] = rewards[-1]
        for t in range(num_frames - 2, -1, -1):
            mc_return[t] = rewards[t] + gamma * mc_return[t + 1]
        mc_return = mc_return.astype(np.float32)

    return is_terminal, mc_return


def compute_returns(config: ComputeReturnsConfig) -> Path:
    """Compute returns and write them into parquet shards."""
    from lerobot.datasets import LeRobotDataset

    logger.info(f"Loading dataset: {config.dataset_repo_id}")
    kwargs = {"repo_id": config.dataset_repo_id, "download_videos": False}
    if config.root:
        kwargs["root"] = config.root
    dataset = LeRobotDataset(**kwargs)

    meta = dataset.meta
    root = Path(meta.root)
    logger.info(f"Dataset root: {root}")
    logger.info(f"Episodes: {meta.total_episodes}, Frames: {meta.total_frames}")

    episode_indices = config.episodes if config.episodes else list(range(meta.total_episodes))

    if config.max_episode_length is not None:
        max_ep_len = config.max_episode_length
    else:
        max_ep_len = max(int(meta.episodes[i]["length"]) for i in episode_indices)
    logger.info(f"Normalization horizon (max_episode_length): {max_ep_len}")

    parquet_files_to_rewrite: dict[Path, list[int]] = {}
    for ep_idx in episode_indices:
        rel_path = meta.get_data_file_path(ep_idx)
        abs_path = root / rel_path
        parquet_files_to_rewrite.setdefault(abs_path, []).append(ep_idx)

    logger.info(f"Parquet shards to rewrite: {len(parquet_files_to_rewrite)}")

    for parquet_path, ep_indices_in_file in tqdm(parquet_files_to_rewrite.items(), desc="Processing shards"):
        table = pq.read_table(parquet_path)

        if not config.force and IS_TERMINAL_COL in table.column_names:
            logger.info(f"Skipping {parquet_path.name} (already has {IS_TERMINAL_COL})")
            continue

        all_is_terminal = np.zeros(len(table), dtype=bool)
        all_mc_return = np.zeros(len(table), dtype=np.float32)

        episode_col = table.column("episode_index").to_pylist()

        for ep_idx in ep_indices_in_file:
            ep_info = meta.episodes[ep_idx]
            ep_from = int(ep_info["dataset_from_index"])
            ep_to = int(ep_info["dataset_to_index"])
            ep_len = ep_to - ep_from

            mask = np.array([v == ep_idx for v in episode_col], dtype=bool)
            local_indices = np.where(mask)[0]

            if len(local_indices) != ep_len:
                logger.warning(
                    f"Episode {ep_idx}: expected {ep_len} frames in shard, "
                    f"found {len(local_indices)}. Using found count."
                )
                ep_len = len(local_indices)

            if ep_len == 0:
                continue

            ep_subtable = table.filter(mask)
            success = _get_episode_success(ep_subtable, config.success_key, config.default_success)

            is_terminal, mc_return = compute_episode_returns(
                num_frames=ep_len,
                success=success,
                c_fail=config.c_fail,
                gamma=config.gamma,
                max_episode_length=max_ep_len,
            )

            all_is_terminal[local_indices] = is_terminal
            all_mc_return[local_indices] = mc_return

        if IS_TERMINAL_COL in table.column_names:
            table = table.drop(IS_TERMINAL_COL)
        if MC_RETURN_COL in table.column_names:
            table = table.drop(MC_RETURN_COL)

        table = table.append_column(IS_TERMINAL_COL, pa.array(all_is_terminal))
        table = table.append_column(MC_RETURN_COL, pa.array(all_mc_return))

        pq.write_table(table, parquet_path)

    _update_info_json(root, meta)

    logger.info("Done. Columns written: is_terminal, mc_return")

    if config.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        logger.info(f"Pushing updated dataset to Hub: {config.dataset_repo_id}")
        api.upload_folder(
            folder_path=str(root),
            repo_id=config.dataset_repo_id,
            repo_type="dataset",
        )
        logger.info("Push to Hub complete.")

    return root


def _update_info_json(root: Path, meta) -> None:
    """Add is_terminal and mc_return to the dataset's info.json features."""
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        logger.warning(f"info.json not found at {info_path}, skipping metadata update.")
        return

    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    changed = False

    if IS_TERMINAL_COL not in features:
        features[IS_TERMINAL_COL] = {
            "dtype": "bool",
            "shape": [1],
            "names": None,
        }
        changed = True

    if MC_RETURN_COL not in features:
        features[MC_RETURN_COL] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        changed = True

    if changed:
        info["features"] = features
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        logger.info("Updated meta/info.json with is_terminal and mc_return features.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-frame is_terminal and mc_return for a LeRobot dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use the 'success' column from the dataset
    lerobot-compute-returns --dataset-repo-id lerobot/aloha_sim_insertion_human_image

    # Override all episodes as successful (demo-only data)
    lerobot-compute-returns --dataset-repo-id my_org/my_dataset --default-success true

    # Custom failure penalty
    lerobot-compute-returns --dataset-repo-id my_org/my_dataset --c-fail 100
        """,
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repo id or local path.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local root directory override for the dataset.",
    )
    parser.add_argument(
        "--success-key",
        type=str,
        default="next.success",
        help="Column name in parquet that indicates episode success (default: 'next.success').",
    )
    parser.add_argument(
        "--default-success",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Override success for all episodes ('true' or 'false').",
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=None,
        help="Normalization horizon H. If not set, uses max episode length in dataset.",
    )
    parser.add_argument(
        "--c-fail",
        type=float,
        default=50.0,
        help="Failure penalty constant (default: 50.0).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount factor (default: 1.0, undiscounted).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Process only these episode indices (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing is_terminal/mc_return columns.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the updated dataset to the Hugging Face Hub after computing returns.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    default_success = None
    if args.default_success is not None:
        default_success = args.default_success.lower() == "true"

    config = ComputeReturnsConfig(
        dataset_repo_id=args.dataset_repo_id,
        root=args.root,
        success_key=args.success_key,
        default_success=default_success,
        max_episode_length=args.max_episode_length,
        c_fail=args.c_fail,
        gamma=args.gamma,
        episodes=args.episodes or [],
        force=args.force,
        push_to_hub=args.push_to_hub,
    )

    root = compute_returns(config)
    logger.info(f"Returns computed and written to: {root}")
    logger.info(f"  Columns added: {IS_TERMINAL_COL}, {MC_RETURN_COL}")
    logger.info("To train the distributional value function, these columns")
    logger.info("will be read as flat batch keys during training.")


if __name__ == "__main__":
    main()
