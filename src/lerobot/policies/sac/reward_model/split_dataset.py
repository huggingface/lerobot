#!/usr/bin/env python
"""Split a LeRobotDataset into train / validation by frame stride.

The reward classifier processes frames independently, so a frame-level split
is sufficient and gives a denser validation set than an episode-level split.

Default ``stride=4``:
    frame index % 4 == 3 → val   (every 4th frame, 25%)
    everything else      → train (75%)

Episode boundaries from the source are preserved on both new datasets.

Usage:
    lerobot-split-reward-dataset --src-repo-id <repo>

Outputs (siblings of the source dataset):
    <src-root>-train
    <src-root>-val
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DONE, REWARD

logger = logging.getLogger(__name__)

_SKIP_KEYS = {"task_index", "timestamp", "episode_index", "frame_index", "index", "task"}


def _build_new_frame(frame: dict) -> dict:
    """Strip dataset-level metadata, reshape scalars, restore image channel-last uint8.

    LeRobotDataset reads image features as channel-first float32 in [0, 1]. The
    write path (add_frame) validates against the feature spec which expects
    channel-last uint8. Undo the read-time transform for image keys.
    """
    import torch

    new_frame: dict = {}
    for key, value in frame.items():
        if key in _SKIP_KEYS:
            continue
        if key in (DONE, REWARD):
            value = value.unsqueeze(0)
        if key.startswith("complementary_info") and hasattr(value, "dim") and value.dim() == 0:
            value = value.unsqueeze(0)
        if "image" in key and isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] in (1, 3):
            # (C, H, W) float32 in [0, 1] -> (H, W, C) uint8
            value = (value.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
        new_frame[key] = value
    return new_frame


def split_dataset(
    src_repo_id: str,
    src_root: str | None = None,
    train_repo_id: str | None = None,
    val_repo_id: str | None = None,
    val_stride: int = 4,
) -> tuple[LeRobotDataset, LeRobotDataset]:
    src = LeRobotDataset(repo_id=src_repo_id, root=src_root)

    if train_repo_id is None:
        train_repo_id = f"{src_repo_id}-train"
    if val_repo_id is None:
        val_repo_id = f"{src_repo_id}-val"

    train_root = Path(str(src.root) + "-train")
    val_root = Path(str(src.root) + "-val")

    common_kwargs = {
        "fps": int(src.fps),
        "robot_type": src.meta.robot_type,
        "features": src.meta.info["features"],
        "use_videos": len(src.meta.video_keys) > 0,
    }

    train_ds = LeRobotDataset.create(repo_id=train_repo_id, root=train_root, **common_kwargs)
    val_ds = LeRobotDataset.create(repo_id=val_repo_id, root=val_root, **common_kwargs)

    prev_episode_index = 0
    train_buffered = False
    val_buffered = False
    train_count = 0
    val_count = 0

    for frame_idx in tqdm(range(len(src)), desc="splitting frames"):
        frame = src[frame_idx]
        ep_idx = frame["episode_index"].item()

        if ep_idx != prev_episode_index:
            if train_buffered:
                train_ds.save_episode()
                train_buffered = False
            if val_buffered:
                val_ds.save_episode()
                val_buffered = False
            prev_episode_index = ep_idx

        new_frame = _build_new_frame(frame)
        new_frame["task"] = frame.get("task", "")

        if frame_idx % val_stride == val_stride - 1:
            val_ds.add_frame(new_frame)
            val_buffered = True
            val_count += 1
        else:
            train_ds.add_frame(new_frame)
            train_buffered = True
            train_count += 1

    if train_buffered:
        train_ds.save_episode()
    if val_buffered:
        val_ds.save_episode()

    train_ds.finalize()
    val_ds.finalize()

    print(
        f"\nSplit complete:\n"
        f"  source: {src_repo_id} ({len(src)} frames)\n"
        f"  train:  {train_repo_id} ({train_count} frames) at {train_root}\n"
        f"  val:    {val_repo_id}   ({val_count} frames) at {val_root}"
    )
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="Split a LeRobotDataset into train/val by frame stride.")
    parser.add_argument("--src-repo-id", type=str, required=True)
    parser.add_argument("--src-root", type=str, default=None)
    parser.add_argument("--train-repo-id", type=str, default=None)
    parser.add_argument("--val-repo-id", type=str, default=None)
    parser.add_argument("--val-stride", type=int, default=4)
    args = parser.parse_args()

    split_dataset(
        src_repo_id=args.src_repo_id,
        src_root=args.src_root,
        train_repo_id=args.train_repo_id,
        val_repo_id=args.val_repo_id,
        val_stride=args.val_stride,
    )


if __name__ == "__main__":
    main()
