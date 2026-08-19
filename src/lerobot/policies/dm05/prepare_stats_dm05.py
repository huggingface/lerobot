#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

"""Prepare missing DM05 state/action statistics in a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.utils import STATS_PATH, serialize_dict
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_dm05 import DM05Config
from .stats_validation_dm05 import (
    dm05_feature_stats_complete,
    dm05_stats_complete,
    validate_dm05_relative_action_stats,
)
from .utils import relative_action_mask

DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_SAMPLE_SEED = 0


def _summarize(values: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize one numeric array into the DM05 normalization statistics."""
    if not np.isfinite(values).all():
        raise ValueError("DM05 normalization samples contain non-finite values.")
    q01, q10, q90, q99 = np.quantile(values, (0.01, 0.10, 0.90, 0.99), axis=0).astype(np.float32)
    return {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0, dtype=np.float64).astype(np.float32),
        "std": values.std(axis=0, dtype=np.float64).astype(np.float32),
        "count": np.asarray([len(values)], dtype=np.int64),
        "q01": q01,
        "q10": q10,
        "q90": q90,
        "q99": q99,
    }


def compute_dm05_stats(
    config: DM05Config,
    dataset: LeRobotDataset,
    *,
    force: bool = False,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute missing state/action stats from deterministic valid action chunks."""
    config.validate_features()
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}.")
    existing = dict(dataset.meta.stats or {})
    if (
        config.use_relative_actions
        and not force
        and dm05_feature_stats_complete(config, existing, ACTION, "ACTION")
    ):
        raise ValueError(
            "Existing action stats do not record whether actions are absolute or relative. "
            "Re-run with --force to generate stats for --use-relative-actions."
        )
    if not force and dm05_stats_complete(config, existing):
        return existing

    numeric = dataset.select_columns([OBS_STATE, ACTION, "episode_index"])
    total_frames = len(numeric)
    episode_indices = np.asarray(numeric["episode_index"], dtype=np.int64).reshape(-1)
    episode_starts = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.flatnonzero(np.diff(episode_indices)) + 1]
    )
    episode_ends = np.concatenate([episode_starts[1:], np.asarray([total_frames], dtype=np.int64)])

    valid_lengths = np.maximum(episode_ends - episode_starts - config.drop_n_last_frames, 0)
    valid_cumulative = np.cumsum(valid_lengths)
    total_valid_starts = int(valid_cumulative[-1]) if len(valid_cumulative) else 0
    sample_count = min(sample_size, total_valid_starts)
    if sample_count == 0:
        raise ValueError("Cannot compute DM05 normalization stats from an empty training selection.")

    rng = np.random.default_rng(seed)
    sampled_offsets = np.sort(rng.choice(total_valid_starts, size=sample_count, replace=False))
    episode_ids = np.searchsorted(valid_cumulative, sampled_offsets, side="right")
    previous_cumulative = np.where(episode_ids == 0, 0, valid_cumulative[episode_ids - 1])
    starts = episode_starts[episode_ids] + sampled_offsets - previous_cumulative
    sampled_episode_ends = episode_ends[episode_ids]
    state_values = np.asarray(numeric.select(starts.tolist())[OBS_STATE], dtype=np.float32)

    action_indices = starts[:, None] + np.arange(config.chunk_size, dtype=np.int64)
    valid_actions = action_indices < sampled_episode_ends[:, None]
    flat_action_indices = action_indices[valid_actions]
    owners = np.broadcast_to(np.arange(sample_count)[:, None], action_indices.shape)[valid_actions]
    action_values = np.asarray(numeric.select(flat_action_indices.tolist())[ACTION], dtype=np.float32)

    if config.use_relative_actions:
        action_dim = action_values.shape[-1]
        if state_values.shape[-1] != action_dim:
            raise ValueError(
                "DM05 relative-action stats require equal state/action dimensions, got "
                f"{state_values.shape[-1]} and {action_dim}."
            )
        mask = np.asarray(
            relative_action_mask(
                action_dim,
                config.action_feature_names,
                config.relative_exclude_joints,
            ),
            dtype=np.float32,
        )
        action_values -= state_values[owners, :action_dim] * mask

    for key, feature_type, values in (
        (OBS_STATE, "STATE", state_values),
        (ACTION, "ACTION", action_values),
    ):
        if (
            force
            or (key == ACTION and config.use_relative_actions)
            or not dm05_feature_stats_complete(config, existing, key, feature_type)
        ):
            existing[key] = _summarize(values)

    validate_dm05_relative_action_stats(config, existing)

    logging.info(
        "Computed DM05 stats from %d/%d valid starts (seed=%d, action_values=%d).",
        sample_count,
        total_valid_starts,
        seed,
        len(action_values),
    )
    return existing


def _write_stats_atomic(stats: dict[str, Any], dataset_root: Path) -> Path:
    """Write ``meta/stats.json`` atomically under the dataset root."""
    stats_path = dataset_root / STATS_PATH
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = stats_path.with_suffix(f"{stats_path.suffix}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(serialize_dict(stats), indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, stats_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return stats_path


def prepare_dm05_stats(
    config: DM05Config,
    dataset: LeRobotDataset,
    *,
    force: bool = False,
) -> tuple[Path, bool]:
    """Compute and write ``meta/stats.json`` in the dataset's existing root."""
    was_complete = dm05_stats_complete(config, dataset.meta.stats)
    if was_complete and not force:
        if config.use_relative_actions:
            raise ValueError(
                "Existing stats do not record whether actions are absolute or relative. "
                "Re-run with --force to generate stats for --use-relative-actions."
            )
        return dataset.root / STATS_PATH, False
    stats = compute_dm05_stats(config, dataset, force=force)
    path = _write_stats_atomic(stats, dataset.root)
    dataset.meta.stats = stats
    return path, True


def _training_episodes(dataset: LeRobotDataset, eval_split: float) -> list[int] | None:
    """Select the DM05 training episodes after task-balanced eval splitting."""
    if eval_split == 0:
        return dataset.episodes
    if not 0 < eval_split < 1:
        raise ValueError(f"eval_split must be in [0, 1), got {eval_split}.")

    candidates = dataset.episodes or list(range(dataset.meta.total_episodes))
    episode_tasks = dataset.meta.episodes["tasks"]
    task_to_episodes: dict[str, list[int]] = {}
    for episode in candidates:
        task = episode_tasks[episode][0] if episode_tasks[episode] else ""
        task_to_episodes.setdefault(task, []).append(episode)

    selected: list[int] = []
    for task_episodes in task_to_episodes.values():
        n_eval = math.ceil(len(task_episodes) * eval_split)
        selected.extend(task_episodes[:-n_eval])
    if not selected:
        raise ValueError("eval_split leaves no training episodes for stats computation.")
    return selected


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for DM05 stats preparation."""
    parser = argparse.ArgumentParser(
        description="Generate missing DM05 state/action stats in a LeRobot dataset's meta/stats.json."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--episodes", nargs="*", type=int)
    parser.add_argument("--eval-split", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--drop-n-last-frames", type=int, default=1)
    parser.add_argument("--use-relative-actions", action="store_true")
    parser.add_argument("--relative-exclude-joints", nargs="*", default=["gripper"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the DM05 stats-preparation CLI."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        revision=args.revision,
        episodes=args.episodes,
    )
    selected_episodes = _training_episodes(dataset, args.eval_split)
    if selected_episodes != dataset.episodes:
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=dataset.root,
            revision=dataset.revision,
            episodes=selected_episodes,
        )

    config = DM05Config(
        chunk_size=args.chunk_size,
        drop_n_last_frames=args.drop_n_last_frames,
        use_relative_actions=args.use_relative_actions,
        relative_exclude_joints=args.relative_exclude_joints,
    )
    config.set_dataset_feature_metadata(dataset.meta.features)

    path, changed = prepare_dm05_stats(config, dataset, force=args.force)
    logging.info("%s %s", "Wrote" if changed else "Reused", path)


if __name__ == "__main__":
    main()
