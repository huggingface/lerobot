#!/usr/bin/env python
"""Shared helpers for the Project 4 space-bar progress-conditioned dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from huggingface_hub import hf_hub_download

SOURCE_REPO_ID = "Carsamba/so101_blind_task2"
DERIVED_DATASET_NAME = "so101_blind_task2_progress_v1"
PROGRESS_FEATURE_NAMES = ["progress", "one_minus_progress", "approach_vs_retract"]
EXPECTED_MOTOR_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def progress_vector(frame_index: int, episode_length: int) -> np.ndarray:
    """Return [progress, 1-progress, approach/retract] for a frame."""
    if episode_length <= 1:
        progress = 0.0
    else:
        progress = float(frame_index) / float(episode_length - 1)
    progress = float(np.clip(progress, 0.0, 1.0))
    phase = 1.0 if progress <= 0.5 else -1.0
    return np.asarray([progress, 1.0 - progress, phase], dtype=np.float32)


def progress_values_for_episode(episode_length: int) -> np.ndarray:
    return np.stack([progress_vector(i, episode_length) for i in range(episode_length)], axis=0)


def episode_lengths_from_metadata(episodes: Iterable[dict]) -> list[int]:
    lengths: list[int] = []
    for episode in episodes:
        if "length" in episode:
            lengths.append(int(episode["length"]))
        else:
            lengths.append(int(episode["dataset_to_index"]) - int(episode["dataset_from_index"]))
    return lengths


def compute_progress_stats(episode_lengths: list[int], stat_keys: Iterable[str] | None = None) -> dict:
    values = np.concatenate([progress_values_for_episode(length) for length in episode_lengths], axis=0)
    supported = {
        "mean": lambda x: x.mean(axis=0),
        "std": lambda x: x.std(axis=0),
        "min": lambda x: x.min(axis=0),
        "max": lambda x: x.max(axis=0),
        "q01": lambda x: np.quantile(x, 0.01, axis=0),
        "q99": lambda x: np.quantile(x, 0.99, axis=0),
    }
    keys = list(stat_keys) if stat_keys else ["mean", "std", "min", "max", "q01", "q99"]
    stats = {key: supported[key](values).astype(np.float32) for key in keys if key in supported}
    for required in ("mean", "std", "min", "max"):
        stats.setdefault(required, supported[required](values).astype(np.float32))
    return stats


def read_dataset_repo_id_from_train_config(policy_path: str | Path) -> str | None:
    train_config_path = Path(policy_path) / "train_config.json"
    if not train_config_path.exists():
        try:
            train_config_path = Path(hf_hub_download(repo_id=str(policy_path), filename="train_config.json"))
        except Exception:
            return None
    with open(train_config_path, encoding="utf-8") as f:
        train_cfg = json.load(f)
    repo_id = train_cfg.get("dataset", {}).get("repo_id")
    return str(repo_id) if repo_id else None
