#!/usr/bin/env python
"""Audit and plot the blind SO101 space-bar dataset trajectories."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from _spacebar_progress import (
    EXPECTED_MOTOR_NAMES,
    PROGRESS_FEATURE_NAMES,
    SOURCE_REPO_ID,
    episode_lengths_from_metadata,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=SOURCE_REPO_ID)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/project4_audit"))
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def load_arrays(dataset: LeRobotDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hf_dataset = dataset.hf_dataset.with_format(None)
    states = np.asarray(hf_dataset[OBS_STATE], dtype=np.float32)
    actions = np.asarray(hf_dataset[ACTION], dtype=np.float32)
    episode_indices = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(hf_dataset["frame_index"], dtype=np.int64)
    return states, actions, episode_indices, frame_indices


def summarize_consistency(
    states: np.ndarray,
    actions: np.ndarray,
    episode_indices: np.ndarray,
    episode_lengths: list[int],
) -> dict:
    summary: dict[str, float | int | list[int]] = {
        "min_episode_length": int(min(episode_lengths)),
        "max_episode_length": int(max(episode_lengths)),
        "unique_episode_lengths": sorted({int(length) for length in episode_lengths}),
    }
    if len(set(episode_lengths)) == 1:
        length = episode_lengths[0]
        state_stack = np.stack([states[episode_indices == ep][:length] for ep in range(len(episode_lengths))])
        action_stack = np.stack([actions[episode_indices == ep][:length] for ep in range(len(episode_lengths))])
        summary.update(
            {
                "mean_state_l2_std_across_demos": float(np.linalg.norm(state_stack.std(axis=0), axis=1).mean()),
                "mean_action_l2_std_across_demos": float(np.linalg.norm(action_stack.std(axis=0), axis=1).mean()),
                "mean_action_l2_step_delta": float(np.linalg.norm(np.diff(action_stack, axis=1), axis=2).mean()),
            }
        )
    return summary


def make_plots(
    output_dir: Path,
    states: np.ndarray,
    actions: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    names: list[str],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = output_dir / "spacebar_state_action_trajectories.png"
    fig, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 7.0), sharex=True)
    if len(names) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for dim, name in enumerate(names):
        for episode in sorted(set(episode_indices.tolist())):
            mask = episode_indices == episode
            x = frame_indices[mask]
            axes[0, dim].plot(x, states[mask, dim], alpha=0.55, linewidth=1.0)
            axes[1, dim].plot(x, actions[mask, dim], alpha=0.55, linewidth=1.0)
        axes[0, dim].set_title(name)
        axes[0, dim].set_ylabel("state" if dim == 0 else "")
        axes[1, dim].set_ylabel("action" if dim == 0 else "")
        axes[1, dim].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    episode_lengths = episode_lengths_from_metadata(dataset.meta.episodes)
    states, actions, episode_indices, frame_indices = load_arrays(dataset)

    state_names = dataset.meta.features[OBS_STATE]["names"]
    action_names = dataset.meta.features[ACTION]["names"]
    report = {
        "repo_id": dataset.repo_id,
        "root": str(dataset.root),
        "robot_type": dataset.meta.robot_type,
        "fps": dataset.meta.fps,
        "total_episodes": dataset.meta.total_episodes,
        "total_frames": dataset.meta.total_frames,
        "episode_lengths": episode_lengths,
        "state_names": state_names,
        "action_names": action_names,
        "has_progress_feature": OBS_ENV_STATE in dataset.meta.features,
        "progress_names": dataset.meta.features.get(OBS_ENV_STATE, {}).get("names"),
        "consistency": summarize_consistency(states, actions, episode_indices, episode_lengths),
    }
    report["checks"] = {
        "twenty_episodes": dataset.meta.total_episodes == 20,
        "six_thousand_frames": dataset.meta.total_frames == 6000,
        "three_hundred_frames_each": sorted(set(episode_lengths)) == [300],
        "thirty_fps": dataset.meta.fps == 30,
        "state_names_expected": state_names == EXPECTED_MOTOR_NAMES,
        "action_names_expected": action_names == EXPECTED_MOTOR_NAMES,
        "progress_names_expected": report["progress_names"] in (None, PROGRESS_FEATURE_NAMES),
    }

    report_path = args.output_dir / "spacebar_dataset_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote audit report: %s", report_path)

    if not args.skip_plots:
        plot_path = make_plots(args.output_dir, states, actions, episode_indices, frame_indices, state_names)
        logger.info("Wrote trajectory plot: %s", plot_path)

    failed = [key for key, ok in report["checks"].items() if not ok]
    if failed:
        logger.warning("Audit checks needing attention: %s", failed)
    else:
        logger.info("All headline audit checks passed.")


if __name__ == "__main__":
    main()
