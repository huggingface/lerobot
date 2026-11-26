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

"""Per-episode quality metrics and outlier flagging for recorded datasets.

Imitation learning quality depends heavily on demonstration quality, but once a dataset grows
past a few dozen episodes, reviewing each one in `lerobot-dataset-viz` is impractical. Bad
episodes — struggled attempts that ran long, sharp corrective jerks, long hesitations, episodes
that end in the wrong pose — stay in the training set simply because nobody had time to find
them.

This read-only analysis CLI computes deterministic per-episode metrics from the `action` column
only (no image/video decoding) and flags statistical outliers with the IQR rule, so that
"watch 200 episodes" becomes "watch these 8":

1. `duration_s` — episode duration in seconds (num_frames / fps).
2. `median_jerk`, `p95_jerk` — trajectory smoothness, as the norm of the third finite
   difference of the action trajectory.
3. `peak_velocity` — largest per-frame action-space speed.
4. `motionless_fraction` — fraction of frames whose action-space speed is below 1% of the
   dataset-wide peak speed (long hesitations).
5. `final_pose_distance` — distance between the episode's final action and the across-episode
   mean final action (episodes that end in the wrong pose).

The intended workflow is: `lerobot-dataset-quality` → inspect the short list in
`lerobot-dataset-viz` → delete with `lerobot-edit-dataset`. The tool itself never modifies the
dataset.

Usage:

```shell
lerobot-dataset-quality --repo-id lerobot/pusht
lerobot-dataset-quality --repo-id lerobot/pusht --root /path/to/pusht --k-iqr 2.0
lerobot-dataset-quality --repo-id lerobot/pusht --output-format json
```
"""

import argparse
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

# Metrics used for IQR outlier flagging, in report order.
METRICS = (
    "duration_s",
    "median_jerk",
    "p95_jerk",
    "peak_velocity",
    "motionless_fraction",
    "final_pose_distance",
)

# A frame counts as near-motionless when its action-space speed is below this fraction of the
# dataset-wide peak speed. Relative (not absolute) so the metric is unit-agnostic.
DEFAULT_MOTIONLESS_FRACTION = 0.01

# Minimum number of frames needed to compute the third finite difference.
MIN_FRAMES_FOR_JERK = 4


@dataclass(frozen=True)
class EpisodeMetrics:
    """Deterministic per-episode quality metrics computed from the action column."""

    num_frames: int
    duration_s: float
    median_jerk: float
    p95_jerk: float
    peak_velocity: float
    motionless_fraction: float
    final_pose_distance: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def load_episode_actions(dataset: LeRobotDataset) -> dict[int, np.ndarray]:
    """Load the action trajectory of every episode, without decoding images or videos.

    Args:
        dataset: The LeRobot dataset to read from.

    Returns:
        Mapping of episode index to its action array of shape (num_frames, action_dim).

    Raises:
        ValueError: If the dataset has no `action` feature.
        AttributeError: If the dataset's storage format has no HF table (e.g. Lance).
    """
    if "action" not in dataset.features:
        raise ValueError(
            f"Dataset {dataset.repo_id!r} has no 'action' feature "
            f"(found: {sorted(dataset.features)}); this tool only requires an action feature."
        )
    actions_table = dataset.hf_dataset.select_columns(["action"])
    episode_actions: dict[int, np.ndarray] = {}
    for ep_idx in range(dataset.num_episodes):
        start = dataset.meta.episodes[ep_idx]["dataset_from_index"]
        end = dataset.meta.episodes[ep_idx]["dataset_to_index"]
        actions = np.asarray(actions_table[start:end]["action"], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[:, None]
        episode_actions[ep_idx] = actions
    return episode_actions


def _speeds(actions: np.ndarray) -> np.ndarray:
    """Per-frame action-space speed (norm of the first finite difference), shape (T - 1,)."""
    return np.linalg.norm(np.diff(actions, axis=0), axis=1)


def compute_episode_metrics(
    actions: np.ndarray,
    fps: int,
    motionless_threshold: float,
    mean_final_pose: np.ndarray,
) -> EpisodeMetrics:
    """Compute the quality metrics of a single episode.

    Args:
        actions: Action trajectory of shape (num_frames, action_dim).
        fps: Dataset recording frequency, used to convert frame counts to seconds.
        motionless_threshold: Speed below which a frame counts as near-motionless.
        mean_final_pose: Across-episode mean of the final action, shape (action_dim,).

    Returns:
        The episode's metrics. Jerk metrics are NaN for episodes shorter than
        `MIN_FRAMES_FOR_JERK` frames.
    """
    num_frames = actions.shape[0]
    speeds = _speeds(actions) if num_frames >= 2 else np.zeros(0)

    if num_frames >= MIN_FRAMES_FOR_JERK:
        jerks = np.linalg.norm(np.diff(actions, n=3, axis=0), axis=1)
        median_jerk = float(np.median(jerks))
        p95_jerk = float(np.percentile(jerks, 95))
    else:
        median_jerk = math.nan
        p95_jerk = math.nan

    motionless_fraction = float(np.mean(speeds <= motionless_threshold)) if num_frames >= 2 else 0.0

    final_pose_distance = float(np.linalg.norm(actions[-1] - mean_final_pose))

    return EpisodeMetrics(
        num_frames=num_frames,
        duration_s=num_frames / fps,
        median_jerk=median_jerk,
        p95_jerk=p95_jerk,
        peak_velocity=float(np.max(speeds)) if speeds.size else 0.0,
        motionless_fraction=motionless_fraction,
        final_pose_distance=final_pose_distance,
    )


def analyze_actions(
    actions_by_episode: Mapping[int, np.ndarray],
    fps: int,
    motionless_fraction: float = DEFAULT_MOTIONLESS_FRACTION,
) -> dict[int, EpisodeMetrics]:
    """Compute per-episode quality metrics for a whole dataset.

    The near-motionless threshold is derived from the dataset-wide peak speed so that the
    metric stays meaningful across robots with different action units, and the final-pose
    reference is the across-episode mean final action.

    Args:
        actions_by_episode: Mapping of episode index to action array (num_frames, action_dim).
        fps: Dataset recording frequency.
        motionless_fraction: Fraction of the dataset-wide peak speed below which a frame
            counts as near-motionless.

    Returns:
        Mapping of episode index to its metrics.

    Raises:
        ValueError: If there are no episodes.
    """
    if not actions_by_episode:
        raise ValueError("No episodes to analyze")

    episode_speeds = [_speeds(a) for a in actions_by_episode.values() if a.shape[0] >= 2]
    peak_speed = max((float(np.max(s)) for s in episode_speeds), default=0.0)
    motionless_threshold = motionless_fraction * peak_speed if peak_speed > 0 else 0.0

    final_poses = np.stack([actions[-1] for actions in actions_by_episode.values()])
    mean_final_pose = final_poses.mean(axis=0)

    return {
        ep_idx: compute_episode_metrics(actions, fps, motionless_threshold, mean_final_pose)
        for ep_idx, actions in actions_by_episode.items()
    }


def flag_outliers(values: Mapping[int, float], k: float) -> set[int]:
    """Flag episodes that are IQR outliers on one metric.

    Episodes with NaN values are ignored (not flagged). When the interquartile range is zero
    — all episodes identical on that metric — nothing is flagged.

    Args:
        values: Mapping of episode index to the metric value.
        k: IQR multiplier (e.g. 1.5 for the standard rule, larger for fewer flags).

    Returns:
        Set of episode indices outside [Q1 - k * IQR, Q3 + k * IQR].
    """
    finite = [v for v in values.values() if math.isfinite(v)]
    if len(finite) < 2:
        return set()
    q1, q3 = np.percentile(finite, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return set()
    low, high = q1 - k * iqr, q3 + k * iqr
    return {ep for ep, v in values.items() if math.isfinite(v) and (v < low or v > high)}


def rank_flagged(
    metrics: Mapping[int, EpisodeMetrics],
    flagged_by_metric: Mapping[str, set[int]],
) -> list[tuple[int, list[str]]]:
    """Rank flagged episodes by number of outlier metrics, worst first.

    Args:
        metrics: Per-episode metrics.
        flagged_by_metric: Mapping of metric name to the episodes flagged on it.

    Returns:
        List of (episode index, offending metric names), sorted by flag count descending and
        episode index ascending for determinism.
    """
    flags_per_episode: dict[int, list[str]] = {}
    for metric_name, episodes in flagged_by_metric.items():
        for ep in episodes:
            flags_per_episode.setdefault(ep, []).append(metric_name)
    return sorted(
        ((ep, sorted(names)) for ep, names in flags_per_episode.items()),
        key=lambda item: (-len(item[1]), item[0]),
    )


def build_delete_command(repo_id: str, episodes: list[int], root: str | None) -> str:
    """Build the ready-to-paste `lerobot-edit-dataset` command removing the given episodes."""
    indices = ", ".join(str(ep) for ep in episodes)
    command = (
        f"lerobot-edit-dataset --repo_id {repo_id} "
        f'--operation.type delete_episodes --operation.episode_indices "[{indices}]"'
    )
    if root is not None:
        command += f" --root {root}"
    return command


def _format_table(
    repo_id: str,
    fps: int,
    k_iqr: float,
    metrics: Mapping[int, EpisodeMetrics],
    ranked: list[tuple[int, list[str]]],
) -> str:
    """Render the human-readable report: flagged episodes, worst first."""
    lines = [
        f"Dataset: {repo_id} (fps={fps}, k_iqr={k_iqr})",
        f"Episodes: {len(metrics)} total, {len(ranked)} flagged",
    ]
    if ranked:
        header = f"{'episode':>8}  {'flags':>5}  {'duration_s':>10}  {'median_jerk':>11}  "
        header += f"{'p95_jerk':>8}  {'peak_velocity':>13}  {'motionless':>10}  {'final_pose':>10}"
        lines.append("")
        lines.append(header)
        for ep, names in ranked:
            m = metrics[ep]
            row = (
                f"{ep:>8}  {len(names):>5}  {m.duration_s:>10.3f}  {m.median_jerk:>11.3g}  "
                f"{m.p95_jerk:>8.3g}  {m.peak_velocity:>13.3g}  {m.motionless_fraction:>10.3f}  "
                f"{m.final_pose_distance:>10.3g}"
            )
            lines.append(row)
            lines.append(f"{'':>8}  flagged: {', '.join(names)}")
    else:
        lines.append("No outlier episodes detected.")
    return "\n".join(lines)


def _json_safe(value: float) -> float | None:
    """Convert non-finite floats (NaN) to None so the output is strict JSON."""
    return value if math.isfinite(value) else None


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Per-episode quality metrics and IQR outlier flagging for recorded datasets "
            "(read-only; only the action column is read)."
        )
    )
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repository id on the Hub.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local dataset root. Defaults to $HF_LEROBOT_HOME/{repo_id}.",
    )
    parser.add_argument(
        "--k-iqr",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier flagging (default: 1.5).",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="table",
        choices=("table", "json"),
        help="Report format (default: table).",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root, download_videos=False)
    logging.info(
        f"Loaded {dataset.repo_id}: {dataset.num_episodes} episodes, {dataset.meta.total_frames} frames"
    )
    actions_by_episode = load_episode_actions(dataset)
    metrics = analyze_actions(actions_by_episode, dataset.meta.fps)

    flagged_by_metric = {
        name: flag_outliers({ep: getattr(m, name) for ep, m in metrics.items()}, args.k_iqr)
        for name in METRICS
    }
    ranked = rank_flagged(metrics, flagged_by_metric)

    if ranked:
        flagged_episodes = [ep for ep, _ in ranked]
        suggested_command = build_delete_command(
            args.repo_id, sorted(flagged_episodes), str(args.root) if args.root else None
        )
    else:
        suggested_command = None

    if args.output_format == "json":
        report = {
            "repo_id": args.repo_id,
            "fps": dataset.meta.fps,
            "k_iqr": args.k_iqr,
            "num_episodes": len(metrics),
            "metrics": {
                str(ep): {
                    name: _json_safe(v) if isinstance(v, float) else v for name, v in m.as_dict().items()
                }
                for ep, m in sorted(metrics.items())
            },
            "flagged_episodes": {str(ep): names for ep, names in ranked},
            "suggested_command": suggested_command,
        }
        print(json.dumps(report, indent=2))
    else:
        output = _format_table(args.repo_id, dataset.meta.fps, args.k_iqr, metrics, ranked)
        if suggested_command:
            output += "\n\nReview flagged episodes in lerobot-dataset-viz, then delete with:\n"
            output += f"  {suggested_command}"
        print(output)


if __name__ == "__main__":
    main()
