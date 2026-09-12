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

"""
Compute per-episode quality metrics for a LeRobotDataset and flag outlier episodes.

For datasets with more than a handful of episodes, reviewing each one in `lerobot-dataset-viz` is
impractical. This tool computes deterministic metrics from the recorded actions of every episode:

- `n_frames` / `duration_s`: episode length
- `median_jerk` / `p95_jerk`: trajectory smoothness (second derivative of the actions)
- `max_velocity`: peak action-space velocity
- `static_fraction`: fraction of near-motionless frames (hesitations, dead time)
- `final_action`: last action of the episode (end-pose consistency across episodes)

Episodes that are outliers on any metric (IQR rule) are flagged and ranked, producing a short list of
candidates to inspect in `lerobot-dataset-viz` and optionally remove with `lerobot-edit-dataset`.

Usage examples:

Analyze a dataset from the Hugging Face hub (or local cache):
```shell
lerobot-dataset-quality --repo-id=user/my_dataset
```

Analyze a local dataset and emit machine-readable JSON:
```shell
lerobot-dataset-quality --repo-id=user/my_dataset --root=/path/to/dataset --output-format=json
```

Use a stricter outlier rule and list more episodes:
```shell
lerobot-dataset-quality --repo-id=user/my_dataset --k-iqr=1.0 --top-bad=20
```
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging


def group_actions_by_episode(hf_dataset) -> dict[int, np.ndarray]:
    """Group action arrays by episode from a LeRobotDataset's underlying hf_dataset."""
    columns = hf_dataset.select_columns(["episode_index", "frame_index", ACTION])
    rows_by_ep: dict[int, list[tuple[int, np.ndarray]]] = {}
    for row in columns:
        ep = int(row["episode_index"])
        rows_by_ep.setdefault(ep, []).append(
            (int(row["frame_index"]), np.asarray(row[ACTION], dtype=np.float64))
        )
    actions_by_ep = {}
    for ep, rows in rows_by_ep.items():
        rows.sort(key=lambda r: r[0])
        actions_by_ep[ep] = np.stack([r[1] for r in rows])
    return actions_by_ep


def compute_episode_metrics(actions: np.ndarray, episode_index: int, fps: float) -> dict:
    """Compute deterministic quality metrics from one episode's actions array of shape (T, action_dim)."""
    n_frames = len(actions)
    metrics = {
        "episode": episode_index,
        "n_frames": int(n_frames),
        "duration_s": round(n_frames / fps, 2),
        "median_jerk": 0.0,
        "p95_jerk": 0.0,
        "max_velocity": 0.0,
        "static_fraction": 0.0,
        "final_action": actions[-1].tolist() if n_frames > 0 else [],
    }
    if n_frames < 3:
        return metrics

    velocities = np.diff(actions, axis=0)  # (T-1, action_dim)
    jerk = np.diff(velocities, axis=0)  # (T-2, action_dim)

    # Median is robust to isolated spikes from quick corrections; p95 captures those spikes instead.
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    metrics["median_jerk"] = round(float(np.median(jerk_magnitude)), 5)
    metrics["p95_jerk"] = round(float(np.percentile(jerk_magnitude, 95)), 5)

    velocity_magnitude = np.linalg.norm(velocities, axis=1)
    metrics["max_velocity"] = round(float(np.max(velocity_magnitude)), 4)

    # A frame is "static" when it moves less than 5% of the median active velocity.
    active_vels = velocity_magnitude[velocity_magnitude > 1e-6]
    if len(active_vels) > 0:
        static_threshold = 0.05 * np.median(active_vels)
        metrics["static_fraction"] = round(float(np.mean(velocity_magnitude < static_threshold)), 3)
    else:
        metrics["static_fraction"] = 1.0

    return metrics


def detect_outliers(metrics: list[dict], k_iqr: float = 1.5) -> dict[int, set[str]]:
    """Flag episodes that are outliers on any metric using the IQR rule.

    Returns a dict mapping episode index to the set of flags raised for it (e.g. "duration_high").
    """
    outliers: dict[int, set[str]] = defaultdict(set)

    def iqr_outliers(values: list[float], episodes: list[int], name: str, direction: str = "both") -> None:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k_iqr * iqr, q3 + k_iqr * iqr
        for v, ep in zip(values, episodes, strict=True):
            if direction in ("both", "high") and v > hi:
                outliers[ep].add(f"{name}_high")
            if direction in ("both", "low") and v < lo:
                outliers[ep].add(f"{name}_low")

    eps = [m["episode"] for m in metrics]
    iqr_outliers([m["n_frames"] for m in metrics], eps, "duration")
    iqr_outliers([m["median_jerk"] for m in metrics], eps, "jerk", "high")
    iqr_outliers([m["p95_jerk"] for m in metrics], eps, "spike_jerk", "high")
    iqr_outliers([m["max_velocity"] for m in metrics], eps, "max_vel", "high")
    iqr_outliers([m["static_fraction"] for m in metrics], eps, "static", "high")

    # End-pose consistency: distance of each episode's final action from the across-episode mean.
    finals = [m["final_action"] for m in metrics if m["final_action"]]
    final_eps = [m["episode"] for m in metrics if m["final_action"]]
    if finals:
        finals_arr = np.array(finals)
        distances = np.linalg.norm(finals_arr - np.mean(finals_arr, axis=0), axis=1)
        iqr_outliers(distances.tolist(), final_eps, "final_state", "high")

    return dict(outliers)


def evaluate_dataset_quality(dataset: LeRobotDataset, k_iqr: float = 1.5) -> dict:
    """Run the full quality analysis: per-episode metrics plus IQR outlier flags."""
    if ACTION not in dataset.meta.features:
        raise ValueError(f"Dataset must contain an '{ACTION}' feature. Found: {list(dataset.meta.features)}")

    actions_by_ep = group_actions_by_episode(dataset.hf_dataset)
    metrics = [compute_episode_metrics(actions_by_ep[ep], ep, dataset.fps) for ep in sorted(actions_by_ep)]
    outliers = detect_outliers(metrics, k_iqr=k_iqr)
    return {"metrics": metrics, "outliers": {ep: sorted(flags) for ep, flags in outliers.items()}}


def print_table_report(repo_id: str, report: dict, top_bad: int = 10) -> None:
    """Print a human-readable summary of the quality analysis."""
    metrics = report["metrics"]
    outliers = report["outliers"]
    n = len(metrics)

    print(f"\nEpisode quality report — {repo_id} ({n} episodes)\n")

    durations = [m["duration_s"] for m in metrics]
    n_frames = [m["n_frames"] for m in metrics]
    jerks = [m["median_jerk"] for m in metrics]
    velocities = [m["max_velocity"] for m in metrics]
    statics = [m["static_fraction"] for m in metrics]

    print("Aggregate stats (mean ± std, min–max):")
    print(
        f"  Duration         : {np.mean(durations):.1f} ± {np.std(durations):.1f}s "
        f"({np.min(durations):.1f}–{np.max(durations):.1f}s)"
    )
    print(
        f"  Frames           : {np.mean(n_frames):.0f} ± {np.std(n_frames):.0f} "
        f"({np.min(n_frames)}–{np.max(n_frames)})"
    )
    print(f"  Median jerk      : {np.mean(jerks):.4f} ± {np.std(jerks):.4f}")
    print(f"  Max velocity     : {np.mean(velocities):.3f} ± {np.std(velocities):.3f}")
    print(f"  Static fraction  : {np.mean(statics) * 100:.1f}% ± {np.std(statics) * 100:.1f}%")
    print()

    flag_counts: dict[str, int] = defaultdict(int)
    for flags in outliers.values():
        for flag in flags:
            flag_counts[flag] += 1
    print(f"Outlier flags ({len(outliers)} episodes flagged at least once):")
    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        print(f"  {flag:25s}: {count} episodes")
    print()

    ranked = sorted(outliers.items(), key=lambda x: -len(x[1]))[:top_bad]
    if ranked:
        print(f"Top {len(ranked)} worst episodes (most flags):")
        print(f"  {'ep':>4} | {'#flags':>6} | flags")
        print(f"  {'-' * 4} | {'-' * 6} | {'-' * 60}")
        for ep, flags in ranked:
            print(f"  {ep:>4} | {len(flags):>6} | {', '.join(flags)}")
        print()

        bad_eps = sorted(ep for ep, _ in ranked)
        print("Suggested next step — review these in lerobot-dataset-viz, then optionally delete:")
        print(f"  lerobot-dataset-viz --repo-id {repo_id} --episode-index {bad_eps[0]}")
        print("  lerobot-edit-dataset \\")
        print(f"    --repo_id {repo_id} \\")
        print(f"    --new_repo_id {repo_id}_clean \\")
        print("    --operation.type delete_episodes \\")
        print(f'    --operation.episode_indices "{bad_eps}"')
    else:
        print("No outlier episodes flagged — dataset looks consistent.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-episode quality metrics for a LeRobotDataset and flag outlier episodes."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of the Hugging Face repository containing a LeRobotDataset (e.g. `user/my_dataset`).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory of a dataset stored locally. By default, the dataset is loaded from the "
        "Hugging Face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--top-bad",
        type=int,
        default=10,
        help="How many of the worst episodes to list in the table report (default: 10).",
    )
    parser.add_argument(
        "--k-iqr",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection; lower is stricter (default: 1.5).",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["table", "json"],
        default="table",
        help="Print a human-readable table (default) or machine-readable JSON.",
    )
    args = parser.parse_args()

    init_logging()
    logging.info(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(args.repo_id, root=args.root)

    report = evaluate_dataset_quality(dataset, k_iqr=args.k_iqr)

    if args.output_format == "json":
        print(json.dumps({"repo_id": args.repo_id, **report}, indent=2))
    else:
        print_table_report(args.repo_id, report, top_bad=args.top_bad)


if __name__ == "__main__":
    main()
