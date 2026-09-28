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
Detect leader/follower calibration drift in a recorded teleoperation dataset.

In leader/follower teleoperation, `action` holds the leader position and `observation.state` holds the
follower position at the same timestep. On frames where the robot is not moving ("stable" frames), the
follower controller has had time to converge, so `action - observation.state` should be close to zero.
A systematic non-zero mean on a joint reveals a calibration offset between the two arms — an error that
is invisible during training (the policy simply learns the biased mapping) but breaks deployment, since
inference feeds follower observations to a policy trained on leader-shifted targets.

This tool computes, per motor, the mean and standard deviation of `action - observation.state` restricted
to stable frames (frames where every joint moves less than `--vel-threshold` units per step).

Note: deltas are expressed in the dataset's native action units (degrees for most Feetech/Dynamixel-based
arms such as SO-100/SO-101). Adjust `--ok-threshold` / `--warn-threshold` if your dataset uses other units.

Usage examples:

Check a dataset from the Hugging Face hub (or local cache):
```shell
lerobot-check-calibration --repo-id=user/my_dataset
```

Check a local dataset and emit machine-readable JSON:
```shell
lerobot-check-calibration --repo-id=user/my_dataset --root=/path/to/dataset --output-format=json
```

Estimate the Cartesian end-effector impact of the worst joint offset for a 30 cm arm:
```shell
lerobot-check-calibration --repo-id=user/my_dataset --arm-length-cm=30
```
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import init_logging

VERDICT_OK = "ok"
VERDICT_MILD = "mild_offset"
VERDICT_SIGNIFICANT = "significant_offset"


def compute_episode_deltas(
    actions: np.ndarray, states: np.ndarray, vel_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute leader-follower deltas for one episode.

    Args:
        actions: Leader positions, shape (T, num_motors).
        states: Follower positions, shape (T, num_motors).
        vel_threshold: A frame is "stable" when every joint of the follower moved less than this amount
            (in action units) since the previous frame.

    Returns:
        A tuple `(stable_deltas, all_deltas)` where `all_deltas` has shape (T-1, num_motors) (frame 0 is
        skipped since it has no velocity estimate) and `stable_deltas` is the subset of rows from stable
        frames, possibly empty.
    """
    if actions.shape != states.shape:
        raise ValueError(f"actions {actions.shape} and states {states.shape} must have the same shape")
    if len(actions) < 2:
        empty = np.empty((0, actions.shape[1]) if actions.ndim == 2 else (0,))
        return empty, empty

    velocities = np.abs(np.diff(states, axis=0))  # (T-1, num_motors)
    stable_mask = np.max(velocities, axis=1) < vel_threshold
    all_deltas = actions[1:] - states[1:]
    return all_deltas[stable_mask], all_deltas


def summarize_calibration(
    stable_deltas: np.ndarray,
    all_deltas: np.ndarray,
    motor_names: list[str],
    ok_threshold: float = 1.0,
    warn_threshold: float = 3.0,
) -> list[dict]:
    """Build a per-motor summary of calibration deltas.

    Returns one dict per motor with keys `motor`, `mean_stable`, `std_stable`, `mean_all` and `verdict`
    (one of "ok", "mild_offset", "significant_offset" based on `|mean_stable|`).
    """
    summary = []
    for j, name in enumerate(motor_names):
        mean_stable = float(np.mean(stable_deltas[:, j]))
        abs_mean = abs(mean_stable)
        if abs_mean < ok_threshold:
            verdict = VERDICT_OK
        elif abs_mean < warn_threshold:
            verdict = VERDICT_MILD
        else:
            verdict = VERDICT_SIGNIFICANT
        summary.append(
            {
                "motor": name,
                "mean_stable": mean_stable,
                "std_stable": float(np.std(stable_deltas[:, j])),
                "mean_all": float(np.mean(all_deltas[:, j])),
                "verdict": verdict,
            }
        )
    return summary


def group_pairs_by_episode(hf_dataset) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Group (actions, states) arrays by episode from a LeRobotDataset's underlying hf_dataset."""
    columns = hf_dataset.select_columns(["episode_index", "frame_index", ACTION, OBS_STATE])
    rows_by_ep: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for row in columns:
        ep = int(row["episode_index"])
        rows_by_ep.setdefault(ep, []).append(
            (
                int(row["frame_index"]),
                np.asarray(row[ACTION], dtype=np.float64),
                np.asarray(row[OBS_STATE], dtype=np.float64),
            )
        )
    pairs_by_ep = {}
    for ep, rows in rows_by_ep.items():
        rows.sort(key=lambda r: r[0])
        pairs_by_ep[ep] = (np.stack([r[1] for r in rows]), np.stack([r[2] for r in rows]))
    return pairs_by_ep


def check_calibration(
    dataset: LeRobotDataset,
    vel_threshold: float = 0.5,
    ok_threshold: float = 1.0,
    warn_threshold: float = 3.0,
) -> dict:
    """Run the calibration drift analysis on a dataset.

    Returns a dict with aggregate counters and the per-motor summary (see `summarize_calibration`).
    Raises ValueError if the dataset has no stable frames at the given velocity threshold.
    """
    if ACTION not in dataset.meta.features or OBS_STATE not in dataset.meta.features:
        raise ValueError(
            f"Dataset must contain both '{ACTION}' and '{OBS_STATE}' features to compare leader and "
            f"follower positions. Found: {list(dataset.meta.features)}"
        )

    pairs_by_ep = group_pairs_by_episode(dataset.hf_dataset)

    stable_chunks = []
    all_chunks = []
    for actions, states in pairs_by_ep.values():
        stable, every = compute_episode_deltas(actions, states, vel_threshold)
        stable_chunks.append(stable)
        all_chunks.append(every)

    stable_deltas = np.concatenate(stable_chunks) if stable_chunks else np.empty((0, 0))
    all_deltas = np.concatenate(all_chunks) if all_chunks else np.empty((0, 0))

    if len(stable_deltas) == 0:
        raise ValueError(
            f"No stable frames found at vel_threshold={vel_threshold}. The robot may never settle in this "
            "dataset; try increasing --vel-threshold."
        )

    motor_names = dataset.meta.features[ACTION].get("names")
    if not motor_names:
        motor_names = [f"motor_{j}" for j in range(stable_deltas.shape[1])]

    return {
        "num_episodes": len(pairs_by_ep),
        "num_frames": int(len(all_deltas)),
        "num_stable_frames": int(len(stable_deltas)),
        "vel_threshold": vel_threshold,
        "motors": summarize_calibration(stable_deltas, all_deltas, motor_names, ok_threshold, warn_threshold),
    }


def print_table_report(repo_id: str, report: dict, arm_length_cm: float | None = None) -> None:
    """Print a human-readable report of the calibration analysis."""
    stable_pct = 100 * report["num_stable_frames"] / report["num_frames"]
    print("\nLeader (action) vs follower (observation.state) calibration check")
    print(f"  Dataset        : {repo_id}")
    print(f"  Episodes       : {report['num_episodes']}")
    print(f"  Frames         : {report['num_frames']}")
    print(f"  Stable frames  : {report['num_stable_frames']} ({stable_pct:.1f}%)")
    print(f"  Vel. threshold : {report['vel_threshold']} units/step\n")

    print(f"{'Motor':20s} | {'mean Δ':>10s} | {'std Δ':>10s} | {'mean Δ all':>10s} | verdict")
    print(f"{'-' * 20} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 20}")
    for m in report["motors"]:
        print(
            f"{m['motor']:20s} | {m['mean_stable']:+10.3f} | {m['std_stable']:10.3f} "
            f"| {m['mean_all']:+10.3f} | {m['verdict']}"
        )

    print()
    print("Interpretation:")
    print("  - mean Δ = action - observation.state on stable frames (follower converged)")
    print("  - mean Δ ≈ 0: leader and follower are aligned on this motor")
    print("  - |mean Δ| systematically above the ok-threshold: calibration offset between the arms")
    print("  - Grippers typically show a high std (fast open/close); focus on the arm joints")

    if arm_length_cm is not None:
        worst_offset = max(abs(m["mean_stable"]) for m in report["motors"])
        cartesian_cm = arm_length_cm * np.tan(np.radians(worst_offset))
        print()
        print(
            f"  Worst-case Cartesian impact at the end effector: ~{cartesian_cm:.1f} cm "
            f"(offset {worst_offset:.1f} deg on a {arm_length_cm:g} cm arm)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect leader/follower calibration drift in a recorded teleoperation dataset."
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
        "--vel-threshold",
        type=float,
        default=0.5,
        help="Max per-joint motion (in action units per step) for a frame to count as stable (default: 0.5).",
    )
    parser.add_argument(
        "--ok-threshold",
        type=float,
        default=1.0,
        help="|mean delta| below this value is reported as 'ok' (default: 1.0).",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=3.0,
        help="|mean delta| above this value is reported as 'significant_offset' (default: 3.0).",
    )
    parser.add_argument(
        "--arm-length-cm",
        type=float,
        default=None,
        help="Optional arm length in cm used to estimate the Cartesian impact of the worst joint offset "
        "(assumes deltas are in degrees).",
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

    try:
        report = check_calibration(
            dataset,
            vel_threshold=args.vel_threshold,
            ok_threshold=args.ok_threshold,
            warn_threshold=args.warn_threshold,
        )
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)

    if args.output_format == "json":
        print(json.dumps({"repo_id": args.repo_id, **report}, indent=2))
    else:
        print_table_report(args.repo_id, report, arm_length_cm=args.arm_length_cm)


if __name__ == "__main__":
    main()
