#!/usr/bin/env python
"""Read-only dataset audit (no model/architecture/loss/execution changes, no new rollouts):
grasp-target label quality (Q1) + horizon-50 supervision coverage (Q2) for select_poker episodes
in lerobot/vlabench_unified. Companion to reactive_close_threshold_audit.py (Q3) -- together they
back examples/safediff_vla/../../outputs/eval/safediff_vla_grounded_grasp_supervision_audit/
summary.json.

Convention (matches compute_subgoal_labels.py and this session's other *_diagnostic.py scripts):
  - action[6] is the gripper channel; > GRIPPER_OPEN_THRESHOLD (0.5) = open, <= closed.
  - "first open->close transition" = first frame index t where gripper flips from open (t-1) to
    closed (t). This is A's candidate TargetPointHead grasp-anchor supervision event.

Known, already-documented constraint (see compute_subgoal_labels.py / vlabench_subgoal.py
docstrings, re-verified here against meta/info.json's own `features` list): true object (target
card) xyz is NOT recoverable for episodes already recorded in lerobot/vlabench_unified --
VLABench's per-episode object layout is not reproducible from env.reset(seed=...), and the dataset
stores no object-pose field at all (only observation.state = 7-dim EE pose+gripper, action = 7-dim
EE pose+gripper command). So this script cannot compute ||first_close_xyz - target_card_xyz||
directly -- see reactive_close_threshold_audit.py for the closest available substitute (real,
simulator-grounded EE-to-target distances reused from this session's existing rollout diagnostics).
What IS computed here, purely from action/state arrays:
  - transition existence / count / first-close timestep per episode
  - first-close action xyz, and the EE xyz immediately preceding it
  - +/-5 step xyz trajectory around first close (action and state)
  - a self-consistency "approach-convergence" signal: EE speed (state xyz finite-difference norm)
    in the window around first close -- a well-formed grasp-anchor label should coincide with the
    approach trajectory actually decelerating/stopping, not a close command issued mid-transit.
    This is the only supervision-quality signal available without ground-truth object pose.
  - a "false/retry close" heuristic (documented below, a deliberate threshold choice, not derived
    from an existing repo constant): a close event is flagged retry-like if the gripper reopens
    again within RETRY_REOPEN_WINDOW steps AND a second close follows within the same episode.
  - horizon-50 coverage: for every frame t, does a first-close event exist within (t, t+50]?

Usage:
    uv run python examples/safediff_vla/grounded_grasp_supervision_audit.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "lerobot/vlabench_unified"
TASK_SUBSTRING = "poker"
GRIPPER_INDEX = 6
GRIPPER_OPEN_THRESHOLD = 0.5
ACTION_HORIZON = 50
WINDOW = 5
RETRY_REOPEN_WINDOW = 15  # steps -- see module docstring; a deliberate choice, not a repo constant


def find_transitions(gripper: np.ndarray) -> np.ndarray:
    """Frame indices where gripper flips open(t-1) -> closed(t). Empty array if none."""
    is_open = gripper > GRIPPER_OPEN_THRESHOLD
    closes = np.flatnonzero((~is_open[1:]) & is_open[:-1]) + 1
    return closes


def analyze_episode(ep_index: int, states: np.ndarray, actions: np.ndarray) -> dict:
    n = len(actions)
    gripper = actions[:, GRIPPER_INDEX]
    closes = find_transitions(gripper)
    is_open = gripper > GRIPPER_OPEN_THRESHOLD

    result: dict = {
        "episode_index": int(ep_index),
        "episode_length": int(n),
        "has_open_close_transition": bool(len(closes) > 0),
        "num_open_close_transitions": int(len(closes)),
        "first_close_step": None,
        "first_close_action_xyz": None,
        "pre_close_ee_xyz": None,
        "trajectory_window": None,
        "approach_speed_before_m_per_step": None,
        "approach_speed_after_m_per_step": None,
        "looks_like_retry_or_false_close": None,
    }
    if len(closes) == 0:
        return result

    c = int(closes[0])
    result["first_close_step"] = c
    result["first_close_action_xyz"] = actions[c, :3].tolist()
    result["pre_close_ee_xyz"] = states[c - 1, :3].tolist() if c - 1 >= 0 else states[c, :3].tolist()

    lo, hi = max(0, c - WINDOW), min(n - 1, c + WINDOW)
    result["trajectory_window"] = {
        "frame_indices": list(range(lo, hi + 1)),
        "action_xyz": actions[lo : hi + 1, :3].tolist(),
        "state_xyz": states[lo : hi + 1, :3].tolist(),
        "gripper_action": gripper[lo : hi + 1].tolist(),
    }

    pre_lo = max(0, c - WINDOW)
    if c - pre_lo >= 1:
        pre_disp = np.diff(states[pre_lo : c + 1, :3], axis=0)
        result["approach_speed_before_m_per_step"] = float(np.linalg.norm(pre_disp, axis=1).mean())
    post_hi = min(n - 1, c + WINDOW)
    if post_hi - c >= 1:
        post_disp = np.diff(states[c : post_hi + 1, :3], axis=0)
        result["approach_speed_after_m_per_step"] = float(np.linalg.norm(post_disp, axis=1).mean())

    retry = False
    if len(closes) >= 2:
        reopen_idx = np.flatnonzero(is_open[c:])
        if len(reopen_idx) > 0 and reopen_idx[0] <= RETRY_REOPEN_WINDOW:
            retry = True
    result["looks_like_retry_or_false_close"] = retry
    result["all_close_steps"] = closes.tolist()

    return result


def horizon_coverage_for_episode(n: int, closes: np.ndarray) -> list[dict]:
    """Per-sample (t=0..n-1) horizon-50 label coverage rows."""
    closes_sorted = np.sort(closes)
    rows = []
    for t in range(n):
        pos = np.searchsorted(closes_sorted, t, side="right")
        nxt = int(closes_sorted[pos]) if pos < len(closes_sorted) else None
        remaining_steps = n - 1 - t
        in_horizon = nxt is not None and (nxt - t) <= ACTION_HORIZON
        rel = (nxt - t) if in_horizon else None
        rows.append(
            {
                "t": t,
                "pre_grasp": nxt is not None,
                "close_within_horizon": bool(in_horizon),
                "relative_close_timestep": rel,
                "horizon_truncated_by_episode_end": remaining_steps < ACTION_HORIZON,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", default="outputs/eval/safediff_vla_grounded_grasp_supervision_audit"
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="meta/tasks.parquet")
    tasks_df = pd.read_parquet(tasks_path).reset_index()

    episodes_meta_path = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename="meta/episodes/chunk-000/file-000.parquet"
    )
    episodes_meta = pd.read_parquet(
        episodes_meta_path,
        columns=["episode_index", "tasks", "length", "data/chunk_index", "data/file_index"],
    )
    poker_meta = episodes_meta[
        episodes_meta["tasks"].apply(lambda ts: any(TASK_SUBSTRING in t.lower() for t in ts))
    ]
    print(f"poker episodes (meta): {len(poker_meta)}")

    ep_index_set = set(poker_meta["episode_index"].tolist())
    (chunk_idx,) = poker_meta["data/chunk_index"].unique().tolist()
    (file_idx,) = poker_meta["data/file_index"].unique().tolist()
    data_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet",
    )
    df = pd.read_parquet(
        data_path,
        columns=["index", "episode_index", "frame_index", "observation.state", "action"],
        filters=[("episode_index", "in", list(ep_index_set))],
    )
    print(f"loaded {len(df)} frames across {df['episode_index'].nunique()} episodes")

    episode_results = []
    histogram = np.zeros(ACTION_HORIZON, dtype=np.int64)
    total_samples = 0
    samples_with_close_in_horizon = 0
    pre_grasp_samples = 0
    pre_grasp_samples_with_valid_label = 0
    truncated_end_samples = 0
    truncated_end_samples_invalid = 0

    for ep_index, group in df.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index")
        actions = np.stack(group["action"].to_numpy()).astype(np.float64)
        states = np.stack(group["observation.state"].to_numpy()).astype(np.float64)
        n = len(actions)

        ep_result = analyze_episode(ep_index, states, actions)
        episode_results.append(ep_result)

        closes = np.array(ep_result.get("all_close_steps", []), dtype=np.int64)
        if not ep_result["has_open_close_transition"]:
            closes = np.array([], dtype=np.int64)

        for row in horizon_coverage_for_episode(n, closes):
            total_samples += 1
            if row["close_within_horizon"]:
                samples_with_close_in_horizon += 1
                histogram[row["relative_close_timestep"] - 1] += 1
            if row["pre_grasp"]:
                pre_grasp_samples += 1
                if row["close_within_horizon"]:
                    pre_grasp_samples_with_valid_label += 1
            if row["horizon_truncated_by_episode_end"]:
                truncated_end_samples += 1
                if not row["close_within_horizon"]:
                    truncated_end_samples_invalid += 1

    n_eps = len(episode_results)
    n_with_transition = sum(1 for r in episode_results if r["has_open_close_transition"])
    n_multi = sum(1 for r in episode_results if r["num_open_close_transitions"] >= 2)
    n_retry = sum(1 for r in episode_results if r.get("looks_like_retry_or_false_close"))
    close_steps = [r["first_close_step"] for r in episode_results if r["first_close_step"] is not None]
    speeds_before = [
        r["approach_speed_before_m_per_step"]
        for r in episode_results
        if r["approach_speed_before_m_per_step"] is not None
    ]
    speeds_after = [
        r["approach_speed_after_m_per_step"]
        for r in episode_results
        if r["approach_speed_after_m_per_step"] is not None
    ]

    def pct(x, q):
        return float(np.percentile(x, q)) if len(x) else None

    label_stats = {
        "dataset": REPO_ID,
        "task_filter": f"select_poker (task string contains '{TASK_SUBSTRING}')",
        "n_episodes": n_eps,
        "n_episodes_with_open_close_transition": n_with_transition,
        "frac_episodes_with_open_close_transition": n_with_transition / n_eps,
        "n_episodes_with_multiple_transitions": n_multi,
        "frac_episodes_with_multiple_transitions": n_multi / n_eps,
        "n_episodes_flagged_retry_or_false_close": n_retry,
        "frac_episodes_flagged_retry_or_false_close": n_retry / n_eps,
        "retry_heuristic": (
            f"gripper reopens within {RETRY_REOPEN_WINDOW} steps of the first close AND a second "
            "close follows later in the same episode -- deliberate threshold, not a repo constant"
        ),
        "first_close_step": {
            "count": len(close_steps),
            "mean": float(np.mean(close_steps)) if close_steps else None,
            "std": float(np.std(close_steps)) if close_steps else None,
            "min": int(np.min(close_steps)) if close_steps else None,
            "p25": pct(close_steps, 25),
            "median": pct(close_steps, 50),
            "p75": pct(close_steps, 75),
            "max": int(np.max(close_steps)) if close_steps else None,
        },
        "approach_speed_m_per_step": {
            "before_close_window_mean": float(np.mean(speeds_before)) if speeds_before else None,
            "after_close_window_mean": float(np.mean(speeds_after)) if speeds_after else None,
            "note": (
                "speed = mean L2 norm of consecutive observation.state xyz deltas in the +/-5 step "
                "window around first_close_step. A well-formed grasp-anchor label should show "
                "after < before (arm has converged/stopped, consistent with having reached the "
                "object), not comparable-or-larger speed after close (would suggest the close is "
                "issued mid-transit, not at an arrival point)."
            ),
        },
        "target_card_xyz_availability": (
            "NOT AVAILABLE. lerobot/vlabench_unified stores only observation.state (7-dim EE "
            "pose+gripper) and action (7-dim EE pose+gripper command) per meta/info.json's feature "
            "list -- no object/target pose field exists. Matches the already-documented finding in "
            "compute_subgoal_labels.py / src/lerobot/datasets/adapters/vlabench_subgoal.py: "
            "VLABench's per-episode object layout is not reproducible from env.reset(seed=...), so "
            "true target positions cannot be recovered for these already-recorded episodes even "
            "with simulator access. ||first_close_xyz - target_card_xyz|| therefore cannot be "
            "computed from this dataset; see reactive_close_threshold_audit.py for the closest "
            "available substitute."
        ),
    }

    horizon_stats = {
        "action_horizon": ACTION_HORIZON,
        "total_samples": total_samples,
        "samples_with_close_within_horizon": samples_with_close_in_horizon,
        "frac_samples_with_close_within_horizon": samples_with_close_in_horizon / total_samples,
        "pre_grasp_samples": pre_grasp_samples,
        "pre_grasp_samples_with_valid_label_in_horizon": pre_grasp_samples_with_valid_label,
        "frac_pre_grasp_samples_with_valid_label": (
            pre_grasp_samples_with_valid_label / pre_grasp_samples if pre_grasp_samples else None
        ),
        "samples_near_episode_end_horizon_truncated": truncated_end_samples,
        "samples_near_episode_end_with_invalid_target": truncated_end_samples_invalid,
        "frac_end_truncated_samples_invalid": (
            truncated_end_samples_invalid / truncated_end_samples if truncated_end_samples else None
        ),
        "relative_close_timestep_histogram_1_to_50": histogram.tolist(),
        "recommended_valid_mask_rule": (
            "valid_target[t] = True iff a first (nearest upcoming) open->close gripper transition "
            "exists strictly within (t, t+action_horizon]. If none exists in that window (episode "
            "never grasps again ahead, or the episode ends before the next grasp), mask the "
            "TargetPointHead regression loss for that sample instead of falling back to "
            "compute_subgoal_labels.py's 'last frame' default -- that default encodes 'nothing left "
            "to reach for', not a grasp-anchor position."
        ),
    }

    (out_dir / "grasp_target_label_stats.json").write_text(
        json.dumps({"summary": label_stats, "episodes": episode_results}, indent=2)
    )
    (out_dir / "close_relative_timestep_histogram.json").write_text(json.dumps(horizon_stats, indent=2))

    print(json.dumps(label_stats, indent=2))
    print(
        json.dumps(
            {k: v for k, v in horizon_stats.items() if k != "relative_close_timestep_histogram_1_to_50"},
            indent=2,
        )
    )
    print(f"wrote {out_dir / 'grasp_target_label_stats.json'}")
    print(f"wrote {out_dir / 'close_relative_timestep_histogram.json'}")


if __name__ == "__main__":
    main()
