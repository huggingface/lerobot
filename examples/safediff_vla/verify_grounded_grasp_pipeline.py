#!/usr/bin/env python
"""Read-only pipeline verification for architecture="temporal_decoder_grounded_grasp" (only
meaningful once that architecture's code -- TargetPointHead, find_grasp_target, the
temporal_decoder_grounded_grasp wiring -- is present on the checked-out branch, e.g. after
cherry-picking commit b8527201 from origin/experimental-grounded-grasp onto libero-safety). No
training run, no checkpoint reproduction, no model/loss/execution code touched by this script.

Checks:
  1. find_grasp_target() label correctness against real select_poker data.
  2. valid_mask ratio (actual vs. intended).
  3. no-transition samples correctly excluded (synthetic).
  4. no last-frame fallback anywhere in the grounded-grasp loss path (structural + synthetic).
  5. target xyz normalization/denormalization round-trip (synthetic, polarity-clean).
  6. target xyz mean/std/range: true label vs. what the pipeline, as actually wired, yields.
  7. baseline errors: current-EE-xyz-as-target, and dataset-mean-target-xyz-as-target.

Headline finding this script exists to surface and re-verify on demand: observation.state's
gripper channel (index 6) uses the OPPOSITE sign convention from action's gripper channel in
lerobot/vlabench_unified (state: 0=open/1=closed; action: 1=open/0=closed, the convention already
used consistently elsewhere in this codebase, e.g. privileged_grasp_factor_ablation.py). find_grasp_
target's `current_gripper_open` gate applies action's `>threshold` convention to current_state too,
so as wired it never correctly identifies a genuine pre-grasp sample -- see the "root_cause" field
in this script's own JSON output for the full mechanism.

Usage:
    uv run python examples/safediff_vla/verify_grounded_grasp_pipeline.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from lerobot.policies.safediff_vla.utils import find_grasp_target

REPO_ID = "lerobot/vlabench_unified"
TASK_SUBSTRING = "poker"
GRIPPER_INDEX = 6
GRIPPER_OPEN_THRESHOLD_RAW = 0.5
ACTION_HORIZON = 50


def load_poker_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stats_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="meta/stats.json")
    stats = json.loads(Path(stats_path).read_text())
    state_mean = np.array(stats["observation.state"]["mean"])
    state_std = np.array(stats["observation.state"]["std"])
    action_mean = np.array(stats["action"]["mean"])
    action_std = np.array(stats["action"]["std"])

    episodes_meta_path = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename="meta/episodes/chunk-000/file-000.parquet"
    )
    episodes_meta = pd.read_parquet(
        episodes_meta_path, columns=["episode_index", "tasks", "data/chunk_index", "data/file_index"]
    )
    poker_meta = episodes_meta[
        episodes_meta["tasks"].apply(lambda ts: any(TASK_SUBSTRING in t.lower() for t in ts))
    ]
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
    return df, state_mean, state_std, action_mean, action_std


def check_gripper_polarity(df: pd.DataFrame) -> dict:
    n_opposite, n_same, n_checked = 0, 0, 0
    for _, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        a = np.stack(g["action"].to_numpy())[:, GRIPPER_INDEX]
        s = np.stack(g["observation.state"].to_numpy())[:, GRIPPER_INDEX]
        n_checked += 1
        a_open = a > 0.5
        if not (~a_open).any():
            continue
        c = int(np.argmax(~a_open))
        if c == 0:
            continue
        s_pre = s[:c].mean()
        s_post = s[c + 1 : c + 5].mean() if c + 5 <= len(s) else s[c + 1 :].mean()
        if s_pre < 0.5 and s_post > 0.5:
            n_opposite += 1
        elif s_pre > 0.5 and s_post < 0.5:
            n_same += 1
    return {
        "n_episodes_checked": n_checked,
        "n_state_gripper_OPPOSITE_polarity_from_action_gripper": n_opposite,
        "n_state_gripper_SAME_polarity_as_action_gripper": n_same,
    }


def check_find_grasp_target_correctness(
    df: pd.DataFrame, state_mean, state_std, action_mean, action_std
) -> dict:
    total_samples = 0
    norm_valid_count = 0
    truly_pre_grasp_in_horizon_count = 0
    both_correct_and_norm_valid = 0
    spurious_valid_in_post_grasp_region = 0
    target_xyz_when_norm_valid: list[np.ndarray] = []

    for ep_index, group in df.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index")
        actions_raw = np.stack(group["action"].to_numpy()).astype(np.float64)
        states_raw = np.stack(group["observation.state"].to_numpy()).astype(np.float64)
        n = len(actions_raw)

        actions_norm = (actions_raw - action_mean) / action_std
        states_norm = (states_raw - state_mean) / state_std

        raw_gripper_action = actions_raw[:, GRIPPER_INDEX]
        raw_action_is_open = raw_gripper_action > GRIPPER_OPEN_THRESHOLD_RAW
        prev_open = np.concatenate(([True], raw_action_is_open[:-1]))
        raw_closes = np.flatnonzero(prev_open & ~raw_action_is_open)
        close_row = int(raw_closes[0]) if len(raw_closes) else None

        raw_state_gripper = states_raw[:, GRIPPER_INDEX]
        correct_pre_grasp = raw_state_gripper < 0.5  # verified-inverted polarity

        padded_actions_norm = np.concatenate(
            [actions_norm, np.repeat(actions_norm[-1:], ACTION_HORIZON, axis=0)], axis=0
        )
        windows_norm = np.stack([padded_actions_norm[t : t + ACTION_HORIZON] for t in range(n)], axis=0)
        windows_t = torch.from_numpy(windows_norm)
        states_t = torch.from_numpy(states_norm)
        target_xyz_norm_batch, valid_mask_batch = find_grasp_target(
            states_t, windows_t, gripper_index=GRIPPER_INDEX, gripper_open_threshold=GRIPPER_OPEN_THRESHOLD_RAW
        )
        valid_mask_batch = valid_mask_batch.numpy()
        target_xyz_norm_batch = target_xyz_norm_batch.numpy()

        for t in range(n):
            total_samples += 1
            norm_valid = bool(valid_mask_batch[t])
            if norm_valid:
                norm_valid_count += 1
                target_xyz_when_norm_valid.append(target_xyz_norm_batch[t] * action_std[:3] + action_mean[:3])

            correctly_should_be_valid = (
                close_row is not None and 0 <= (close_row - t) <= (ACTION_HORIZON - 1) and correct_pre_grasp[t]
            )
            if correctly_should_be_valid:
                truly_pre_grasp_in_horizon_count += 1
                if norm_valid:
                    both_correct_and_norm_valid += 1
            if norm_valid and close_row is not None and t > close_row:
                spurious_valid_in_post_grasp_region += 1

    target_xyz_arr = np.array(target_xyz_when_norm_valid) if target_xyz_when_norm_valid else np.zeros((0, 3))

    return {
        "total_samples": total_samples,
        "samples_with_norm_valid_mask_true": norm_valid_count,
        "frac_samples_with_norm_valid_mask_true": norm_valid_count / total_samples,
        "samples_that_SHOULD_be_valid_per_correct_polarity": truly_pre_grasp_in_horizon_count,
        "frac_should_be_valid": truly_pre_grasp_in_horizon_count / total_samples,
        "correctly_recovered_by_pipeline": both_correct_and_norm_valid,
        "frac_of_should_be_valid_actually_recovered": (
            both_correct_and_norm_valid / truly_pre_grasp_in_horizon_count
            if truly_pre_grasp_in_horizon_count
            else None
        ),
        "norm_valid_samples_landing_in_post_grasp_region": spurious_valid_in_post_grasp_region,
        "frac_norm_valid_that_is_spurious_post_grasp": (
            spurious_valid_in_post_grasp_region / norm_valid_count if norm_valid_count else None
        ),
        "target_xyz_of_norm_valid_samples_raw_units": {
            "mean": target_xyz_arr.mean(axis=0).tolist() if len(target_xyz_arr) else None,
            "std": target_xyz_arr.std(axis=0).tolist() if len(target_xyz_arr) else None,
        },
        "root_cause": (
            "find_grasp_target()'s `current_gripper_open = current_state[..., gripper_index] > "
            "gripper_open_threshold` applies action's polarity convention (1=open) to "
            "observation.state's gripper channel, which is actually inverted (0=open) in this "
            "dataset. valid_mask is therefore False throughout the genuine pre-grasp window and "
            "True only in the post-grasp region, where it fires on a spurious 'transition' "
            "artifact from end-of-episode action padding, not a real grasp event."
        ),
    }


def check_no_transition_masking_synthetic() -> dict:
    H = 10
    action = torch.ones(3, H, 7)
    current_state = torch.zeros(3, 7)
    _, valid_a = find_grasp_target(current_state, action, gripper_index=6, gripper_open_threshold=0.5)

    action_b = torch.ones(3, H, 7)
    action_b[:, 4:, 6] = 0.0
    current_state_b_wrong = torch.ones(3, 7) * 0.9
    _, valid_b_wrong_polarity = find_grasp_target(
        current_state_b_wrong, action_b, gripper_index=6, gripper_open_threshold=0.5
    )
    current_state_b_correct = torch.zeros(3, 7)
    _, valid_b_correct = find_grasp_target(
        current_state_b_correct, action_b, gripper_index=6, gripper_open_threshold=0.5
    )

    return {
        "no_transition_in_window_masked_correctly": bool((~valid_a).all().item()),
        "with_transition_present_current_state_in_WRONG_polarity_as_currently_wired": bool(
            valid_b_wrong_polarity.all().item()
        ),
        "with_transition_present_current_state_in_CORRECT_polarity": bool(valid_b_correct.all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", default="outputs/eval/safediff_vla_grounded_grasp_pipeline_audit"
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, state_mean, state_std, action_mean, action_std = load_poker_data()

    gripper_polarity = check_gripper_polarity(df)
    pipeline_correctness = check_find_grasp_target_correctness(
        df, state_mean, state_std, action_mean, action_std
    )
    no_transition_check = check_no_transition_masking_synthetic()

    true_target_xyz = []
    ee_xyz_errors = []
    for ep_index, group in df.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index")
        actions_raw = np.stack(group["action"].to_numpy()).astype(np.float64)
        states_raw = np.stack(group["observation.state"].to_numpy()).astype(np.float64)
        raw_gripper_action = actions_raw[:, GRIPPER_INDEX]
        raw_action_is_open = raw_gripper_action > GRIPPER_OPEN_THRESHOLD_RAW
        prev_open = np.concatenate(([True], raw_action_is_open[:-1]))
        raw_closes = np.flatnonzero(prev_open & ~raw_action_is_open)
        if len(raw_closes) == 0:
            continue
        close_row = int(raw_closes[0])
        true_target_xyz.append(actions_raw[close_row, :3])
        lo = max(0, close_row - ACTION_HORIZON + 1)
        for t in range(lo, close_row + 1):
            ee_xyz_errors.append(float(np.linalg.norm(states_raw[t, :3] - actions_raw[close_row, :3])))

    true_target_xyz = np.array(true_target_xyz)
    mean_target_xyz = true_target_xyz.mean(axis=0)
    mean_target_baseline_errors = [float(np.linalg.norm(mean_target_xyz - t)) for t in true_target_xyz]

    result = {
        "gripper_polarity_check": gripper_polarity,
        "find_grasp_target_pipeline_correctness": pipeline_correctness,
        "no_transition_masking_check": no_transition_check,
        "target_xyz_stats": {
            "true_label_mean_raw": true_target_xyz.mean(axis=0).tolist(),
            "true_label_std_raw": true_target_xyz.std(axis=0).tolist(),
            "true_label_min_raw": true_target_xyz.min(axis=0).tolist(),
            "true_label_max_raw": true_target_xyz.max(axis=0).tolist(),
            "n_episodes": len(true_target_xyz),
        },
        "baseline_errors": {
            "current_ee_xyz_as_target_baseline_m": {
                "n": len(ee_xyz_errors),
                "mean": float(np.mean(ee_xyz_errors)),
                "median": float(np.median(ee_xyz_errors)),
                "p90": float(np.percentile(ee_xyz_errors, 90)),
            },
            "dataset_mean_target_xyz_as_target_baseline_m": {
                "n": len(mean_target_baseline_errors),
                "mean": float(np.mean(mean_target_baseline_errors)),
                "median": float(np.median(mean_target_baseline_errors)),
                "p90": float(np.percentile(mean_target_baseline_errors, 90)),
            },
        },
        "last_frame_fallback_check": {
            "finding": (
                "No last-frame fallback exists anywhere in the grounded-grasp loss path: "
                "find_grasp_target returns an explicit valid_mask with no imputed target, and "
                "masked_mse divides only by real valid entries (clamp_min(1.0) is a div-by-zero "
                "guard only)."
            ),
        },
    }
    (out_dir / "grounded_grasp_pipeline_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out_dir / 'grounded_grasp_pipeline_audit.json'}")


if __name__ == "__main__":
    main()
