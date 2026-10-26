#!/usr/bin/env python
"""INDEPENDENT re-verification of find_grasp_target() after the gripper-polarity/padding fix
(commit 6f8e923c) -- deliberately does NOT reuse or call A's own
examples/safediff_vla/audit_grasp_target_supervision.py or diagnose_target_point_head.py; ground
truth here is re-derived from scratch against the same real select_poker data this session's prior
(pre-fix) audit used, so the two can be compared line-for-line. No training, no rollout, no
threshold/lambda/model changes.

Checks (exhaustively, over all 500 select_poker episodes -- not a random sample):
  1. true pre-grasp + close-in-next-50 samples -> valid=True (recall)
  2. post-grasp samples -> valid=False (false-positive rate)
  3. no-transition-in-window samples -> valid=False (false-positive rate)
  4. padding-derived false positives
  5. action/state gripper polarity handled via separate threshold functions (structural)
  6. raw (not normalized) threshold applied correctly (structural, via the real call convention)
  7. valid coverage vs. the dataset-wide Q2 horizon-coverage audit's independent expectation (~66.3%)

Usage:
    uv run python examples/safediff_vla/verify_grounded_grasp_fix_independent.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from lerobot.policies.safediff_vla.utils import _action_gripper_is_open, _state_gripper_is_open, find_grasp_target

REPO_ID = "lerobot/vlabench_unified"
TASK_SUBSTRING = "poker"
GRIPPER_INDEX = 6
GRIPPER_OPEN_THRESHOLD_RAW = 0.5
ACTION_HORIZON = 50
# Independently computed by examples/safediff_vla/grounded_grasp_supervision_audit.py (dataset-only
# Q2 horizon-coverage audit, before this fix existed) -- the fixed pipeline's valid coverage should
# reproduce this number exactly, since it comes from the same raw action/state arithmetic.
DATASET_WIDE_EXPECTED_COVERAGE = 0.6630408829225586


def load_poker_data():
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
        repo_id=REPO_ID, repo_type="dataset", filename=f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet"
    )
    return pd.read_parquet(
        data_path,
        columns=["index", "episode_index", "frame_index", "observation.state", "action"],
        filters=[("episode_index", "in", list(ep_index_set))],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", default="outputs/eval/safediff_vla_grounded_grasp_pipeline_audit_v2"
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = torch.tensor([0.0, 0.3, 0.7, 1.0])  # 0.5 itself excluded: both strict-inequality funcs correctly read False there
    action_open = _action_gripper_is_open(probe, 0.5).tolist()
    state_open = _state_gripper_is_open(probe, 0.5).tolist()
    polarity_structural_check = {
        "probe_values": probe.tolist(),
        "_action_gripper_is_open_should_be_True_for_HIGH_values": action_open,
        "_state_gripper_is_open_should_be_True_for_LOW_values": state_open,
        "functions_are_opposite": action_open == [not v for v in state_open],
    }
    assert polarity_structural_check["functions_are_opposite"]

    df = load_poker_data()

    total_samples = 0
    n_true_pre_grasp_in_horizon = 0
    n_true_pre_grasp_in_horizon_correctly_valid = 0
    n_post_grasp = 0
    n_post_grasp_correctly_invalid = 0
    n_no_transition_in_window = 0
    n_no_transition_correctly_invalid = 0
    n_norm_valid_total = 0
    target_xyz_when_valid = []

    for ep_index, group in df.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index")
        actions_raw = torch.from_numpy(np.stack(group["action"].to_numpy()).astype(np.float64))
        states_raw = torch.from_numpy(np.stack(group["observation.state"].to_numpy()).astype(np.float64))
        n = len(actions_raw)

        raw_gripper_action = actions_raw[:, GRIPPER_INDEX].numpy()
        raw_action_is_open = raw_gripper_action > GRIPPER_OPEN_THRESHOLD_RAW
        prev_open = np.concatenate(([True], raw_action_is_open[:-1]))
        raw_closes = np.flatnonzero(prev_open & ~raw_action_is_open)
        close_row = int(raw_closes[0]) if len(raw_closes) else None

        raw_state_gripper = states_raw[:, GRIPPER_INDEX].numpy()
        correct_pre_grasp = raw_state_gripper < 0.5

        padded_actions = torch.cat([actions_raw, actions_raw[-1:].expand(ACTION_HORIZON, -1)], dim=0)
        windows = torch.stack([padded_actions[t : t + ACTION_HORIZON] for t in range(n)], dim=0)
        is_pad_row = torch.cat([torch.zeros(n, dtype=torch.bool), torch.ones(ACTION_HORIZON, dtype=torch.bool)])
        action_is_pad = torch.stack([is_pad_row[t : t + ACTION_HORIZON] for t in range(n)], dim=0)

        target_xyz_batch, valid_mask_batch = find_grasp_target(
            states_raw,
            windows,
            gripper_index=GRIPPER_INDEX,
            gripper_open_threshold=GRIPPER_OPEN_THRESHOLD_RAW,
            action_is_pad=action_is_pad,
        )
        valid_mask_batch = valid_mask_batch.numpy()
        target_xyz_batch = target_xyz_batch.numpy()

        for t in range(n):
            total_samples += 1
            norm_valid = bool(valid_mask_batch[t])
            if norm_valid:
                n_norm_valid_total += 1
                target_xyz_when_valid.append(target_xyz_batch[t])

            true_in_horizon = close_row is not None and 0 <= (close_row - t) <= (ACTION_HORIZON - 1)
            is_pre_grasp = bool(correct_pre_grasp[t])
            is_post_grasp = close_row is not None and t > close_row

            if true_in_horizon and is_pre_grasp:
                n_true_pre_grasp_in_horizon += 1
                if norm_valid:
                    n_true_pre_grasp_in_horizon_correctly_valid += 1
            if is_post_grasp:
                n_post_grasp += 1
                if not norm_valid:
                    n_post_grasp_correctly_invalid += 1
            if not true_in_horizon:
                n_no_transition_in_window += 1
                if not norm_valid:
                    n_no_transition_correctly_invalid += 1

    target_xyz_arr = np.array(target_xyz_when_valid) if target_xyz_when_valid else np.zeros((0, 3))
    coverage_frac = n_norm_valid_total / total_samples

    result = {
        "gripper_polarity_structural_check": polarity_structural_check,
        "total_samples": total_samples,
        "valid_coverage": {
            "n_valid": n_norm_valid_total,
            "frac_valid": coverage_frac,
            "dataset_wide_expected_coverage": DATASET_WIDE_EXPECTED_COVERAGE,
            "matches_expectation": abs(coverage_frac - DATASET_WIDE_EXPECTED_COVERAGE) < 0.01,
        },
        "true_pre_grasp_in_horizon": {
            "n": n_true_pre_grasp_in_horizon,
            "n_correctly_valid": n_true_pre_grasp_in_horizon_correctly_valid,
            "recall": (
                n_true_pre_grasp_in_horizon_correctly_valid / n_true_pre_grasp_in_horizon
                if n_true_pre_grasp_in_horizon
                else None
            ),
        },
        "post_grasp_samples": {
            "n": n_post_grasp,
            "n_correctly_invalid": n_post_grasp_correctly_invalid,
            "false_positive_rate": (1 - n_post_grasp_correctly_invalid / n_post_grasp) if n_post_grasp else None,
        },
        "no_transition_in_window_samples": {
            "n": n_no_transition_in_window,
            "n_correctly_invalid": n_no_transition_correctly_invalid,
            "false_positive_rate": (
                1 - n_no_transition_correctly_invalid / n_no_transition_in_window
                if n_no_transition_in_window
                else None
            ),
        },
        "target_xyz_when_valid_raw": {
            "mean": target_xyz_arr.mean(axis=0).tolist() if len(target_xyz_arr) else None,
            "std": target_xyz_arr.std(axis=0).tolist() if len(target_xyz_arr) else None,
            "n": len(target_xyz_arr),
        },
    }
    (out_dir / "independent_reverification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out_dir / 'independent_reverification.json'}")


if __name__ == "__main__":
    main()
