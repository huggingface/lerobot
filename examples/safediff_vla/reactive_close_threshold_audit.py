#!/usr/bin/env python
"""Read-only aggregation (Q3): reactive-close proximity-threshold statistics.

Runs NO new rollouts -- reuses three rollout-instrumentation JSON files already produced earlier
this session by other (unmodified-policy / privileged-diagnostic) scripts against the same B
checkpoint on select_poker:
  1. outputs/eval/safediff_vla_b_padfix20k_place_forensics/episodes_raw.json
     -- 10 UNMODIFIED-policy episodes (seeds 1000-1009), full per-step ee_pos/card_pos from live
        MuJoCo physics (real target-card position; the dataset itself has no such field -- see
        grasp_target_label_stats.json's "target_card_xyz_availability" note).
  2. outputs/eval/safediff_vla_privileged_grasp_factor_ablation/summary.json
     -- 12 episodes (4 conditions x 3 seeds), each with a real MuJoCo `is_grasped` contact query
        (`first_physical_contact_step`) independent of which controller produced the xyz.
  3. outputs/eval/safediff_vla_oracle_xyz_diagnostic/summary.json -- 3 oracle-corrected episodes.

Important caveat, reported explicitly rather than silently worked around: across every rollout
batch on record for this checkpoint on select_poker (10+12+3 = 25 episodes total), TASK SUCCESS is
0/25 -- the checkpoint never completes a full pick (see each file's own `success`/`task_success`
field). So there is no genuine "successful rollout" population to draw from, only episodes where a
real MuJoCo GRASP-CONTACT event occurred (`is_grasped` True at some step) even though the overall
task then still failed (lift/orientation/timing). This script uses grasp-contact (not task success)
as the closest available real, physics-verified event for calibrating a *grasp-approach* proximity
trigger -- which is what `proximity_close_threshold` is for -- and says so in the output.

Usage:
    uv run python examples/safediff_vla/reactive_close_threshold_audit.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def from_place_forensics(path: Path) -> list[dict]:
    episodes = json.loads(path.read_text())
    out = []
    for ep in episodes:
        steps = ep["steps"]
        dists = np.array([s["ee_to_card_dist"] for s in steps])
        ee = np.array([s["ee_pos"] for s in steps])
        card = np.array([s["card_pos"] for s in steps])
        delta = ee - card  # dx, dy, dz (ee - card, world frame; a residual, not a signed error metric)

        close_step = ep["first_commanded_close_step"]
        grasp_step = ep["first_physics_grasp_step"]

        record: dict = {
            "source": "place_phase_forensics(policy_baseline)",
            "seed": ep["seed"],
            "task_success": ep["success"],
            "grasp_contact_achieved": grasp_step is not None,
        }

        if close_step is not None and close_step < len(dists):
            record["dist_at_commanded_close"] = float(dists[close_step])
            record["dxyz_at_commanded_close"] = delta[close_step].tolist()
            pre = dists[:close_step]
            if len(pre) > 0:
                argmin = int(np.argmin(pre))
                record["min_dist_before_close"] = float(pre[argmin])
                record["dxyz_at_min_dist_before_close"] = delta[argmin].tolist()

        if grasp_step is not None and grasp_step < len(dists):
            record["dist_at_first_physical_contact"] = float(dists[grasp_step])
            record["dxyz_at_first_physical_contact"] = delta[grasp_step].tolist()

        record["min_dist_whole_episode"] = float(dists.min())
        record["dxyz_at_min_dist_whole_episode"] = delta[int(np.argmin(dists))].tolist()

        out.append(record)
    return out


def from_privileged_ablation(path: Path) -> list[dict]:
    episodes = json.loads(path.read_text())
    out = []
    for ep in episodes:
        steps = ep["step_records"]
        dists = {s["step_ix"] + 1: s["ee_to_card_dist"] for s in steps}
        record = {
            "source": f"privileged_grasp_factor_ablation({ep['condition']})",
            "seed": ep["seed"],
            "condition": ep["condition"],
            "task_success": ep["task_success"],
            "grasp_contact_achieved": ep["grasp_contact_success"],
            "dist_at_original_policy_close": ep["ee_to_card_dist_at_original_policy_close"],
            "dist_at_oracle_close": ep["ee_to_card_dist_at_oracle_close"],
        }
        if ep["first_physical_contact_step"] is not None:
            record["dist_at_first_physical_contact"] = dists.get(ep["first_physical_contact_step"])
        if ep["first_near_contact_step"] is not None:
            record["dist_at_first_near_contact"] = dists.get(ep["first_near_contact_step"])
        out.append(record)
    return out


def from_oracle_diag(path: Path) -> list[dict]:
    episodes = json.loads(path.read_text())
    return [
        {
            "source": "oracle_xyz_grasp_diagnostic",
            "seed": ep["seed"],
            "task_success": ep["success"],
            "dist_at_close": ep["ee_to_card_dist_at_close"],
        }
        for ep in episodes
    ]


def summarize(values: list[float]) -> dict | None:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": arr.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--place-forensics-raw",
        default="outputs/eval/safediff_vla_b_padfix20k_place_forensics/episodes_raw.json",
    )
    parser.add_argument(
        "--privileged-ablation",
        default="outputs/eval/safediff_vla_privileged_grasp_factor_ablation/summary.json",
    )
    parser.add_argument(
        "--oracle-diag", default="outputs/eval/safediff_vla_oracle_xyz_diagnostic/summary.json"
    )
    parser.add_argument(
        "--output-dir", default="outputs/eval/safediff_vla_grounded_grasp_supervision_audit"
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pf = from_place_forensics(Path(args.place_forensics_raw))
    pa = from_privileged_ablation(Path(args.privileged_ablation))
    od = from_oracle_diag(Path(args.oracle_diag))

    n_total_rollout_episodes = len(pf) + len(pa) + len(od)
    n_task_success = (
        sum(r["task_success"] for r in pf)
        + sum(r["task_success"] for r in pa)
        + sum(r["task_success"] for r in od)
    )

    contact_dists_policy_own_xyz = [
        r["dist_at_first_physical_contact"] for r in pf if r.get("dist_at_first_physical_contact") is not None
    ] + [
        r["dist_at_first_physical_contact"]
        for r in pa
        if r["condition"] == "policy_baseline" and r.get("dist_at_first_physical_contact") is not None
    ]
    contact_dists_oracle_xyz = [
        r["dist_at_first_physical_contact"]
        for r in pa
        if r["condition"] != "policy_baseline" and r.get("dist_at_first_physical_contact") is not None
    ]

    min_before_close = [r["min_dist_before_close"] for r in pf if r.get("min_dist_before_close") is not None]

    dxyz_at_contact = [
        r["dxyz_at_first_physical_contact"] for r in pf if r.get("dxyz_at_first_physical_contact") is not None
    ]
    dx = [v[0] for v in dxyz_at_contact]
    dy = [v[1] for v in dxyz_at_contact]
    dz = [v[2] for v in dxyz_at_contact]

    dist_at_commanded_close_all = [r["dist_at_commanded_close"] for r in pf if "dist_at_commanded_close" in r]

    result = {
        "note": (
            "No full-task-success rollout exists in any recorded batch for this checkpoint on "
            f"select_poker: {n_task_success}/{n_total_rollout_episodes} task_success across every "
            "pooled source. Statistics below use real MuJoCo grasp-CONTACT events "
            "(`is_grasped`/first_physical_contact_step True) as the closest available proxy for "
            "'successful' in a grasp-approach sense -- see module docstring."
        ),
        "n_rollout_episodes_pooled": n_total_rollout_episodes,
        "n_task_success": int(n_task_success),
        "n_grasp_contact_achieved_policy_own_xyz": len(contact_dists_policy_own_xyz),
        "n_grasp_contact_achieved_oracle_xyz": len(contact_dists_oracle_xyz),
        "dist_at_first_physical_contact_policy_own_xyz_m": summarize(contact_dists_policy_own_xyz),
        "dist_at_first_physical_contact_oracle_xyz_m": summarize(contact_dists_oracle_xyz),
        "min_dist_before_commanded_close_m": summarize(min_before_close),
        "dist_at_commanded_close_all_policy_baseline_episodes_m": summarize(dist_at_commanded_close_all),
        "axis_residual_at_first_physical_contact_m": {
            "dx": summarize(dx),
            "dy": summarize(dy),
            "dz": summarize(dz),
            "note": (
                "ee_pos - card_pos (world frame, robot base offset not subtracted) at the first "
                "real contact step, policy_baseline (own xyz) episodes only."
            ),
        },
        "raw_records": {
            "place_phase_forensics_policy_baseline": pf,
            "privileged_grasp_factor_ablation": pa,
            "oracle_xyz_grasp_diagnostic": od,
        },
    }

    (out_dir / "reactive_close_threshold_stats.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "raw_records"}, indent=2))
    print(f"wrote {out_dir / 'reactive_close_threshold_stats.json'}")


if __name__ == "__main__":
    main()
