#!/usr/bin/env python
"""PHASE R1: reactive-close failure audit. OFFLINE ONLY -- reads the 18 already-completed E50
`GroundedGrasp-v2` episodes from Phase 1A (`outputs/eval/safediff_vla_grounded_grasp_v2_monitor_
baseline/<task>/episodes_raw.json`). NO new rollout, NO policy modification, NO reactive-close
logic change. GT target is used ONLY as an evaluation label below, never as a gating/inference
condition (consistent with every prior phase on this branch).

Current reactive-close condition (`_grounded_grasp_v2_reactive_close`, UNCHANGED here):
    ||EE - predicted_target|| <= grounded_grasp_close_threshold_m (0.03m)
where `predicted_target` is the cached `planner_target` (`_v2_last_target_xyz`), a one-shot latch.

Per close event, this script records (all already present in the saved per-step JSON except
`recent_var_m`/`planner_monitor_gap_m`/`monitor_jump_m`, computed the same way Phase 2A did):
    1. ee_to_planner_m        -- ||EE - predicted_target|| (by construction <= 0.03m at the
                                  REAL trigger step; computed from `ee_pos`/`planner_target_world`)
    2. ee_to_gt_m              -- `ee_to_target_dist` (already logged)
    3. planner_target_dist_to_gt_m -- `planner_target_dist_to_gt` (already logged) -- the predicted
                                  target's OWN bias from the true object
    4. monitor_target_dist_to_gt_m -- `monitor_target_dist_to_gt` (already logged)
    5. recent_var_m            -- K=10-step RMS spread of monitor_target (Phase 2A's own definition)
    6. planner_monitor_gap_m   -- ||monitor_target - planner_target|| (Phase 2A's own definition)
    7. monitor_jump_m          -- ||monitor_target(t) - monitor_target(t-1)|| (Phase 2A's own def)
    8. is_grasped_at_close     -- privileged sim contact, diagnostic only
    9. instructed-object-is-nearest -- NOT COMPUTABLE from existing logs (per-step full scene
       object list was never logged by any prior phase's logger) -- reported as unavailable, no
       new rollout run to get it (explicitly disallowed this phase).
    10. eventual native success -- `success` (episode-level)

Usage:
    uv run python examples/safediff_vla/phase_r1_reactive_close_failure_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import STAGE2_TASK_NAMES  # noqa: E402

SRC_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_monitor_baseline")
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_r1_reactive_close_audit")
K_WINDOW = 10
GOOD_CLOSE_GT_THRESHOLD_M = 0.05  # evaluation label only, never an inference condition
TAU_VAR_M = 0.0507  # reused verbatim from Phase 2A's own 80th-percentile-derived value, NOT retuned
TAU_GAP_M = 0.10  # reused verbatim from Phase 2A/2C, NOT retuned


def vec(steps: list[dict], key: str) -> np.ndarray:
    return np.array([s[key] if s[key] is not None else [np.nan, np.nan, np.nan] for s in steps], dtype=float)


def per_step_signals(ep: dict) -> dict:
    steps = ep["steps"]
    n = len(steps)
    monitor = vec(steps, "monitor_target_world")
    planner = vec(steps, "planner_target_world")
    ee = vec(steps, "ee_pos")
    valid_m = ~np.isnan(monitor).any(axis=1)
    valid_p = ~np.isnan(planner).any(axis=1)

    ee_to_planner = np.full(n, np.nan)
    ee_to_planner[valid_p] = np.linalg.norm((ee - planner)[valid_p], axis=1)

    gap = np.full(n, np.nan)
    both = valid_m & valid_p
    gap[both] = np.linalg.norm((monitor - planner)[both], axis=1)

    jump = np.full(n, np.nan)
    for t in range(1, n):
        if valid_m[t] and valid_m[t - 1]:
            jump[t] = float(np.linalg.norm(monitor[t] - monitor[t - 1]))

    recent_var = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - K_WINDOW + 1)
        window = monitor[lo : t + 1]
        window = window[~np.isnan(window).any(axis=1)]
        if len(window) >= 2:
            mean = window.mean(axis=0)
            recent_var[t] = float(np.sqrt(np.mean(np.linalg.norm(window - mean, axis=1) ** 2)))

    return {"ee_to_planner": ee_to_planner, "gap": gap, "jump": jump, "recent_var": recent_var}


def main() -> None:
    all_episodes = []
    for task in STAGE2_TASK_NAMES:
        raw_path = SRC_DIR / task / "episodes_raw.json"
        if not raw_path.exists():
            continue
        for ep in json.loads(raw_path.read_text()):
            ep["task"] = task
            all_episodes.append(ep)

    close_events = []  # one row per REAL reactive_close_event
    no_close_episodes = []
    close_step_fractions = []  # Q5

    for ep in all_episodes:
        steps = ep["steps"]
        sig = per_step_signals(ep)
        close_idx = next((i for i, s in enumerate(steps) if s.get("reactive_close_event")), None)
        if close_idx is None:
            no_close_episodes.append((ep["task"], ep["seed"]))
            continue

        close_step_fractions.append(close_idx / ep["episode_len"])
        s = steps[close_idx]
        row = {
            "task": ep["task"], "seed": ep["seed"], "close_step": close_idx, "episode_len": ep["episode_len"],
            "ee_to_planner_m": float(sig["ee_to_planner"][close_idx]) if not np.isnan(sig["ee_to_planner"][close_idx]) else None,
            "ee_to_gt_m": s["ee_to_target_dist"],
            "planner_target_dist_to_gt_m": s.get("planner_target_dist_to_gt"),
            "monitor_target_dist_to_gt_m": s.get("monitor_target_dist_to_gt"),
            "recent_var_m": float(sig["recent_var"][close_idx]) if not np.isnan(sig["recent_var"][close_idx]) else None,
            "planner_monitor_gap_m": float(sig["gap"][close_idx]) if not np.isnan(sig["gap"][close_idx]) else None,
            "monitor_jump_m": float(sig["jump"][close_idx]) if not np.isnan(sig["jump"][close_idx]) else None,
            "is_grasped_at_close": s.get("is_grasped"),
            "ever_grasped_privileged": any(st.get("is_grasped") for st in steps),
            "native_success": ep["success"],
        }
        row["good_close"] = (row["ee_to_gt_m"] is not None) and (row["ee_to_gt_m"] <= GOOD_CLOSE_GT_THRESHOLD_M)

        # Counterfactual gates evaluated AT THIS EXACT (real) trigger step only -- post-trigger
        # trajectory is already contaminated by the real close (gripper forced shut, phase flips),
        # so "would it have fired later instead" cannot be simulated from this log; this answers
        # "would a stricter gate have accepted or rejected closing AT THE MOMENT proximity alone
        # already does."
        var_ok = row["recent_var_m"] is not None and row["recent_var_m"] <= TAU_VAR_M
        gap_ok = row["planner_monitor_gap_m"] is not None and row["planner_monitor_gap_m"] <= TAU_GAP_M
        row["gate_B_var_passes"] = bool(var_ok)
        row["gate_C_gap_passes"] = bool(gap_ok)
        row["gate_D_both_pass"] = bool(var_ok and gap_ok)

        close_events.append(row)

    (OUT_DIR).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "close_events.json").write_text(json.dumps(close_events, indent=2))

    n_total = len(all_episodes)
    n_close = len(close_events)
    n_good = sum(1 for r in close_events if r["good_close"])
    n_bad = n_close - n_good

    print(f"episodes with a reactive-close event: {n_close}/{n_total}")
    print(f"  of those: GOOD (ee_to_gt<={GOOD_CLOSE_GT_THRESHOLD_M}m)={n_good}, BAD={n_bad}")
    print(f"episodes with NO close event at all: {len(no_close_episodes)} -> {no_close_episodes}\n")

    print("per-close-event detail:")
    for r in close_events:
        tag = "GOOD" if r["good_close"] else "BAD "
        print(
            f"  [{tag}] {r['task']:14s} seed={r['seed']} step={r['close_step']:3d}/{r['episode_len']:3d} "
            f"ee_to_gt={r['ee_to_gt_m']:.3f}m  planner_bias(pred->GT)={r['planner_target_dist_to_gt_m']:.3f}m  "
            f"recent_var={r['recent_var_m']:.3f}m  gap={r['planner_monitor_gap_m']:.3f}m  "
            f"gates(B,C,D)=({r['gate_B_var_passes']},{r['gate_C_gap_passes']},{r['gate_D_both_pass']})  "
            f"success={r['native_success']}"
        )

    # Q1: does the ALWAYS-<=3cm ee_to_planner distance correlate with ee_to_gt? (it structurally
    # cannot vary much -- report its own range for honesty) vs. planner_target_dist_to_gt's
    # correlation with ee_to_gt (the real driver of outcome quality).
    ee_to_planner_vals = [r["ee_to_planner_m"] for r in close_events if r["ee_to_planner_m"] is not None]
    ee_to_gt_vals = np.array([r["ee_to_gt_m"] for r in close_events])
    bias_vals = np.array([r["planner_target_dist_to_gt_m"] for r in close_events])
    corr_bias_vs_gt = float(np.corrcoef(bias_vals, ee_to_gt_vals)[0, 1]) if len(close_events) > 2 else None
    print(f"\nQ1: ee_to_planner_m range at close = [{min(ee_to_planner_vals):.3f}, {max(ee_to_planner_vals):.3f}]m "
          f"(always small, by construction -- NOT predictive of outcome)")
    print(f"    corr(planner_target_dist_to_gt_m, ee_to_gt_m) across {len(close_events)} close events = {corr_bias_vs_gt:.3f}")

    # Q2: do BAD closes have large planner bias?
    bad = [r for r in close_events if not r["good_close"]]
    good = [r for r in close_events if r["good_close"]]
    print(f"\nQ2: mean planner_target_dist_to_gt_m -- GOOD closes: "
          f"{np.mean([r['planner_target_dist_to_gt_m'] for r in good]) if good else float('nan'):.3f}m, "
          f"BAD closes: {np.mean([r['planner_target_dist_to_gt_m'] for r in bad]) if bad else float('nan'):.3f}m")
    frac_bad_explained_by_bias = (
        np.mean([r["planner_target_dist_to_gt_m"] > GOOD_CLOSE_GT_THRESHOLD_M for r in bad]) if bad else None
    )
    print(f"    fraction of BAD closes where planner_target_dist_to_gt_m > {GOOD_CLOSE_GT_THRESHOLD_M}m: {frac_bad_explained_by_bias}")

    # Q3: does stability (recent_var / gap) separate GOOD from BAD?
    print(f"\nQ3: recent_var_m -- GOOD: mean={np.mean([r['recent_var_m'] for r in good]) if good else float('nan'):.4f}, "
          f"BAD: mean={np.mean([r['recent_var_m'] for r in bad]) if bad else float('nan'):.4f}")
    print(f"    planner_monitor_gap_m -- GOOD: mean={np.mean([r['planner_monitor_gap_m'] for r in good]) if good else float('nan'):.4f}, "
          f"BAD: mean={np.mean([r['planner_monitor_gap_m'] for r in bad]) if bad else float('nan'):.4f}")

    # Q5: how early does PRE_GRASP end when a close happens?
    print(f"\nQ5: close_step / episode_len fraction -- mean={np.mean(close_step_fractions):.3f}, "
          f"median={np.median(close_step_fractions):.3f}, values={[round(f, 2) for f in close_step_fractions]}")

    # Counterfactual confusion table
    print("\nCounterfactual confusion table (evaluated AT the real trigger step only):")
    for gate_name, key in [("B (var)", "gate_B_var_passes"), ("C (gap)", "gate_C_gap_passes"), ("D (both)", "gate_D_both_pass")]:
        preserved_good = sum(1 for r in good if r[key])
        rejected_good = sum(1 for r in good if not r[key])
        preserved_bad = sum(1 for r in bad if r[key])
        rejected_bad = sum(1 for r in bad if not r[key])
        print(
            f"  gate {gate_name}: GOOD closes preserved={preserved_good}/{len(good)}, rejected={rejected_good}/{len(good)}  |  "
            f"BAD closes preserved={preserved_bad}/{len(bad)}, rejected={rejected_bad}/{len(bad)}"
        )

    summary = {
        "n_total_episodes": n_total, "n_close_events": n_close, "n_good": n_good, "n_bad": n_bad,
        "no_close_episodes": no_close_episodes,
        "corr_planner_bias_vs_ee_to_gt": corr_bias_vs_gt,
        "close_step_fraction_mean": float(np.mean(close_step_fractions)) if close_step_fractions else None,
        "close_step_fraction_median": float(np.median(close_step_fractions)) if close_step_fractions else None,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_DIR / 'close_events.json'} and {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
