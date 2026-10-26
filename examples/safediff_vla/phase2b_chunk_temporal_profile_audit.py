#!/usr/bin/env python
"""PHASE 2B: OFFLINE audit of whether a committed 50-step `GroundedGrasp-v2` chunk has an internal
"xy alignment early -> z descent late" temporal structure -- the mechanism Phase 1B's execute_
horizon ablation (oscillation/jerk up, reactive-close down as execute_horizon shrinks) suggested
but did not directly verify. Reads the SAME 18 already-completed E50 episodes Phase 2A used
(`outputs/eval/safediff_vla_grounded_grasp_v2_monitor_baseline/<task>/episodes_raw.json`) -- no new
rollout, no policy behavior change.

For every env step, `offset_in_chunk = t - (index of the most recent actual replan)` is recovered
from the already-logged `planner_replanned_this_step` flags (every `execute_horizon=50` steps at
E50, so offsets span 0..~49 per committed chunk). For every pair of consecutive steps THAT SHARE
THE SAME CHUNK (never diffing across a replan boundary), from the already-logged `commanded_xyz_
world` / `commanded_rot_euler_raw` / `commanded_gripper_raw` (the ACTUAL executed action at each
step, popped from that chunk):
    z displacement      = commanded_z(t) - commanded_z(t-1)      (signed; negative = descending)
    xy correction mag   = ||commanded_xy(t) - commanded_xy(t-1)||
    rotation change mag = ||commanded_rot(t) - commanded_rot(t-1)||  (raw Euler diff -- NOT
                           wrap-corrected; a caveat for any step pair that happens to straddle the
                           +-pi branch cut, see `rotation_encoding.py`'s own docstring on this)
    gripper value        = commanded_gripper_raw(t)  (trend, not a diff -- raw convention:
                           `action[...,6] > grounded_grasp_gripper_open_threshold(=0.5)` == open,
                           so a value trending below 0.5 as offset grows means "closing")

All of these are binned and averaged by `offset_in_chunk` (0..49), pooled across every completed
chunk from all 18 episodes, to answer Q1/Q2. A SEPARATE profile restricted to only the chunks that
contain a `reactive_close_event` answers Q3. Q4 (does E10/E5 ever reach the late-chunk region) is
answered directly from `execute_horizon`'s own value -- no new computation needed.

Usage:
    uv run python examples/safediff_vla/phase2b_chunk_temporal_profile_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import STAGE2_TASK_NAMES  # noqa: E402

SRC_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_monitor_baseline")
OUT_DIR = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase2b_chunk_profile_audit")
MAX_OFFSET = 49


def vec(steps: list[dict], key: str) -> np.ndarray:
    return np.array([s[key] if s[key] is not None else [np.nan, np.nan, np.nan] for s in steps], dtype=float)


def per_step_arrays(ep: dict) -> dict:
    steps = ep["steps"]
    n = len(steps)
    commanded_xyz = vec(steps, "commanded_xyz_world")
    commanded_rot = vec(steps, "commanded_rot_euler_raw")
    commanded_grip = np.array([s["commanded_gripper_raw"] if s["commanded_gripper_raw"] is not None else np.nan for s in steps], dtype=float)
    replanned = np.array([bool(s["planner_replanned_this_step"]) for s in steps], dtype=bool)
    reactive_close = np.array([bool(s["reactive_close_event"]) for s in steps], dtype=bool)

    offset = np.full(n, -1, dtype=int)
    chunk_id = np.full(n, -1, dtype=int)
    cur_offset = -1
    cur_chunk = -1
    for t in range(n):
        if replanned[t]:
            cur_offset = 0
            cur_chunk += 1
        elif cur_offset >= 0:
            cur_offset += 1
        offset[t] = cur_offset
        chunk_id[t] = cur_chunk

    return {
        "commanded_xyz": commanded_xyz, "commanded_rot": commanded_rot, "commanded_grip": commanded_grip,
        "offset": offset, "chunk_id": chunk_id, "reactive_close": reactive_close, "n": n,
    }


def collect_profile_rows(all_episodes: list[dict], only_close_chunks: bool) -> list[dict]:
    rows = []
    for ep in all_episodes:
        a = per_step_arrays(ep)
        n = a["n"]
        close_chunk_ids = set(a["chunk_id"][a["reactive_close"]].tolist()) if only_close_chunks else None
        for t in range(1, n):
            if a["offset"][t] < 1 or a["chunk_id"][t] != a["chunk_id"][t - 1]:
                continue  # never diff across a replan boundary, and offset 0 has no same-chunk predecessor
            if only_close_chunks and a["chunk_id"][t] not in close_chunk_ids:
                continue
            dz = a["commanded_xyz"][t, 2] - a["commanded_xyz"][t - 1, 2]
            dxy = np.linalg.norm(a["commanded_xyz"][t, :2] - a["commanded_xyz"][t - 1, :2])
            drot = np.linalg.norm(a["commanded_rot"][t] - a["commanded_rot"][t - 1])
            if np.isnan(dz) or np.isnan(dxy) or np.isnan(drot):
                continue
            rows.append(
                {
                    "offset": int(a["offset"][t]), "dz": float(dz), "dxy": float(dxy), "drot": float(drot),
                    "grip": float(a["commanded_grip"][t]) if not np.isnan(a["commanded_grip"][t]) else None,
                }
            )
    return rows


def bin_by_offset(rows: list[dict], key: str, max_offset: int = MAX_OFFSET) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.arange(0, max_offset + 1)
    means = np.full(len(offsets), np.nan)
    counts = np.zeros(len(offsets), dtype=int)
    for i, off in enumerate(offsets):
        vals = [r[key] for r in rows if r["offset"] == off and r[key] is not None]
        if vals:
            means[i] = float(np.mean(vals))
            counts[i] = len(vals)
    return offsets, means, counts


def main() -> None:
    all_episodes = []
    for task in STAGE2_TASK_NAMES:
        raw_path = SRC_DIR / task / "episodes_raw.json"
        if not raw_path.exists():
            continue
        for ep in json.loads(raw_path.read_text()):
            ep["task"] = task
            all_episodes.append(ep)

    rows_all = collect_profile_rows(all_episodes, only_close_chunks=False)
    rows_close = collect_profile_rows(all_episodes, only_close_chunks=True)
    print(f"total same-chunk consecutive-step pairs: all={len(rows_all)}, close-chunks-only={len(rows_close)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Chunk-internal temporal profile (E50, pooled across 18 episodes) vs. offset within committed chunk")

    off, dz_mean, dz_n = bin_by_offset(rows_all, "dz")
    axes[0, 0].plot(off, dz_mean, c="tab:blue", label="all chunks")
    off_c, dz_mean_c, _ = bin_by_offset(rows_close, "dz")
    axes[0, 0].plot(off_c, dz_mean_c, c="tab:red", label="reactive-close chunks only")
    axes[0, 0].axhline(0, color="gray", lw=0.5)
    axes[0, 0].set_xlabel("offset within chunk (steps since replan)")
    axes[0, 0].set_ylabel("mean commanded dz (m/step, -=descend)")
    axes[0, 0].set_title("Z displacement vs. chunk offset")
    axes[0, 0].legend(fontsize=8)

    off, dxy_mean, _ = bin_by_offset(rows_all, "dxy")
    axes[0, 1].plot(off, dxy_mean, c="tab:blue", label="all chunks")
    off_c, dxy_mean_c, _ = bin_by_offset(rows_close, "dxy")
    axes[0, 1].plot(off_c, dxy_mean_c, c="tab:red", label="reactive-close chunks only")
    axes[0, 1].set_xlabel("offset within chunk")
    axes[0, 1].set_ylabel("mean |commanded xy correction| (m/step)")
    axes[0, 1].set_title("XY correction magnitude vs. chunk offset")
    axes[0, 1].legend(fontsize=8)

    off, drot_mean, _ = bin_by_offset(rows_all, "drot")
    axes[1, 0].plot(off, drot_mean, c="tab:blue", label="all chunks")
    off_c, drot_mean_c, _ = bin_by_offset(rows_close, "drot")
    axes[1, 0].plot(off_c, drot_mean_c, c="tab:red", label="reactive-close chunks only")
    axes[1, 0].set_xlabel("offset within chunk")
    axes[1, 0].set_ylabel("mean |commanded rotation change| (rad/step, NOT wrap-corrected)")
    axes[1, 0].set_title("Rotation change vs. chunk offset")
    axes[1, 0].legend(fontsize=8)

    off, grip_mean, grip_n = bin_by_offset(rows_all, "grip")
    axes[1, 1].plot(off, grip_mean, c="tab:blue", label="all chunks")
    off_c, grip_mean_c, _ = bin_by_offset(rows_close, "grip")
    axes[1, 1].plot(off_c, grip_mean_c, c="tab:red", label="reactive-close chunks only")
    axes[1, 1].axhline(0.5, color="black", ls=":", lw=0.7, label="open/closed threshold")
    axes[1, 1].set_xlabel("offset within chunk")
    axes[1, 1].set_ylabel("mean commanded gripper raw value")
    axes[1, 1].set_title("Gripper command vs. chunk offset")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "chunk_temporal_profile.png", dpi=110)
    plt.close(fig)

    # Quantitative early-vs-late comparison (E10 reaches offsets 0-9 only, E5 reaches 0-4 only).
    early = [r for r in rows_all if r["offset"] <= 9]
    late = [r for r in rows_all if r["offset"] >= 40]
    mid = [r for r in rows_all if 10 <= r["offset"] <= 39]

    def summarize(name: str, rs: list[dict]) -> dict:
        return {
            "n": len(rs),
            "mean_dz": float(np.mean([r["dz"] for r in rs])) if rs else None,
            "mean_abs_dz": float(np.mean([abs(r["dz"]) for r in rs])) if rs else None,
            "mean_dxy": float(np.mean([r["dxy"] for r in rs])) if rs else None,
            "mean_grip": float(np.mean([r["grip"] for r in rs if r["grip"] is not None])) if rs else None,
        }

    summary = {
        "offset_0_9 (E10-reachable, E5 reaches only 0-4 of this)": summarize("early", early),
        "offset_10_39 (unreachable by E10/E5)": summarize("mid", mid),
        "offset_40_49 (late-chunk, unreachable by E10/E5)": summarize("late", late),
    }
    print(json.dumps(summary, indent=2))
    (OUT_DIR / "early_vs_late_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_DIR / 'chunk_temporal_profile.png'} and {OUT_DIR / 'early_vs_late_summary.json'}")


if __name__ == "__main__":
    main()
