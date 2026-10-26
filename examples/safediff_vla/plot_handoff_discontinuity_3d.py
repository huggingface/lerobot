#!/usr/bin/env python
"""3D trajectory plots of the LOCAL_GRASP -> POST_GRASP handoff window (+/-5 steps) for selected
episodes from `episodes_raw.json` (see `rollout_with_handoff_instrumentation.py` and
`handoff_discontinuity_metrics.py`). Pure plotting of existing rollout logs -- no re-simulation.

Each plot shows, in world frame xyz:
    - the actual TCP trajectory over the +/-5 step window (line)
    - the phase-transition step (X marker)
    - the last LOCAL_GRASP commanded xyz (derived: ee_pos + local_delta_xyz_m at the transition step)
    - the first POST_GRASP commanded xyz (the global decoder's first replanned action[0])
    - a dashed "handoff jump vector" between those two commanded points
    - the TargetPointHead's predicted_target_world and the GT target_pos at the transition step

Edit CASES below to add/remove episodes; each entry is
(label, base_dir, task, episode_ix, transition_step_ix).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_ep(base_dir: Path, task: str, episode_ix: int) -> dict:
    d = json.load(open(base_dir / task / "episodes_raw.json"))
    for ep in d:
        if ep["episode_ix"] == episode_ix:
            return ep
    raise KeyError(episode_ix)


def plot_case(out_dir: Path, label: str, base_dir: Path, task: str, episode_ix: int, T: int) -> None:
    ep = load_ep(base_dir, task, episode_ix)
    steps = ep["steps"]
    idx = {s["step_ix"]: s for s in steps}
    lo, hi = max(0, T - 5), min(len(steps) - 1, T + 5)
    window = [idx[k] for k in range(lo, hi + 1)]

    actual_xyz = np.array([s["ee_pos"] for s in window])
    sT = idx[T]
    last_local_xyz = np.array(sT["ee_pos"]) + np.array(sT["local_delta_xyz_m"])

    T1 = None
    for k in range(T + 1, min(T + 5, len(steps))):
        if idx[k].get("commanded_action_xyz_world") is not None:
            T1 = k
            break
    first_post_xyz = np.array(idx[T1]["commanded_action_xyz_world"]) if T1 is not None else None

    predicted_target = np.array(sT["predicted_target_world"]) if sT.get("predicted_target_world") else None
    gt_target = np.array(sT["target_pos"])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(actual_xyz[:, 0], actual_xyz[:, 1], actual_xyz[:, 2], "-o", color="tab:blue",
             label="actual TCP (T-5..T+5)", markersize=4, linewidth=1.5)
    ax.scatter(*sT["ee_pos"], color="black", s=120, marker="X", label=f"phase transition (step {T})")

    ax.scatter(*last_local_xyz, color="tab:green", s=90, marker="^",
               label="last LOCAL_GRASP commanded xyz")
    if first_post_xyz is not None:
        ax.scatter(*first_post_xyz, color="tab:red", s=90, marker="v",
                   label=f"first POST_GRASP commanded xyz (step {T1})")
        ax.plot([last_local_xyz[0], first_post_xyz[0]], [last_local_xyz[1], first_post_xyz[1]],
                [last_local_xyz[2], first_post_xyz[2]], "--", color="purple", linewidth=1,
                label="handoff jump vector")
    if predicted_target is not None:
        ax.scatter(*predicted_target, color="orange", s=90, marker="*", label="predicted_target (TargetPointHead)")
    ax.scatter(*gt_target, color="magenta", s=90, marker="D", label="GT target")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"{label}\ntask={task} ep={episode_ix} seed={ep['seed']} success={ep['success']}\nT={T}")
    ax.legend(loc="upper left", fontsize=7, bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


if __name__ == "__main__":
    ORIG = Path("outputs/eval/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_50k_handoff")
    RERUN = Path("outputs/eval/safediff_vla_grounded_grasp_v2_5_phase_aware_loss_50k_handoff_rerun")
    OUT = ORIG / "handoff_discontinuity_analysis"

    CASES = [
        ("book_seed1000_orig_abnormal", ORIG, "select_book", 0, 83),
        ("book_seed1001_orig_abnormal", ORIG, "select_book", 1, 325),
        ("book_seed1001_rerun_abnormal", RERUN, "select_book", 1, 78),
        ("drink_seed1000_orig_abnormal", ORIG, "select_drink", 0, 70),
        ("drink_seed1002_orig_abnormal", ORIG, "select_drink", 2, 262),
        ("toy_seed1001_orig_comparison", ORIG, "select_toy", 1, 122),
        ("mahjong_seed1000_orig_comparison", ORIG, "select_mahjong", 0, 57),
        ("poker_seed1002_orig_comparison", ORIG, "select_poker", 2, 107),
        ("poker_seed1001_orig_NORMAL", ORIG, "select_poker", 1, 82),
        ("mahjong_seed1001_rerun_NORMAL", RERUN, "select_mahjong", 1, 60),
        ("toy_seed1002_rerun_NORMAL", RERUN, "select_toy", 2, 54),
    ]
    for case in CASES:
        plot_case(OUT, *case)
