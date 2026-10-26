#!/usr/bin/env python
"""PHASE D0 step A: per-episode grasp anchor + referent extraction from `lerobot/vlabench_unified` (read-only).

Anchor = EXACTLY GroundedGrasp-v2's supervision definition (`utils.find_grasp_target`): the commanded xyz
(`action[..., :3]`, robot frame) at the first open->close transition of the gripper (action: open iff > 0.5;
state: open iff < 0.5, opposite polarity), requiring the state to be open when the window starts. Applied
episode-wide (equivalent for any window that contains the transition and starts pre-grasp).
Also records, per episode, the transition step and per-frame data needed to quantify WHERE in the episode the
target head is supervised (a window only has a valid target if the transition lies within `action_horizon`=50).
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import STAGE2_TASK_NAMES, classify_task  # noqa: E402

SNAP = Path("/home/user_lerobot/.cache/huggingface/lerobot/hub/datasets--lerobot--vlabench_unified/snapshots/f61ad898199e746c20af3112f278cc86eb3ca3c4")
OUT = Path("outputs/eval/safediff_vla_phase_d0_training_data_audit"); OUT.mkdir(parents=True, exist_ok=True)
REFERENT_RE = {
    "select_poker": r"poker (.+)$", "select_book": r"book (.+)$", "select_mahjong": r"mahjong of (.+)$",
    "select_fruit": r"Put the (.+?) into", "select_toy": r"Put the (.+?) into", "select_drink": r"Take out the (.+?) from",
}
H = 50


def main():
    tasks = pd.read_parquet(SNAP / "meta/tasks.parquet")
    idx2task = {int(r.task_index): s for s, r in tasks.iterrows()}
    frames = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state", "action", "task_index"])
                        for p in sorted((SNAP / "data").rglob("*.parquet"))], ignore_index=True)
    rows = []
    for ep, g in frames.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index")
        tstr = idx2task[int(g["task_index"].iloc[0])]
        task = classify_task(tstr)
        if task not in STAGE2_TASK_NAMES:
            continue
        st = np.stack(g["observation.state"].to_numpy()); ac = np.stack(g["action"].to_numpy())
        act_open = ac[:, 6] > 0.5; st_open = st[:, 6] < 0.5
        seq_open = np.concatenate([[st_open[0]], act_open])
        trans = np.where(seq_open[:-1] & ~seq_open[1:])[0]
        referent = re.search(REFERENT_RE[task], tstr).group(1).strip()
        rec = {"episode": int(ep), "task": task, "instruction": tstr, "referent": referent, "n_frames": len(g),
               "start_state_open": bool(st_open[0]), "has_anchor": bool(len(trans) and st_open[0])}
        if rec["has_anchor"]:
            t = int(trans[0])
            rec.update(t_close=t, anchor=ac[t, :3].tolist(), ee_at_close=st[min(t, len(st) - 1), :3].tolist(),
                       ee_start=st[0, :3].tolist())
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_json(OUT / "episodes.json", orient="records", indent=0)
    print(df.groupby("task").agg(n=("episode", "size"), has_anchor=("has_anchor", "mean"), t_close_med=("t_close", "median"),
                                 t_close_min=("t_close", "min"), n_frames_med=("n_frames", "median")))
    # supervision coverage: fraction of frames whose 50-step window holds the transition (valid target) and the
    # frame-index profile -- is the step-0 frame ever supervised?
    cov = {}
    for task, d in df[df.has_anchor].groupby("task"):
        tc = d.t_close.to_numpy(); nf = d.n_frames.to_numpy()
        valid_total = (np.minimum(tc + 1, nf)).clip(min=0)  # frames f<=t_close ... valid iff t_close - f < H and f<=t_close
        valid_frames = np.minimum(tc + 1, H) # frames f in (t_close-H, t_close]
        cov[task] = {"frac_frames_with_valid_target": float(valid_frames.sum() / nf.sum()),
                     "frac_episodes_where_frame0_valid(t_close<50)": float(np.mean(tc < H)),
                     "frac_episodes_t_close_lt_100": float(np.mean(tc < 100)), "t_close_pctiles_5_50_95": np.percentile(tc, [5, 50, 95]).tolist()}
    (OUT / "supervision_coverage.json").write_text(json.dumps(cov, indent=1)); print(json.dumps(cov, indent=1))


if __name__ == "__main__":
    main()
