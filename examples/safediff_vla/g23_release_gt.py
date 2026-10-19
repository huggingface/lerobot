#!/usr/bin/env python
"""G23 STEP 1: release ground truth from the expert demonstrations (lerobot/vlabench_unified, 6 tasks, all 500 episodes each; parquet only). Gripper action channel (index 6, raw units; <=0.5 = closed command, >0.5 = open command, the same convention as the rest of this branch). Per episode: grasp (first open->close command) time t_close, whether a later closed->open command (release) exists, its time, episode length, EE position at grasp / release (robot frame, from observation.state), release-grasp displacement. The dataset has no object/container poses, so EE-goal / EE-object relations are summarised through the EE displacement from the grasp point and the spread of the release EE position across episodes (a fixed goal would give a tight spread)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g23"); OUT.mkdir(parents=True, exist_ok=True); TASKS = ["select_poker", "select_mahjong", "select_toy", "select_fruit", "select_drink", "select_book"]


def main():
    d = LeRobotDataset("lerobot/vlabench_unified"); fi = d.meta.episodes["dataset_from_index"]; ti = d.meta.episodes["dataset_to_index"]; E = pd.read_json("outputs/eval/safediff_vla_phase_d0_training_data_audit/episodes.json"); rows = []
    for task in TASKS:
        for r in E[E.task == task].itertuples():
            ep = int(r.episode); sl = d.hf_dataset[int(fi[ep]):int(ti[ep])]; A = np.array(sl["action"]); S = np.array(sl["observation.state"]); g = A[:, 6]; closed = g <= 0.5; n = len(g)
            tc = int(np.argmax(closed)) if closed.any() else None; rel = None
            if tc is not None:
                after = np.where(~closed[tc:])[0]; rel = int(tc + after[0]) if len(after) else None
            rows.append({"task": task, "ep": ep, "n": n, "tc": tc, "t_rel": rel, "ee_grasp": S[tc, :3] if tc is not None else None, "ee_rel": S[rel, :3] if rel is not None else None, "ee_end": S[-1, :3], "g_end_closed": bool(closed[-1]), "ee_zmax_after_grasp": float(S[tc:, 2].max()) if tc is not None else None})
    df = pd.DataFrame(rows); df.to_pickle(OUT / "release_gt.pkl")
    print(f"{'task':16s}{'eps':>5s}{'len mean':>9s}{'close':>7s}{'release exists':>15s}{'t_close':>9s}{'t_release':>10s}{'hold steps':>11s}{'ends closed':>12s}{'rel-grasp |d| cm':>17s}{'rel dz cm':>10s}{'rel xy std cm':>14s}{'rel z std cm':>13s}{'grasp xy std':>13s}")
    for task in TASKS:
        x = df[df.task == task]; c = x[x.tc.notna()]; r = c[c.t_rel.notna()]
        if len(r): dd = np.array([np.linalg.norm(a - b) for a, b in zip(r.ee_rel, r.ee_grasp)]) * 100; dz = np.array([a[2] - b[2] for a, b in zip(r.ee_rel, r.ee_grasp)]) * 100; rx = np.stack(r.ee_rel.values)
        print(f"{task:16s}{len(x):5d}{x.n.mean():9.0f}{len(c):7d}{len(r):>10d}/{len(c):<4d}{c.tc.mean():9.0f}{(r.t_rel.mean() if len(r) else float('nan')):10.0f}{((r.t_rel - r.tc).mean() if len(r) else float('nan')):11.0f}{c.g_end_closed.mean():12.2f}{(dd.mean() if len(r) else float('nan')):17.1f}{(dz.mean() if len(r) else float('nan')):10.1f}{(np.linalg.norm(rx[:, :2].std(0)) * 100 if len(r) else float('nan')):14.1f}{(rx[:, 2].std() * 100 if len(r) else float('nan')):13.1f}{np.linalg.norm(np.stack(c.ee_grasp.values)[:, :2].std(0)) * 100:13.1f}")
    print("\n(release = a closed->open command after the first grasp; 'ends closed' = the demonstration finishes with the gripper still commanded closed)")


if __name__ == "__main__":
    main()
