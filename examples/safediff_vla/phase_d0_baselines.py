#!/usr/bin/env python
"""PHASE D0 steps B-E: lightweight diagnostic baselines on `lerobot/vlabench_unified` grasp anchors.
Episode-level 5-fold CV (fixed seed) everywhere. No neural networks, no training of any VLA.

  A  task-mean xyz              B  task-mean xy (same xy error as A; reported for completeness)
  C  task + referent-mean       (referent unseen / <MIN_SUPPORT train eps -> task mean)
  Delta_ref = err(A) - err(C);  permutation null: shuffle referent labels within task, recompute C
  Q4 referent from anchor position: nearest-centroid on anchor xy, episode CV, vs chance and majority
  S  state-only: anchor predicted from the CURRENT EE state at frames where the target is supervised
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
SNAP = Path("/home/user_lerobot/.cache/huggingface/lerobot/hub/datasets--lerobot--vlabench_unified/snapshots/f61ad898199e746c20af3112f278cc86eb3ca3c4")
OUT = Path("outputs/eval/safediff_vla_phase_d0_training_data_audit")
MIN_SUPPORT, K, NPERM, H = 3, 5, 200, 50
TASKS = ["select_book", "select_poker", "select_mahjong", "select_drink", "select_fruit", "select_toy"]
rng = np.random.default_rng(0)


def folds(n):
    perm = rng.permutation(n); return [perm[i::K] for i in range(K)]


def cv_errors(A, refs, use_ref):
    """per-episode xyz error vector for predictor A (task mean) or C (referent mean) under K-fold CV."""
    n = len(A); pred = np.zeros_like(A); unseen = 0
    for te in folds(n):
        tr = np.setdiff1d(np.arange(n), te); mu = A[tr].mean(0)
        if use_ref:
            g = {}
            for r in np.unique(refs[tr]):
                m = tr[refs[tr] == r]
                if len(m) >= MIN_SUPPORT: g[r] = A[m].mean(0)
            for i in te:
                if refs[i] in g: pred[i] = g[refs[i]]
                else: pred[i] = mu; unseen += 1
        else:
            pred[te] = mu
    return A - pred, unseen


def main():
    df = pd.read_json(OUT / "episodes.json"); df = df[df.has_anchor]
    res = {}
    for task in TASKS:
        d = df[df.task == task].reset_index(drop=True)
        A = np.stack(d.anchor.to_numpy()); refs = d.referent.to_numpy(); n = len(d)
        cnt = d.referent.value_counts()
        r = {"n_episodes": n, "n_unique_instructions": int(d.instruction.nunique()), "n_unique_referents": int(cnt.size),
             "episodes_per_referent_min_med_max": [int(cnt.min()), int(cnt.median()), int(cnt.max())],
             "anchor_mean_xyz_robot_frame": A.mean(0).round(3).tolist(), "anchor_std_xyz": A.std(0).round(3).tolist()}
        eA, _ = cv_errors(A, refs, False); eC, unseen = cv_errors(A, refs, True)
        xy = lambda e: np.linalg.norm(e[:, :2], axis=1); xyz = lambda e: np.linalg.norm(e, axis=1)
        r["A_task_mean"] = {"xyz_mean": float(xyz(eA).mean()), "xyz_median": float(np.median(xyz(eA))), "xy_mean": float(xy(eA).mean()), "xy_median": float(np.median(xy(eA)))}
        r["C_task+referent_mean"] = {"xyz_mean": float(xyz(eC).mean()), "xyz_median": float(np.median(xyz(eC))), "xy_mean": float(xy(eC).mean()), "xy_median": float(np.median(xy(eC))), "frac_fallback_to_task_mean": unseen / n}
        for tag, f in (("xy", xy), ("xyz", xyz)):
            r[f"Delta_ref_{tag}_mean"] = float(f(eA).mean() - f(eC).mean())
            r[f"Delta_ref_{tag}_rel"] = float((f(eA).mean() - f(eC).mean()) / f(eA).mean())
        null = []
        for _ in range(NPERM):
            e, _ = cv_errors(A, rng.permutation(refs), True); null.append(xy(eA).mean() - xy(e).mean())
        null = np.array(null); obs = r["Delta_ref_xy_mean"]
        r["Delta_ref_xy_permutation_null_mean_std"] = [float(null.mean()), float(null.std())]
        r["Delta_ref_xy_perm_p_one_sided"] = float((null >= obs).mean())
        # variance decomposition of anchor xy
        mu = A[:, :2].mean(0); tot = ((A[:, :2] - mu) ** 2).sum(1).mean()
        within = np.mean([((A[refs == q][:, :2] - A[refs == q][:, :2].mean(0)) ** 2).sum(1).mean() for q in np.unique(refs)])
        between = float(sum((refs == q).sum() * ((A[refs == q][:, :2].mean(0) - mu) ** 2).sum() for q in np.unique(refs)) / n)
        r["xy_variance_total_cm2"] = float(tot * 1e4); r["xy_between_referent_share_eta2"] = float(between / tot)
        k = len(np.unique(refs)); r["eta2_expected_under_no_effect(~(R-1)/(N-1))"] = float((k - 1) / (n - 1))
        # Q4: referent from anchor xy, nearest-centroid CV (referents with >= 2*MIN_SUPPORT episodes only)
        keep = np.isin(refs, cnt.index[cnt >= 2 * MIN_SUPPORT]); idx = np.where(keep)[0]
        if len(np.unique(refs[idx])) >= 2:
            correct = 0; tot_n = 0
            for te in folds(len(idx)):
                te = idx[te]; tr = np.setdiff1d(idx, te)
                cents = {q: A[tr][refs[tr] == q][:, :2].mean(0) for q in np.unique(refs[tr])}
                for i in te:
                    tot_n += 1; correct += min(cents, key=lambda q: np.linalg.norm(A[i, :2] - cents[q])) == refs[i]
            maj = cnt[cnt >= 2 * MIN_SUPPORT].max() / cnt[cnt >= 2 * MIN_SUPPORT].sum()
            r["Q4_referent_from_anchor_xy"] = {"acc": correct / tot_n, "chance_1_over_R": 1 / len(np.unique(refs[idx])), "majority": float(maj), "R": int(len(np.unique(refs[idx])))}
        res[task] = r
    (OUT / "baselines.json").write_text(json.dumps(res, indent=1))

    # ---- S: state-only predictor at SUPERVISED frames (valid target: frame f in (t_close-H, t_close]) ----
    frames = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state", "action"]) for p in sorted((SNAP / "data").rglob("*.parquet"))])
    eps = {int(e): g.sort_values("frame_index") for e, g in frames.groupby("episode_index")}
    S = {}
    for task in TASKS:
        d = df[df.task == task]; rows = []
        for _, e in d.iterrows():
            g = eps[int(e.episode)]; st = np.stack(g["observation.state"].to_numpy()); tc = int(e.t_close); anc = np.array(e.anchor)
            for f in range(max(0, tc - H + 1), tc + 1):
                rows.append((int(e.episode), tc - f, np.linalg.norm(st[f, :2] - anc[:2]), np.linalg.norm(st[f, :3] - anc), f))
        a = pd.DataFrame(rows, columns=["ep", "steps_to_close", "ee_to_anchor_xy", "ee_to_anchor_xyz", "frame"])
        bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
        S[task] = {f"steps_to_close_{lo}-{hi}": {"ee_as_anchor_xy_mean_cm": float(a[(a.steps_to_close >= lo) & (a.steps_to_close < hi)].ee_to_anchor_xy.mean() * 100),
                                                   "ee_as_anchor_xyz_mean_cm": float(a[(a.steps_to_close >= lo) & (a.steps_to_close < hi)].ee_to_anchor_xyz.mean() * 100)} for lo, hi in bins}
        S[task]["all_supervised_frames"] = {"ee_as_anchor_xy_mean_cm": float(a.ee_to_anchor_xy.mean() * 100), "ee_as_anchor_xyz_mean_cm": float(a.ee_to_anchor_xyz.mean() * 100),
                                            "n_frames": int(len(a)), "task_mean_xy_baseline_cm": res[task]["A_task_mean"]["xy_mean"] * 100,
                                            "frac_supervised_frames_with_frame_index_lt_15": float((a.frame < 15).mean())}
    (OUT / "state_only_baseline.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({t: {k: v for k, v in r.items() if k in ("n_episodes", "n_unique_referents", "episodes_per_referent_min_med_max", "anchor_std_xyz")} for t, r in res.items()}, indent=0))


if __name__ == "__main__":
    main()
