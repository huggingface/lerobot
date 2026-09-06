#!/usr/bin/env python
"""Analyze LIBERO / LIBERO-Safety action semantics.

Creates regression/correlation metrics comparing stored `action` to observed
per-step physical deltas and velocities for translation and rotation.

Usage examples:
  uv run python -m lerobot.scripts.analyze_libero_action_semantics \
      --dataset LIBERO-Safety/libero_safety --dataset-type libero_safety \
      --max-transitions 5000

  uv run python -m lerobot.scripts.analyze_libero_action_semantics \
      --dataset lerobot/libero_10_image --dataset-type libero --max-transitions 5000
"""
from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.utils.rotation import Rotation


def safe_asarray(x):
    return np.asarray(x, dtype=float)


def pearsonr(x, y):
    # fallback implementation without scipy
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return float("nan")
    xm = x.mean()
    ym = y.mean()
    num = np.sum((x - xm) * (y - ym))
    den = math.sqrt(np.sum((x - xm) ** 2) * np.sum((y - ym) ** 2))
    return float(num / den) if den > 0 else float("nan")


def linear_regression(x, y, fit_intercept=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return dict(slope=float("nan"), intercept=float("nan"), r2=float("nan"))
    if fit_intercept:
        A = np.vstack([x, np.ones_like(x)]).T
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = float(w[0]), float(w[1])
        y_pred = slope * x + intercept
    else:
        # slope only through origin
        denom = np.sum(x * x)
        slope = float(np.sum(x * y) / denom) if denom > 0 else float("nan")
        intercept = 0.0
        y_pred = slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(slope=slope, intercept=intercept, r2=r2, y_pred=y_pred)


def metrics_for_pair(x, y, exclude_small=1e-6):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    corr = pearsonr(x, y)
    reg = linear_regression(x, y, fit_intercept=True)
    mae = float(np.mean(np.abs(y - reg["slope"] * x - reg["intercept"])) )
    medae = float(np.median(np.abs(y - reg["slope"] * x - reg["intercept"])))
    # ratio y/x with small-action exclusion
    ratio_mask = np.abs(x) > exclude_small
    ratios = (y[ratio_mask] / x[ratio_mask]) if ratio_mask.any() else np.array([])
    ratio_stats = dict(median=float(np.median(ratios)) if ratios.size else float("nan"), p10=float(np.quantile(ratios, 0.1)) if ratios.size else float("nan"), p90=float(np.quantile(ratios, 0.9)) if ratios.size else float("nan"))
    return dict(corr=corr, slope=reg["slope"], intercept=reg["intercept"], r2=reg["r2"], mae=mae, medae=medae, ratio_stats=ratio_stats)


def rot_delta_from_states(s_t, s_t1):
    """Compute rotvec delta between two state vectors.

    Heuristics:
    - If state has length >= 7 and norm(state[3:7]) ~ 1 -> treat as quaternion [x,y,z,w]
    - Else treat state[3:6] as axis-angle (rotvec)
    Returns rotvec (3,)
    """
    s_t = np.asarray(s_t)
    s_t1 = np.asarray(s_t1)
    # prefer quaternion if present
    if s_t.shape[0] >= 7 and s_t1.shape[0] >= 7:
        q1 = s_t[3:7]
        q2 = s_t1[3:7]
        nq1 = np.linalg.norm(q1)
        nq2 = np.linalg.norm(q2)
        if nq1 > 0.9 and nq2 > 0.9:
            try:
                R1 = Rotation.from_quat(q1)
                R2 = Rotation.from_quat(q2)
            except Exception:
                # fall back to rotvec
                pass
            else:
                R_delta = R1.inv() * R2
                return R_delta.as_rotvec()
    # otherwise assume axis-angle in indices 3:6
    rot1 = s_t[3:6]
    rot2 = s_t1[3:6]
    R1 = Rotation.from_rotvec(rot1)
    R2 = Rotation.from_rotvec(rot2)
    R_delta = R1.inv() * R2
    return R_delta.as_rotvec()


def analyze_pairs(pairs, fps: int, out_dir: Path, prefix: str, velocity_compare=False):
    # pairs: list of dict with keys: pos_delta (3,), rot_delta (3,), action_trans (3,), action_rot (3,)
    N = len(pairs)
    pos_d = np.stack([p["pos_delta"] for p in pairs], axis=0)
    act_t = np.stack([p["action_trans"] for p in pairs], axis=0)
    rot_d = np.stack([p["rot_delta"] for p in pairs], axis=0)
    act_r = np.stack([p["action_rot"] for p in pairs], axis=0)

    results = {"translation": {}, "rotation": {}}

    # Translation per-axis and flattened
    axes = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        res = metrics_for_pair(act_t[:, i], pos_d[:, i])
        results["translation"][ax] = res
    # flattened
    res_flat = metrics_for_pair(act_t.flatten(), pos_d.flatten())
    results["translation"]["all"] = res_flat

    # Rotation
    for i, ax in enumerate(axes):
        res = metrics_for_pair(act_r[:, i], rot_d[:, i])
        results["rotation"][ax] = res
    res_flat_r = metrics_for_pair(act_r.flatten(), rot_d.flatten())
    results["rotation"]["all"] = res_flat_r

    # best-fit scalar k for translation / rotation (through origin)
    def best_k(A, B):
        num = np.sum(A * B)
        den = np.sum(A * A)
        return float(num / den) if den > 0 else float("nan")

    k_trans = best_k(act_t.flatten(), pos_d.flatten())
    k_rot = best_k(act_r.flatten(), rot_d.flatten())

    # Evaluate hypotheses k=1, k=2 and best-fit
    def eval_k(A, B, k):
        pred = A * k
        mae = float(np.mean(np.abs(B - pred)))
        ss_res = np.sum((B - pred) ** 2)
        ss_tot = np.sum((B - B.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        corr = pearsonr(A.flatten(), B.flatten())
        return dict(k=k, mae=mae, r2=r2, corr=corr)

    hyp_k = {"k=1": eval_k(act_t, pos_d, 1.0), "k=2": eval_k(act_t, pos_d, 2.0), "best": eval_k(act_t, pos_d, k_trans)}
    hyp_k_r = {"k=1": eval_k(act_r, rot_d, 1.0), "k=2": eval_k(act_r, rot_d, 2.0), "best": eval_k(act_r, rot_d, k_rot)}

    summary = dict(
        translation_results=results["translation"],
        rotation_results=results["rotation"],
        k_trans=k_trans,
        k_rot=k_rot,
        hyp_trans=hyp_k,
        hyp_rot=hyp_k_r,
    )

    # velocity compare if requested
    if velocity_compare:
        lin_vel = pos_d * fps
        ang_vel = rot_d * fps
        k_lin = best_k(act_t.flatten(), lin_vel.flatten())
        k_ang = best_k(act_r.flatten(), ang_vel.flatten())
        summary.update(dict(k_lin=k_lin, k_ang=k_ang))

    # plots: scatter action vs observed delta for each axis
    out_dir.mkdir(parents=True, exist_ok=True)
    def scatter_and_save(A, B, title, fname, reference_lines=None):
        plt.figure(figsize=(5, 5))
        plt.scatter(A, B, s=2, alpha=0.4)
        mn = min(np.nanmin(A), np.nanmin(B))
        mx = max(np.nanmax(A), np.nanmax(B))
        rng = mx - mn if mx > mn else 1.0
        xs = np.linspace(mn - 0.05 * rng, mx + 0.05 * rng, 100)
        if reference_lines:
            for label, slope in reference_lines:
                plt.plot(xs, slope * xs, label=label)
            plt.legend()
        plt.xlabel("action")
        plt.ylabel("observed_delta")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / fname)
        plt.close()

    # translation axes
    for i, ax in enumerate(axes):
        scatter_and_save(act_t[:, i], pos_d[:, i], f"{prefix} trans {ax}", f"{prefix}_trans_{ax}.png", reference_lines=[("y=x", 1.0), ("y=2x", 2.0)])
    scatter_and_save(act_t.flatten(), pos_d.flatten(), f"{prefix} trans all", f"{prefix}_trans_all.png", reference_lines=[("y=x", 1.0), ("y=2x", 2.0)])

    # rotation axes
    for i, ax in enumerate(axes):
        scatter_and_save(act_r[:, i], rot_d[:, i], f"{prefix} rot {ax}", f"{prefix}_rot_{ax}.png", reference_lines=[("y=x", 1.0), ("y=2x", 2.0)])
    scatter_and_save(act_r.flatten(), rot_d.flatten(), f"{prefix} rot all", f"{prefix}_rot_all.png", reference_lines=[("y=x", 1.0), ("y=2x", 2.0)])

    return summary


def analyze_libero_safety(repo_id: str, max_transitions: int, cache_dir: str | None, out_dir: Path):
    from lerobot.datasets.adapters.libero_safety_v21 import LiberoSafetyV21Dataset

    ds = LiberoSafetyV21Dataset(repo_id=repo_id, revision="main", cache_dir=cache_dir)
    fps = int(ds.meta.fps)
    pairs = []
    total = 0
    for row_meta in ds.episode_rows:
        ep = int(row_meta["episode_index"])
        rows = ds._rows(ep)
        # rows are sequential frames in episode
        for i in range(len(rows) - 1):
            r0 = rows[i]
            r1 = rows[i + 1]
            s0 = np.asarray(r0["observation.state"])
            s1 = np.asarray(r1["observation.state"])
            a0 = np.asarray(r0["actions"])  # shape (7,)
            pos_delta = s1[0:3] - s0[0:3]
            rot_delta = rot_delta_from_states(s0, s1)
            pairs.append({"pos_delta": pos_delta, "rot_delta": rot_delta, "action_trans": a0[0:3], "action_rot": a0[3:6]})
            total += 1
            if total >= max_transitions:
                break
        if total >= max_transitions:
            break
    summary = analyze_pairs(pairs, fps, out_dir, prefix="libero_safety", velocity_compare=True)
    return dict(dataset=repo_id, fps=fps, n_transitions=len(pairs), summary=summary)


def analyze_libero_standard(repo_id: str, max_transitions: int, cache_dir: str | None, out_dir: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=repo_id, root=cache_dir, episodes=None, return_uint8=True)
    fps = int(ds.fps)
    pairs = []
    total = 0
    # iterate frames via reader.hf_dataset to safely access raw rows
    hf = ds.hf_dataset
    for idx in range(len(hf) - 1):
        row0 = hf[idx]
        row1 = hf[idx + 1]
        # ensure same episode
        if int(row0.get("episode_index", -1)) != int(row1.get("episode_index", -2)):
            continue
        s0 = np.asarray(row0["observation.state"])
        s1 = np.asarray(row1["observation.state"])
        # actions might be stored under 'action' or 'actions'
        a0 = np.asarray(row0.get("action", row0.get("actions")))
        pos_delta = s1[0:3] - s0[0:3]
        rot_delta = rot_delta_from_states(s0, s1)
        pairs.append({"pos_delta": pos_delta, "rot_delta": rot_delta, "action_trans": a0[0:3], "action_rot": a0[3:6]})
        total += 1
        if total >= max_transitions:
            break
    summary = analyze_pairs(pairs, fps, out_dir, prefix="libero", velocity_compare=True)
    return dict(dataset=repo_id, fps=fps, n_transitions=len(pairs), summary=summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset-type", choices=("libero_safety", "libero"), default="libero_safety")
    p.add_argument("--max-transitions", type=int, default=5000)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out-dir", default="outputs/analysis_libero_action_semantics")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if args.dataset_type == "libero_safety":
        report = analyze_libero_safety(args.dataset, args.max_transitions, args.cache_dir, out_dir)
    else:
        report = analyze_libero_standard(args.dataset, args.max_transitions, args.cache_dir, out_dir)

    print("Analysis complete:\n")
    print(f"Dataset: {report['dataset']}")
    print(f"FPS: {report['fps']}")
    print(f"Transitions used: {report['n_transitions']}")
    print("Summary keys:\n", ", ".join(report['summary'].keys()))
    # Save a summary file
    (out_dir / "summary.txt").write_text(str(report))


if __name__ == "__main__":
    main()
