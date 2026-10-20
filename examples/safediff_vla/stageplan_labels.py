#!/usr/bin/env python
"""StagePlan training-label extractor + audit (TRAIN data only: `lerobot/vlabench_unified` expert actions).

Task-agnostic: labels come from trajectory EVENTS only (gripper close/open transitions, motion dwell, polyline corners).
No task name, no simulator state, no world-z offset rules. Task family is used ONLY to group the audit report.

Stages (per episode; `a` = expert action [x y z rx ry rz g], g=1 open / 0 closed, 10 Hz):
  tc  = first close transition (g: 1 -> 0). t1 = last moving frame before the pre-close dwell (speed < DWELL_SPEED).
  P1  = a[t1]            GRASP pose: expert commanded EE pose when it arrives and stops, immediately before CLOSE.
  P0  = a[j]             PREGRASP: last polyline vertex of the pre-grasp path (RDP, CORNER_EPS) whose arc-distance back from P1
                         is >= MIN_STAGE_ARC (skips tiny hops near the grasp). If the path has no such vertex (single straight
                         run) P0 = the first action pose (flag p0_is_start).
  P2  = POSTGRASP/EXTRACTION: on the post-close path (until the first open transition or episode end) the first polyline
                         vertex with arc-distance >= MIN_STAGE_ARC from P1, else the last pre-open/end pose.
  P3  = RELEASE/GOAL: a[t_open-1], the commanded pose just before the first open transition after tc. release_valid = 1 iff
                         such a transition exists; otherwise P3 is masked.
Global constants (one set, set once from the TRAIN trajectory statistics printed by the audit; never tuned on eval):
  DWELL_SPEED = 2 mm/frame (expert speed is ~10 mm/frame); CORNER_EPS = 1 cm; MIN_STAGE_ARC = 5 cm (half the modal 10 cm
  expert prepare-to-grasp run measured in the audit).
Usage: uv run python examples/safediff_vla/stageplan_labels.py   # writes outputs/stageplan/labels.pkl + audit
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DWELL_SPEED = 0.002
CORNER_EPS = 0.01
MIN_STAGE_ARC = 0.05
OUT = Path("outputs/stageplan")


def rdp(P: np.ndarray, eps: float) -> list[int]:
    if len(P) < 3:
        return [0, len(P) - 1]
    a, b = P[0], P[-1]; ab = b - a; n = np.linalg.norm(ab)
    d = np.linalg.norm(np.cross(P - a, ab), axis=1) / n if n > 1e-9 else np.linalg.norm(P - a, axis=1)
    i = int(np.argmax(d))
    if d[i] > eps:
        left, right = rdp(P[: i + 1], eps), rdp(P[i:], eps)
        return left + [x + i for x in right[1:]]
    return [0, len(P) - 1]


def extract(a: np.ndarray) -> dict:
    """a: [T,7] expert actions. Returns stage poses (xyz+euler), frame indices and validity flags."""
    T = len(a); g = a[:, 6] > 0.5; tr = np.where(g[1:] != g[:-1])[0] + 1
    out = {"valid": False, "reason": ""}
    if not g[0]: out["reason"] = "starts closed"; return out
    closes = [t for t in tr if not g[t]]
    if not closes: out["reason"] = "no close"; return out
    tc = int(closes[0]); sp = np.r_[0, np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1)]
    t1 = tc - 1
    while t1 > 0 and sp[t1] < DWELL_SPEED: t1 -= 1
    if t1 < 5: out["reason"] = "no approach"; return out
    P = a[: t1 + 1, :3]; s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    v = rdp(P, CORNER_EPS); back = s[-1] - s[v]
    cand = [vi for vi, b in zip(v, back) if b >= MIN_STAGE_ARC]
    j0 = cand[-1] if cand else 0
    opens = [t for t in tr if g[t] and t > tc]; t_open = int(opens[0]) if opens else None
    end = (t_open - 1) if t_open is not None else T - 1
    post = a[tc: end + 1, :3]
    ps = np.r_[0, np.cumsum(np.linalg.norm(np.diff(post, axis=0), axis=1))] if len(post) > 1 else np.zeros(1)
    vp = rdp(post, CORNER_EPS) if len(post) > 2 else [0, len(post) - 1]
    # first post-grasp vertex far enough from the grasp pose (arc from the first MOVING post-close frame)
    cand2 = [tc + vi for vi in vp if ps[vi] >= MIN_STAGE_ARC]
    j2 = cand2[0] if cand2 else end
    out.update(valid=True, tc=tc, t1=t1, j0=j0, j2=int(j2), t_open=t_open, release_valid=t_open is not None, p0_is_start=(j0 == 0),
               P0=a[j0, :6].copy(), P1=a[t1, :6].copy(), P2=a[j2, :6].copy(), P3=a[t_open - 1, :6].copy() if t_open is not None else np.full(6, np.nan),
               arc01=float(s[-1] - s[j0]), arc12=float(np.linalg.norm(a[j2, :3] - a[t1, :3])), T=T,
               arc23=float(np.linalg.norm(a[t_open - 1, :3] - a[j2, :3])) if t_open is not None else np.nan)
    return out


def load_df() -> pd.DataFrame:
    p = OUT / "df6.pkl"
    return pd.read_pickle(p)


def main() -> None:
    df = load_df(); labels = {}
    for ep, e in df.groupby("episode_index"):
        a = np.stack(e.action.values).astype(np.float64); lab = extract(a); lab["fam"] = e.fam.iloc[0]; lab["task_index"] = int(e.task_index.iloc[0]); labels[int(ep)] = lab
    pickle.dump(labels, open(OUT / "labels.pkl", "wb"))
    R = pd.DataFrame([{k: v for k, v in l.items() if k in ("fam", "valid", "reason", "p0_is_start", "release_valid", "arc01", "arc12", "arc23", "T", "tc", "j2")} for l in labels.values()])
    print("episodes", len(R), "invalid", int((~R.valid).sum()), R[~R.valid].groupby(["fam", "reason"]).size().to_dict())
    V = R[R.valid]
    print(V.groupby("fam").agg(n=("valid", "size"), release_valid=("release_valid", "mean"), p0_is_start=("p0_is_start", "mean"),
                               d01_med=("arc01", lambda x: x.median() * 100), d01_p10=("arc01", lambda x: x.quantile(.1) * 100), d01_p90=("arc01", lambda x: x.quantile(.9) * 100),
                               d12_med=("arc12", lambda x: x.median() * 100), d12_p90=("arc12", lambda x: x.quantile(.9) * 100), d23_med=("arc23", lambda x: np.nanmedian(x) * 100)).round(2).to_string())
    print("\nrotation (sin/cos euler) spread of P1 per family: std of (sin,cos) per axis")
    for f, ls in V.groupby("fam"):
        P1 = np.array([labels[i]["P1"] for i, l in labels.items() if l["valid"] and l["fam"] == f])
        sc = np.concatenate([np.sin(P1[:, 3:]), np.cos(P1[:, 3:])], 1)
        print(f"  {f:16s} std={sc.std(0).round(2).tolist()}  P1 xyz std(cm)={(P1[:, :3].std(0) * 100).round(1).tolist()}")


if __name__ == "__main__":
    sys.exit(main())


def split_episodes(df: pd.DataFrame, eval_split: float = 0.1) -> tuple[list[int], list[int]]:
    """Same rule as `lerobot.datasets.factory.make_train_eval_datasets`: per task string, the LAST ceil(n*eval_split)
    episodes (ascending episode_index) are held out; everything else is TRAIN. The eval seeds 3000-3019 are simulator
    seeds, unrelated to these dataset episodes."""
    import math
    train, val = [], []
    for _, g in df.groupby("task_index"):
        eps = sorted(g.episode_index.unique().tolist()); n_eval = math.ceil(len(eps) * eval_split)
        train += eps[: len(eps) - n_eval]; val += eps[len(eps) - n_eval:]
    return sorted(train), sorted(val)


def controller_constants(df: pd.DataFrame, labels: dict, train_eps: list[int]) -> dict:
    """ONE global set of controller constants measured on TRAIN expert trajectories (never on eval)."""
    from scipy.spatial.transform import Rotation as Rot
    vx, vr, h_pre, h_post, h_open_pre, h_open_post, segN = [], [], [], [], [], [], []
    for ep in train_eps:
        e = df[df.episode_index == ep]; a = np.stack(e.action.values).astype(np.float64); l = labels[ep]
        if not l["valid"]: continue
        sp = np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1)
        R = Rot.from_euler("xyz", a[:, 3:6]); ang = (R[:-1].inv() * R[1:]).magnitude()
        upto = (l["t_open"] or l["T"]) - 1
        mv = sp[:upto] > DWELL_SPEED; vx += sp[:upto][mv].tolist()
        rot_mv = (ang[:upto] > 1e-3) & (ang[:upto] < 0.3); vr += ang[:upto][rot_mv].tolist()   # <0.3 rad/frame excludes euler-branch flips
        tc, t1 = l["tc"], l["t1"]; h_pre.append(tc - t1)
        k = tc
        while k < len(sp) and sp[k] < DWELL_SPEED: k += 1
        h_post.append(k - tc)
        if l["t_open"] is not None:
            to = l["t_open"]; q = to - 1
            while q > 0 and sp[q - 1] < DWELL_SPEED: q -= 1
            h_open_pre.append(to - q)
            k = to
            while k < len(sp) and sp[k] < DWELL_SPEED: k += 1
            h_open_post.append(k - to)
        for (s0, s1) in ((l["j0"], l["t1"]), (l["t1"], l["j2"])): segN.append(s1 - s0)
    q = lambda x, p: float(np.percentile(x, p))
    return {"v_xyz": q(vx, 50), "v_rot": q(vr, 50), "hold_before_close": int(round(q(h_pre, 50))), "hold_after_close": int(round(q(h_post, 50))),
            "hold_before_open": int(round(q(h_open_pre, 50))), "hold_after_open": int(round(q(h_open_post, 50))),
            "seg_min": max(2, int(q(segN, 1))), "seg_max": int(q(segN, 99)), "n_train_episodes": len(train_eps)}
