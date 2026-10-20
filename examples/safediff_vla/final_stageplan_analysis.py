#!/usr/bin/env python
"""Per-episode stage funnel + tables for the FINAL StagePlan closed-loop evaluation (`final_stageplan_eval.py` outputs). Privileged
snapshots are used for SCORING only. `verified.json` (task/seed -> bool) holds the video-verified physical success.

Funnel: plan(P1 within 5cm of keypoint) -> pregrasp (GRASP phase reached) -> grasp (CLOSE reached, EE within 3cm of keypoint)
        -> grasped -> postgrasp (lift >= 5cm while held) -> transport (place tasks: object within 6cm XY of goal) -> release -> outcome.
Usage: uv run python examples/safediff_vla/final_stageplan_analysis.py [--root outputs/eval/safediff_stageplan_final]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

TASKS = ["select_poker", "select_mahjong", "select_drink", "select_book", "select_fruit", "select_toy"]
PLACE = {"select_fruit": "plate", "select_toy": "giftbox", "select_mahjong": None}  # mahjong place semantics logged via release_prob only


def episode(root: Path, task: str, seed: int) -> dict | None:
    d = root / task / f"seed{seed}"
    if not (d / "result.json").exists(): return None
    R = json.load(open(d / "result.json")); L = json.load(open(d / "stage_log.json")); S = json.load(open(d / "snapshots.json"))
    n = min(len(L), len(S) - 1); L, S = L[:n], S[1: n + 1]  # S[t+1] is the state after step t
    ph = [x["phase"] for x in L]; plan = R["plan"][0] if R["plan"] else None
    m = {"task": task, "seed": seed, "native": bool(R["success"]), "len": R["len"], "timeout": (not R["success"]) and R["len"] >= 500, "lang": R["language"][0]["source"]}
    ee = np.array([s["ee_p"] for s in S]); ob = np.array([s["obj_p"] for s in S]); g = np.array([bool(s["grasped"]) for s in S]); w = np.array([s["width"] for s in S]) * 1000
    kp0 = np.array(R["geo"][0]["kp_robot"][0]); base = np.array(R["geo"][0]["base"]); ee_r = ee - base if np.abs(ee[0] - base).max() < np.abs(ee[0]).max() else ee  # ee_p is world frame
    ee_r = ee - base
    if plan is None: m["stage"] = "F10 simulator / construction / invalid"; m["plan_ok"] = False; return m
    P = np.array(plan["poses"]); m["release_prob"] = plan["release_prob"]
    m["p1_err_cm"] = float(np.linalg.norm(P[1, :3] - kp0) * 100); m["p0_err_cm"] = float(np.linalg.norm(P[0, :3] - kp0) * 100)
    m["plan_ok"] = m["p1_err_cm"] < 5.0; m["plan_ok3"] = m["p1_err_cm"] < 3.0
    first = lambda name: next((i for i, p in enumerate(ph) if p == name), None)
    i_g, i_c, i_p, i_t, i_r = first("GRASP"), first("CLOSE"), first("POSTGRASP"), first("TRANSPORT"), first("RELEASE")
    m["pregrasp"] = i_g is not None; m["close_reached"] = i_c is not None
    m["ee_to_kp_at_close_cm"] = float(np.linalg.norm(ee_r[i_c] - kp0) * 100) if i_c is not None else None
    m["ee_to_p1_at_close_cm"] = float(np.linalg.norm(ee_r[i_c] - P[1, :3]) * 100) if i_c is not None else None
    m["reach3"] = bool(m["ee_to_kp_at_close_cm"] is not None and m["ee_to_kp_at_close_cm"] < 3.0)
    d_t = np.linalg.norm(ob - ob[0], axis=1) * 100; oth0 = S[0]["others"]
    skip = PLACE.get(task) or "~"
    disp_o = np.array([max([np.linalg.norm(np.array(s["others"][k]) - np.array(oth0[k])) * 100 for k in oth0 if k != "table" and skip not in k] or [0]) for s in S])
    end_pre = i_c if i_c is not None else n - 1
    m["collision"] = bool(d_t[: end_pre + 1].max() > 1.5 or disp_o[: end_pre + 1].max() > 1.5); m["nontarget_disp_cm"] = float(disp_o.max())
    if i_c is not None:
        k = min(i_c + 12, n - 1); m["between"] = bool(w[k] > 10); m["grasped"] = bool(g[i_c:].any()); m["empty_close"] = bool(not g[i_c: i_c + 30].any() and w[k] <= 10)
        gi = np.where(g[i_c:])[0]; run = 0
        if len(gi):
            t0 = i_c + gi[0]
            while t0 + run < n and g[t0 + run]: run += 1
        m["held_steps"] = int(run); m["lift_cm"] = float((ob[i_c:, 2].max() - ob[0, 2]) * 100)
        m["retained"] = bool(run >= 10 and m["lift_cm"] >= 5.0); m["early_drop"] = bool(run > 0 and t0 + run < n and not R["success"])
    else: m.update(between=False, grasped=False, empty_close=False, held_steps=0, lift_cm=0.0, retained=False, early_drop=False)
    m["postgrasp"] = bool(i_p is not None and m["grasped"] and m["lift_cm"] >= 5.0)
    if task in PLACE and PLACE[task]:
        gk = [k for k in oth0 if PLACE[task] in k]
        if gk:
            gp = np.array([s["others"][gk[0]] for s in S]); dxy = np.linalg.norm(ob[:, :2] - gp[:, :2], axis=1) * 100
            m["transported"] = bool((dxy < 6).any()); m["dxy_min_cm"] = float(dxy.min()); m["release_exec"] = i_r is not None
            m["released"] = bool(i_r is not None and m["transported"] and w[-1] > 60 and not g[-1]); m["placed"] = bool(m["transported"] and not g[-1] and dxy[-1] < 8)
        else: m.update(transported=None, release_exec=i_r is not None, released=None, placed=None)
    else: m.update(transported=None, release_exec=i_r is not None, released=None, placed=None)
    m["phys_proxy"] = bool(m["retained"] if m["transported"] is None else (m["placed"] or False))
    m["mismatch"] = m["native"] != m["phys_proxy"]
    # primary failure stage = first failed funnel stage
    if m["native"] and m["phys_proxy"]: s = "success"
    elif m["native"]: s = "F9 native metric false positive?"
    elif not m["plan_ok"]: s = "F1 grounding"
    elif not m["pregrasp"]: s = "F2 global approach / waypoint"
    elif not m["close_reached"] or not m["reach3"]: s = "F3 local alignment"
    elif not m["grasped"]: s = "F5 empty grasp / acquisition" if m["empty_close"] else "F4 premature / missed close"
    elif not m["retained"]: s = "F6 unstable grasp / retention"
    elif m["transported"] is False: s = "F7 transport"
    elif m["transported"] and not m["released"]: s = "F8 release"
    else: s = "F6 unstable grasp / retention"
    m["stage"] = s
    return m


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="outputs/eval/safediff_stageplan_final"); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(3000, 3020))); a = ap.parse_args()
    root = Path(a.root); rows = [m for t in TASKS for s in a.seeds if (m := episode(root, t, s))]
    (root / "episode_metrics.json").write_text(json.dumps(rows, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    ver = json.load(open(root / "verified.json")) if (root / "verified.json").exists() else {}
    print(f"{'task':16s}{'N':>3s}{'native':>8s}{'verif':>6s}{'P1<5cm':>7s}{'P1<3cm':>7s}{'p1err med':>10s}{'GRASP':>6s}{'reach3':>7s}{'grasped':>8s}{'retain':>7s}{'transp':>7s}{'relExec':>8s}{'placed':>7s}{'collis':>7s}{'timeout':>8s}")
    tot = {"N": 0, "native": 0, "verif": 0}; macro = []
    for t in TASKS:
        r = [x for x in rows if x["task"] == t]
        if not r: continue
        f = lambda k: sum(bool(x.get(k)) for x in r); v = sum(ver.get(f"{t}/{x['seed']}", x.get("phys_proxy", False)) for x in r)
        tr = "NA" if all(x.get("transported") is None for x in r) else str(f("transported")); pl = "NA" if all(x.get("placed") is None for x in r) else str(f("placed"))
        print(f"{t:16s}{len(r):3d}{f('native'):8d}{v:6d}{f('plan_ok'):7d}{f('plan_ok3'):7d}{np.median([x['p1_err_cm'] for x in r if 'p1_err_cm' in x]):10.1f}{f('pregrasp'):6d}{f('reach3'):7d}{f('grasped'):8d}{f('retained'):7d}{tr:>7s}{f('release_exec'):8d}{pl:>7s}{f('collision'):7d}{f('timeout'):8d}")
        tot["N"] += len(r); tot["native"] += f("native"); tot["verif"] += v; macro.append((f("native") / len(r), v / len(r)))
    print(f"MICRO N={tot['N']} native={tot['native'] / max(tot['N'], 1):.3f} verified={tot['verif'] / max(tot['N'], 1):.3f} | MACRO native={np.mean([m[0] for m in macro]):.3f} verified={np.mean([m[1] for m in macro]):.3f}")
    print("\nprimary failure stage:")
    for t in TASKS: print(f"  {t:16s}", dict(Counter(x["stage"] for x in rows if x["task"] == t)))


if __name__ == "__main__":
    main()
