#!/usr/bin/env python
"""Per-episode stage metrics + tables for the PRIMARY learned evaluation (`final_learned_eval.py` outputs). All
privileged reads (GT target pose, is_grasped, contacts) are DIAGNOSTIC ONLY. `phys_proxy` is a PROGRAMMATIC proxy for
physical success; the final 'verified physical SR' additionally requires the video check (see `verified.json`).
Usage: uv run python examples/safediff_vla/final_A_analysis.py [--root outputs/eval/safediff_final_A]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TASKS = ["select_poker", "select_mahjong", "select_drink", "select_book", "select_fruit", "select_toy"]
PLACE = {"select_fruit": "plate", "select_toy": "giftbox"}  # goal-container entity-name substring; others are pick-only
LIFT_CM, HELD_MIN = 5.0, 10


def episode(root: Path, task: str, seed: int) -> dict | None:
    d = root / task / f"seed{seed}"
    if not (d / "snapshots.json").exists():
        return None
    e = json.load(open(d / "episodes_raw.json"))[0]
    S = json.load(open(d / "snapshots.json"))
    S = S if isinstance(S[0], dict) else max(S, key=len)  # the stepped episode (a trailing reset adds a 1-length list)
    prov = json.load(open(d / "provenance.json"))
    R = e["steps"]; n = min(len(R), len(S)); R, S = R[:n], S[:n]
    tp = np.array([r["target_pos"] for r in R]); ee = np.array([r["ee_pos"] for r in R]); ob = np.array([s["obj_p"] for s in S])
    g = np.array([bool(s["grasped"]) for s in S]); w = np.array([s["width"] for s in S]) * 1000; ncon = np.array([s["ncon"] for s in S])
    dist = np.linalg.norm(ee - tp, axis=1) * 100
    ci = next((i for i, r in enumerate(R) if r["reactive_close_event"]), None); m_dec = False
    if ci is None:  # decoder-initiated close (no v2 proximity close fired): first physical finger closing (width < 70 mm after being open)
        ci = next((i for i in range(1, n) if w[i] < 70 <= w[:i].max()), None); m_dec = ci is not None
    end_pre = ci if ci is not None else n - 1
    m = {"task": task, "seed": seed, "native": bool(e["success"]), "len": e["episode_len"], "timeout": (not e["success"]) and e["episode_len"] >= 500,
         "lang": prov["language"][0]["source"], "n_resets": len(prov["language"])}
    first = next((r["planner_target_dist_to_gt"] for r in R if r.get("planner_target_dist_to_gt") is not None), None)
    m["first_target_err_cm"] = None if first is None else first * 100
    m["ground_ok"] = first is not None and first * 100 < 5.0  # first planner target within 5cm of GT object centre
    # nearest-object identity: predicted target nearer to GT target (initial) than to any other entity's initial pos
    pt = next((r["planner_target_world"] for r in R if r.get("planner_target_world") is not None), None)
    if pt is not None:
        others = [np.array(v) for k, v in S[0]["others"].items() if k != "table"]
        dt_ = np.linalg.norm(np.array(pt) - tp[0]); m["nearest_ok"] = bool(all(dt_ <= np.linalg.norm(np.array(pt) - o) for o in others))
    else: m["nearest_ok"] = False
    m["reach5"] = bool(dist[: end_pre + 1].min() < 5); m["reach3"] = bool(dist[: end_pre + 1].min() < 3); m["min_dist_cm"] = float(dist.min())
    d_t = np.linalg.norm(tp - tp[0], axis=1) * 100
    oth0 = S[0]["others"]; disp_o = np.zeros(n)
    for i, s in enumerate(S):
        disp_o[i] = max([np.linalg.norm(np.array(s["others"][k]) - np.array(oth0[k])) * 100 for k in oth0 if k != "table" and (PLACE.get(task, "~") not in k)] or [0])
    m["collision"] = bool(d_t[: end_pre + 1].max() > 1.5 or disp_o[: end_pre + 1].max() > 1.5); m["nontarget_disp_cm"] = float(disp_o.max())
    m["close"] = ci is not None; m["close_by_decoder"] = m_dec
    if ci is not None:
        k3 = min(ci + 3, n - 1); m["between"] = bool(w[k3] > 10); m["width_close3_mm"] = float(w[k3]); m["ncon_close"] = float(np.mean(ncon[ci: ci + 10]))
        m["is_grasped"] = bool(g[ci:].any()); m["empty_close"] = bool(not g[ci:min(ci + 30, n)].any() and w[k3] <= 10)
        m["lift_cm"] = float((ob[ci:, 2].max() - ob[0, 2]) * 100); m["post_disp_cm"] = float(d_t[ci:].max())
        gi = np.where(g[ci:])[0]
        if len(gi):
            t0 = ci + gi[0]; run = 0
            while t0 + run < n and g[t0 + run]: run += 1
            m["held_steps"] = int(run); m["early_drop"] = bool(t0 + run < n and not e["success"] and (ob[t0 + run:, 2].max() - ob[0, 2]) * 100 < LIFT_CM)
            m["follows_ee_end"] = bool(np.linalg.norm(ob[-1] - ee[-1]) * 100 < 10)
            m["phys_lift"] = bool(m["lift_cm"] >= LIFT_CM and run >= HELD_MIN)
        else: m.update(held_steps=0, early_drop=False, follows_ee_end=False, phys_lift=False)
    else: m.update(between=False, is_grasped=False, empty_close=False, lift_cm=0.0, post_disp_cm=float(d_t.max()), held_steps=0, early_drop=False, follows_ee_end=False, phys_lift=False)
    if task in PLACE:
        goal = [np.array(v) for k, v in S[0]["others"].items() if PLACE[task] in k]
        if goal:
            gp = np.array([np.array(s["others"][[k for k in oth0 if PLACE[task] in k][0]]) for s in S]); dxy = np.linalg.norm(ob[:, :2] - gp[:, :2], axis=1) * 100
            m["transported"] = bool((dxy < 6).any()); m["dxy_min_cm"] = float(dxy.min()); wr = w[-1] > 60
            m["released"] = bool(m["transported"] and wr and not g[-1]); m["exec_open_after_close"] = bool(ci is not None and (np.array([r["commanded_gripper_raw"] or 0 for r in R])[ci + 2:] > 0.5).any())
            m["phys_proxy"] = bool(m["transported"] and not g[-1] and dxy[-1] < 8)
        else: m.update(transported=None, released=None, phys_proxy=False)
    else: m.update(transported=None, released=None, exec_open_after_close=None, phys_proxy=bool(m["phys_lift"]))
    m["mismatch"] = m["native"] != m["phys_proxy"]
    return m


def stage(m: dict) -> str:  # primary failure stage heuristic (verified/adjusted against video in the report)
    if m["native"] and m["phys_proxy"]: return "success"
    if m["native"] and not m["phys_proxy"]: return "F9 native false positive?"
    if not m["ground_ok"] and not m["reach3"]: return "F1 grounding"
    if not m["reach3"]: return "F2 global approach / waypoint" if m["min_dist_cm"] > 5 else "F3 local alignment"
    if not m["close"]: return "F4 premature / missed close"
    if m["empty_close"] or not m["is_grasped"]: return "F5 empty grasp / acquisition"
    if m["is_grasped"] and not m["phys_lift"]: return "F6 unstable grasp / retention"
    if m.get("transported") is False and m["task"] in PLACE: return "F7 transport"
    if m["task"] in PLACE and m.get("transported") and not m.get("released"): return "F8 release"
    return "F6 unstable grasp / retention"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="outputs/eval/safediff_final_A"); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(3000, 3020))); a = ap.parse_args()
    root = Path(a.root); rows = []
    for t in TASKS:
        for s in a.seeds:
            m = episode(root, t, s)
            if m: m["stage"] = stage(m); rows.append(m)
    (root / "episode_metrics.json").write_text(json.dumps(rows, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    ver = json.load(open(root / "verified.json")) if (root / "verified.json").exists() else {}
    hdr = f"{'task':16s}{'N':>3s}{'native':>8s}{'proxy':>7s}{'verif':>7s}{'ground':>8s}{'reach<3':>8s}{'close':>6s}{'btwn':>5s}{'grasp':>6s}{'lift':>5s}{'transp':>7s}{'collis':>7s}{'timeout':>8s}{'mism':>5s}"
    print(hdr)
    tot = {k: 0 for k in ("N", "native", "proxy", "verif")}; macro = []
    for t in TASKS:
        r = [x for x in rows if x["task"] == t]
        if not r: continue
        f = lambda k: sum(bool(x[k]) for x in r)
        v = sum(ver.get(f"{t}/{x['seed']}", x["phys_proxy"]) for x in r)
        tr = "NA" if t not in PLACE else f"{sum(bool(x['transported']) for x in r)}"
        print(f"{t:16s}{len(r):3d}{f('native'):8d}{f('phys_proxy'):7d}{v:7d}{f('ground_ok'):8d}{f('reach3'):8d}{f('close'):6d}{f('between'):5d}{f('is_grasped'):6d}{f('phys_lift'):5d}{tr:>7s}{f('collision'):7d}{f('timeout'):8d}{f('mismatch'):5d}")
        tot["N"] += len(r); tot["native"] += f("native"); tot["proxy"] += f("phys_proxy"); tot["verif"] += v; macro.append((f("native") / len(r), v / len(r)))
    print(f"MICRO N={tot['N']} native={tot['native']/max(tot['N'],1):.3f} verified(or proxy)={tot['verif']/max(tot['N'],1):.3f} | MACRO native={np.mean([m[0] for m in macro]):.3f} verified={np.mean([m[1] for m in macro]):.3f}")
    from collections import Counter
    print("\nprimary failure stage per task:")
    for t in TASKS:
        c = Counter(x["stage"] for x in rows if x["task"] == t); print(f"  {t:16s}", dict(c))


if __name__ == "__main__":
    main()
