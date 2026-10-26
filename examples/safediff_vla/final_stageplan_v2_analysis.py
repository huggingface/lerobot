#!/usr/bin/env python
"""Per-episode funnel + tables for the FINAL StagePlan-v2 evaluation (`final_stageplan_v2_eval.py` outputs). Privileged snapshots are used for SCORING only.
Failure stages (primary = first failed): grasp-predictor error / approach / close transition / grasp acquisition / post-grasp decoder / retention / release / native-mismatch.
Usage: uv run python examples/safediff_vla/final_stageplan_v2_analysis.py [--root outputs/eval/safediff_stageplan_v2_final]
"""

from __future__ import annotations

import argparse, json
from collections import Counter
from pathlib import Path

import numpy as np

TASKS = ["select_poker", "select_mahjong", "select_drink", "select_book", "select_fruit", "select_toy"]
PLACE = {"select_fruit": "plate", "select_toy": "giftbox", "select_mahjong": "placemat"}


def episode(root: Path, task: str, seed: int) -> dict | None:
    d = root / task / f"seed{seed}"
    if not (d / "result.json").exists(): return None
    R = json.load(open(d / "result.json")); L = json.load(open(d / "stage_log.json")); S = json.load(open(d / "snapshots.json"))
    n = min(len(L), len(S) - 1); L, S = L[:n], S[1: n + 1]
    m = {"task": task, "seed": seed, "native": bool(R["success"]), "len": R["len"], "timeout": (not R["success"]) and R["len"] >= 500, "lang": R["language"][0]["source"]}
    sv = R["plan"][0]["sv"] if R["plan"] else None
    if sv is None: m["stage"] = "F10 invalid"; return m
    base = np.array(R["geo"][0]["base"]); kp = np.array(R["geo"][0]["kp_robot"][0]); G = np.array(sv["grasp_xyz"])
    ee = np.array([s["ee_p"] for s in S]) - base; ob = np.array([s["obj_p"] for s in S]); g = np.array([bool(s["grasped"]) for s in S]); w = np.array([s["width"] for s in S]) * 1000
    ph = np.array([x["phase"] for x in L]); ic = int(np.argmax(ph == 1)) if (ph == 1).any() else None
    m["grasp_err_cm"] = float(np.linalg.norm(G - kp) * 100); m["grasp_ok5"] = m["grasp_err_cm"] < 5; m["grasp_ok3"] = m["grasp_err_cm"] < 3
    end_pre = ic if ic is not None else n - 1
    m["min_ee_to_G_cm"] = float(np.linalg.norm(ee[: end_pre + 1] - G, axis=1).min() * 100); m["approach_ok"] = m["min_ee_to_G_cm"] < 3
    d_t = np.linalg.norm(ob - ob[0], axis=1) * 100; oth0 = S[0]["others"]; skip = PLACE.get(task, "~")
    disp_o = np.array([max([np.linalg.norm(np.array(s["others"][k]) - np.array(oth0[k])) * 100 for k in oth0 if k != "table" and skip not in k] or [0]) for s in S])
    m["collision"] = bool(d_t[: end_pre + 1].max() > 1.5 or disp_o[: end_pre + 1].max() > 1.5)
    m["close"] = ic is not None; m["close_dist_cm"] = (R["close_events"][0]["dist_m"] * 100) if R["close_events"] else None
    if ic is not None:
        k = min(ic + 10, n - 1); m["width10"] = float(w[k]); m["grasped"] = bool(g[ic:].any())
        gi = np.where(g[ic:])[0]; run = 0
        if len(gi):
            t0 = ic + gi[0]
            while t0 + run < n and g[t0 + run]: run += 1
        m["held"] = int(run); m["lift_cm"] = float((ob[ic:, 2].max() - ob[0, 2]) * 100); m["lifted"] = bool(m["grasped"] and m["lift_cm"] >= 5 and run >= 10)
        m["lost"] = bool(run > 0 and t0 + run < n and not R["success"])
    else: m.update(width10=None, grasped=False, held=0, lift_cm=0.0, lifted=False, lost=False)
    gk = [k for k in oth0 if PLACE.get(task, "~") in k] if task in PLACE else []
    if gk:
        gp = np.array([s["others"][gk[0]] for s in S]); dxy = np.linalg.norm(ob[:, :2] - gp[:, :2], axis=1) * 100
        m["transported"] = bool((dxy < 6).any()); m["released"] = bool(m["transported"] and ic is not None and (w[ic + 10:] > 60).any()); m["placed"] = bool(m["transported"] and not g[-1] and dxy[-1] < 8)
    else: m.update(transported=None, released=None, placed=None)
    m["phys_proxy"] = bool(m["lifted"] if m["transported"] is None else (m["placed"] or m["transported"]))
    if R["success"]: s = "success"
    elif not m["grasp_ok5"] and not m["approach_ok"]: s = "grasp predictor error"
    elif not m["approach_ok"]: s = "approach failure"
    elif not m["close"]: s = "close transition failure"
    elif not m["grasped"]: s = "grasp acquisition failure" if m["grasp_ok5"] else "grasp predictor error"
    elif not m["lifted"]: s = "post-grasp decoder failure"
    elif m["lost"]: s = "retention failure"
    elif m["transported"] is False: s = "post-grasp decoder failure (transport)"
    elif m["transported"] and not m["released"]: s = "release failure"
    else: s = "other / retention"
    m["stage"] = s; return m


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="outputs/eval/safediff_stageplan_v2_final"); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(3000, 3020))); a = ap.parse_args()
    root = Path(a.root); rows = [m for t in TASKS for s in a.seeds if (m := episode(root, t, s))]
    (root / "episode_metrics.json").write_text(json.dumps(rows, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    ver = json.load(open(root / "verified.json")) if (root / "verified.json").exists() else {}
    print(f"{'task':16s}{'N':>3s}{'native':>8s}{'verif':>6s}{'G<5cm':>7s}{'G<3cm':>7s}{'G err med':>10s}{'approach':>9s}{'close':>6s}{'grasped':>8s}{'lifted':>7s}{'transp':>7s}{'releas':>7s}{'collis':>7s}{'timeout':>8s}")
    tot = {"N": 0, "native": 0, "verif": 0}; macro = []
    for t in TASKS:
        r = [x for x in rows if x["task"] == t]
        if not r: continue
        f = lambda k: sum(bool(x.get(k)) for x in r); v = sum(ver.get(f"{t}/{x['seed']}", x.get("phys_proxy", False)) for x in r)
        tr = "NA" if all(x.get("transported") is None for x in r) else str(f("transported")); rl = "NA" if all(x.get("released") is None for x in r) else str(f("released"))
        print(f"{t:16s}{len(r):3d}{f('native'):8d}{v:6d}{f('grasp_ok5'):7d}{f('grasp_ok3'):7d}{np.median([x['grasp_err_cm'] for x in r if 'grasp_err_cm' in x]):10.1f}{f('approach_ok'):9d}{f('close'):6d}{f('grasped'):8d}{f('lifted'):7d}{tr:>7s}{rl:>7s}{f('collision'):7d}{f('timeout'):8d}")
        tot["N"] += len(r); tot["native"] += f("native"); tot["verif"] += v; macro.append((f("native") / len(r), v / len(r)))
    print(f"MICRO N={tot['N']} native={tot['native'] / max(tot['N'], 1):.3f} proxy/verified={tot['verif'] / max(tot['N'], 1):.3f} | MACRO native={np.mean([m[0] for m in macro]):.3f} proxy/verified={np.mean([m[1] for m in macro]):.3f}")
    print("\nprimary failure stage:")
    for t in TASKS: print(f"  {t:16s}", dict(Counter(x["stage"] for x in rows if x["task"] == t)))


if __name__ == "__main__":
    main()
