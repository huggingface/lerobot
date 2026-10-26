#!/usr/bin/env python
"""G6 diagnostics (no new rollouts): (1) LOCK DRIFT: distance (front px) from the lock location L / raw selected fine location to the nearest GT card, per replan index; fraction >45 px (= not on any card).
(2) anchor error among identity-correct replans by replan index (raw vs stab) to separate the selection effect (stab has more identity-correct LATE replans) from a real anchor degradation.
(3) post-close behaviour: replans after the reactive close event (phase 1) -- planner-target distance to the instructed card, raw vs stab."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from g4b_geometry import CAL, proj  # noqa: E402
from g5_analysis import analyse, OUT  # noqa: E402


def main():
    eps = {m: json.load(open(OUT / m / "episodes_raw.json")) for m in ("g4d", "g4d_stab")}; res = {}
    print("== (1) distance (px) from the identity location to the nearest GT card; frac > 45px (not on a card) ==")
    for m in eps:
        rows = []
        for e in eps[m]:
            S = e["steps"]
            for j, i in enumerate([i for i in range(1, len(S)) if S[i]["planner_replanned_this_step"]]):
                g = S[i]["grounder"]; st = g.get("stab"); loc = np.array(st["L"] if (st and st["locked"] and st["L"] is not None) else g["fine_px_front"]); px = [np.array(proj(CAL[0], v)) for v in S[i - 1]["candidates_world"].values()]; rows.append((j, min(np.linalg.norm(p - loc) for p in px)))
        for lab, lo, hi in (("1st", 0, 1), ("2nd-3rd", 1, 3), ("4th-6th", 3, 6), ("7th+", 6, 99)):
            xs = np.array([d for j, d in rows if lo <= j < hi]); print(f"   {m:9s} {lab:8s} n={len(xs):3d} median {np.median(xs):5.1f}px | >45px {np.mean(xs > 45):.2f}")
    print("\n== (2) anchor error (planner target vs instructed card, XY cm) among replans whose identity location is on the instructed card ==")
    for m in eps:
        out = []
        for lab, lo, hi in (("1st-3rd", 0, 3), ("4th-6th", 3, 6), ("7th+", 6, 99)):
            xs = []
            for e in eps[m]:
                S = e["steps"]; a = analyse(e); tgt = e["meta"]["target_entity"]
                for j, (i, r) in enumerate(zip([i for i in range(1, len(S)) if S[i]["planner_replanned_this_step"]], a["replans"])):
                    if not (lo <= j < hi): continue
                    g = S[i]["grounder"]; st = g.get("stab"); loc = np.array(st["L"] if (st and st["locked"] and st["L"] is not None) else g["fine_px_front"]); cw = S[i - 1]["candidates_world"]; k = min(cw, key=lambda kk: np.linalg.norm(np.array(proj(CAL[0], cw[kk])) - loc))
                    if k == tgt: xs.append(r["planner_to_gt_xy"] * 100)
            xs = np.array(xs); out.append(f"{lab}: n={len(xs)} median {np.median(xs):.1f} >5cm {np.mean(xs > 5):.2f}" if len(xs) else f"{lab}: -")
        print(f"   {m:9s} " + " | ".join(out))
    print("\n== (3) replans AFTER the reactive close (phase 1): planner target nearest-card correct / dist to instructed card (cm) ==")
    for m in eps:
        xs = []
        for e in eps[m]:
            S = e["steps"]; a = analyse(e); ce = next((i for i, s in enumerate(S) if s["reactive_close_event"]), None)
            if ce is None: continue
            for r in a["replans"]:
                if r["i"] > ce: xs.append((e["seed"], r["target_correct"], r["planner_to_gt_xy"] * 100))
        print(f"   {m:9s} n={len(xs)} seeds {sorted({x[0] for x in xs})}: correct {np.mean([x[1] for x in xs]) if xs else float('nan'):.2f} | dist median {np.median([x[2] for x in xs]) if xs else float('nan'):.1f} cm")
        for s in sorted({x[0] for x in xs}): print(f"        seed {s}: " + " ".join(f"{'OK' if c else 'x'}{d:.0f}" for sd, c, d in xs if sd == s))


if __name__ == "__main__":
    main()
