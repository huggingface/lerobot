#!/usr/bin/env python
"""G24 analysis: stage funnel per task (seeds 3000-3004): grasp -> retained (>=30 consecutive is_grasped steps after the close) -> transported (object within 15cm XY of the goal while held) -> release_ready -> decoder RAW OPEN present while ready -> OPEN executed -> placed (object within 6cm of the goal and not held at the episode end after a release) -> native success. Reference = the latched pipeline (G21 geo, G23 logging re-run): same oracle+geometry, gripper forced closed after the close."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

OUT = Path("outputs/eval/safediff_vla_g24"); TASKS = ["select_toy", "select_fruit", "select_mahjong"]; REF = {"select_toy": "G21 0/5, G23-rerun 1/5", "select_fruit": "G21 0/5, G23-rerun 3/5", "select_mahjong": "G21 4/5, G23-rerun 5/5, determinism-rerun 4/5"}


def longest(b):
    best = run = 0
    for v in b: run = run + 1 if v else 0; best = max(best, run)
    return best


def main():
    for t in TASKS:
        p = OUT / t / "g24_log.pkl"
        if not p.exists(): continue
        L = pickle.load(open(p, "rb")); n = len(L["episodes"]); print(f"\n===== {t} (goal entity per episode: {sorted(set(L['goal'].values()))}) | latched reference: {REF[t]} =====")
        F = {k: 0 for k in ("grasp", "retained", "transported", "ready", "raw_open_while_ready", "opened", "placed", "success")}; rows = []
        for i, e in enumerate(L["episodes"]):
            S = [s for s in L["steps"] if s["ep"] == i and s["post"]]; ci = e["close"]
            if not S: rows.append(f"   seed {e['seed']}: no post-grasp steps (close={ci}) success={int(e['success'])}"); F["success"] += int(e["success"]); continue
            g = [s["grasped"] for s in S]; d = np.array([s["dxy_goal"] for s in S]); ready = [s["ready"] for s in S]; rawo = [s["raw"] > 0.5 for s in S]; ex = [s["exec"] > 0.5 for s in S]
            f_grasp = any(g); f_ret = longest(g) >= 30; f_tr = bool(np.any((d < 0.15) & np.array(g))); f_rd = any(ready); f_raw = any(r and o for r, o in zip(ready, rawo)); f_open = any(r and o for r, o in zip(ready, ex))
            last = L["steps"]; ES = [s for s in L["steps"] if s["ep"] == i]; fin = ES[-1]; first_rd = next((k for k, r in enumerate(ready) if r), None); f_placed = bool(f_open and fin["dxy_goal"] < 0.06 and not any(s["grasped"] for s in ES[-5:]))
            for k, v in (("grasp", f_grasp), ("retained", f_ret), ("transported", f_tr), ("ready", f_rd), ("raw_open_while_ready", f_raw), ("opened", f_open), ("placed", f_placed), ("success", e["success"])): F[k] += int(bool(v))
            tpos = np.array(e["tpos"]); zl = (tpos[:, 2].max() - tpos[0, 2]) * 100
            rows.append(f"   seed {e['seed']}: len {e['len']} close@{ci} | grasped-steps {sum(g)} (longest run {longest(g)}) | min obj-goal xy {np.nanmin(d) * 100:.1f}cm | ready steps {sum(ready)} (first at post-step {first_rd}) | raw OPEN while ready {sum(r and o for r, o in zip(ready, rawo))} | executed OPEN while ready {sum(r and o for r, o in zip(ready, ex))} | final obj-goal xy {fin['dxy_goal'] * 100:.1f}cm held-at-end {int(fin['grasped'])} | max lift {zl:.1f}cm | success {int(e['success'])}")
        print("\n".join(rows)); print("   FUNNEL (episodes of %d): " % n + " -> ".join(f"{k} {v}" for k, v in F.items()))
        ready_no_raw = sum(1 for i in range(n) for _ in [0] if (lambda S: any(s["ready"] for s in S) and not any(s["ready"] and s["raw"] > 0.5 for s in S))([s for s in L["steps"] if s["ep"] == i and s["post"]]))
        print(f"   CASE C check (release_ready reached but decoder raw OPEN never present while ready): {ready_no_raw} episodes")


if __name__ == "__main__":
    main()
