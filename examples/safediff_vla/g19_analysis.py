#!/usr/bin/env python
"""G19 closed-loop analysis (poker seeds 3000-3009). Arms: A = raw G4d + learned decoder (log run) | B = G4d target + geometric chunk | C = GT anchor + geometric chunk (diagnostic upper bound) | D = GT anchor + learned decoder (G7 oracle, log run, reference). GT is used for diagnostics only."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g5_analysis import analyse  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g19"); G5D = Path("outputs/eval/safediff_vla_g5_closed_loop"); SEEDS = list(range(3000, 3010))
ARMS = {"A g4d+decoder": ("cl_log", "g4d"), "B g4d+GEO": ("cl_geo", "g4d"), "D GT+decoder": ("cl_log", "oracle"), "C GT+GEO": ("cl_geo", "oracle")}


def main():
    R = {}
    for a, (d, m) in ARMS.items():
        E = json.load(open(OUT / d / m / "episodes_raw.json")); L = pickle.load(open(OUT / d / m / "g19_log.pkl", "rb")); R[a] = {"E": E, "ana": [analyse(e) for e in E], "by": {i: [r for r in L["log"] if r["ep"] == i] for i in range(len(E))}, "chunks": L["chunks"]}
    print("== reproducibility of the logging runs vs the stored G5 episodes (max |EE diff| cm; success identical) ==")
    for a, m in (("A g4d+decoder", "g4d"), ("D GT+decoder", "oracle")):
        E0 = json.load(open(G5D / m / "episodes_raw.json")); d = [np.abs(np.array([s["ee_pos"] for s in x["steps"]]) - np.array([s["ee_pos"] for s in y["steps"]])).max() if len(x["steps"]) == len(y["steps"]) else 99 for x, y in zip(R[a]["E"], E0)]
        print(f"   {a}: max {max(d) * 100:.3f} cm; success identical {sum(x['success'] == y['success'] for x, y in zip(R[a]['E'], E0))}/10")
    print("\n== PRIMARY: reaching the target ==\n   " + f"{'metric':44s}" + "".join(f"{a:>16s}" for a in ARMS))
    def row(name, f, fmt="{:16.2f}"): print(f"   {name:44s}" + "".join(fmt.format(f(R[a])) if not isinstance(f(R[a]), str) else f"{f(R[a]):>16s}" for a in ARMS))
    mn = lambda r, k: np.array([min(s[k] for s in r["by"][i]) for i in range(10)])
    row("min EE-GT XY cm (mean)", lambda r: np.mean([a["min_ee_gt_xy"] for a in r["ana"]]) * 100); row("min EE-GT XY cm (median)", lambda r: np.median([a["min_ee_gt_xy"] for a in r["ana"]]) * 100)
    row("min |EE-GT z| cm (mean)", lambda r: np.mean([min(abs(s["dz"]) for s in r["by"][i]) for i in range(10)]) * 100); row("min EE-GT 3D cm (mean)", lambda r: np.mean([a["min_ee_gt_3d"] for a in r["ana"]]) * 100); row("min EE-GT 3D cm (median)", lambda r: np.median([a["min_ee_gt_3d"] for a in r["ana"]]) * 100)
    row("min EE-keypoint 3D cm (step log, mean)", lambda r: np.mean([min(s["d3"] for s in r["by"][i]) for i in range(10)]) * 100); row("min EE-keypoint XY cm (step log, mean)", lambda r: np.mean([min(s["dxy"] for s in r["by"][i]) for i in range(10)]) * 100)
    row("episodes reaching 3D<3cm", lambda r: f"{sum(any(s['d3'] < .03 for s in r['by'][i]) for i in range(10))}/10"); row("episodes reaching XY<3cm", lambda r: f"{sum(any(s['dxy'] < .03 for s in r['by'][i]) for i in range(10))}/10")
    row("time-to-3D<3cm (mean step; none=500)", lambda r: np.mean([next((s["t"] for s in r["by"][i] if s["d3"] < .03), 500) for i in range(10)]), "{:16.0f}")
    row("closest approached card is the target", lambda r: f"{sum(a['closest_card_is_target'] for a in r['ana'])}/10"); row("first plan targets the correct card", lambda r: f"{sum(a['first_plan_correct'] for a in r['ana'])}/10"); row("correct replans / replans", lambda r: f"{sum(a['n_correct_replans'] for a in r['ana'])}/{sum(a['n_replans'] for a in r['ana'])}")
    row("wrong-card displaced (episodes)", lambda r: f"{sum(a['wrong_card_displaced'] for a in r['ana'])}/10"); row("reactive close occurred", lambda r: f"{sum(a['close'] is not None for a in r['ana'])}/10"); row("close near the correct card (of closes)", lambda r: (lambda c: f"{sum(c)}/{len(c)}")([a['close']['correct'] for a in r['ana'] if a['close']]))
    print("\n== SECONDARY ==\n   " + f"{'metric':44s}" + "".join(f"{a:>16s}" for a in ARMS)); row("target lifted >3cm", lambda r: f"{sum(a['target_max_lift_z'] > .03 for a in r['ana'])}/10"); row("native success", lambda r: f"{sum(a['success'] for a in r['ana'])}/10")
    print("\n== geometric-chunk execution quality (replaced PRE_GRASP replans) ==")
    for a in ("B g4d+GEO", "C GT+GEO"):
        c = R[a]["chunks"]; print(f"   {a}: replans {len(c)} | first-action offset mean {np.mean([x['first_off'] for x in c]) * 100:.2f} cm max {np.max([x['first_off'] for x in c]) * 100:.2f}" if c else f"   {a}: none")
    print("\n== anchor error of the PRE_GRASP planner target at the replaced replans (B/C) vs GT keypoint, XY cm ==")
    for a in ("B g4d+GEO", "C GT+GEO"):
        xs = [np.linalg.norm(np.asarray(c["target"])[:2] - np.asarray(R[a]["by"][c["ep"]][min(c["t"], len(R[a]["by"][c["ep"]]) - 1)]["gt"])[:2]) * 100 for c in R[a]["chunks"]]; print(f"   {a}: per replan {[round(x, 1) for x in xs]} | mean {np.mean(xs):.1f}")
    print("\n== per-seed: min EE-GT 3D cm | close(c=correct,w=wrong) | success ==")
    for i, s in enumerate(SEEDS): print(f"   {s} " + " | ".join(f"{a[:1]} {R[a]['ana'][i]['min_ee_gt_3d'] * 100:5.1f} {('c' if R[a]['ana'][i]['close']['correct'] else 'w') if R[a]['ana'][i]['close'] else '-'} {int(R[a]['ana'][i]['success'])}" for a in ARMS))
    pickle.dump({"ok": 1}, open(OUT / "g19_cl_done.pkl", "wb"))


if __name__ == "__main__":
    main()
