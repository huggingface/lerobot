#!/usr/bin/env python
"""Diagnostic: is the late-replan correctness drop a survivorship artefact or related to the arm being over the cards? Same-episode control: only episodes that ran the full 500 steps (all models), replans bucketed by EE-to-NEAREST-card XY distance at decision time."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from g5_analysis import analyse, OUT, MODELS  # noqa: E402


def main():
    raw = {m: json.load(open(OUT / m / "episodes_raw.json")) for m in MODELS}; ana = {m: [analyse(e) for e in raw[m]] for m in MODELS}
    print("== replan correctness vs EE proximity to the nearest card (XY, decision time); ALL replans ==")
    res = {}
    for m in MODELS:
        rows = []
        for e, a in zip(raw[m], ana[m]):
            S = e["steps"]
            for r in a["replans"]:
                d = r["i"] - 1; ee = np.array(S[d]["ee_pos"])[:2]; near = min(np.linalg.norm(np.array(v)[:2] - ee) for v in S[d]["candidates_world"].values()); rows.append((near, r["target_correct"], r.get("fine_correct"), r.get("map_correct"), r["i"], a["len"]))
        out = {}
        for lab, lo, hi in ((">25cm", .25, 9), ("12-25cm", .12, .25), ("<12cm (arm over cards)", 0, .12)):
            xs = [x for x in rows if lo <= x[0] < hi]; out[lab] = {"n": len(xs), "planner_card_correct": float(np.mean([x[1] for x in xs])) if xs else None}
            if m == "g4d" and xs: out[lab].update({"fine_correct": float(np.mean([x[2] for x in xs])), "map_correct": float(np.mean([x[3] for x in xs]))})
        res[m] = out; print(f"   {m:4s} " + " | ".join(f"{k}: n={v['n']} planner {v['planner_card_correct']:.2f}" + (f" fine {v['fine_correct']:.2f} map {v['map_correct']:.2f}" if 'fine_correct' in v else "") for k, v in out.items() if v["n"]))
    print("\n== survivorship control: episodes that ran 500 steps in ALL THREE models (same scenes); correctness by replan index ==")
    full = [i for i in range(len(raw["v2"])) if all(ana[m][i]["len"] >= 499 for m in MODELS)]; print("   seeds:", [raw['v2'][i]["seed"] for i in full])
    for m in MODELS:
        out = []
        for lab, lo, hi in (("1st", 0, 1), ("2nd-3rd", 1, 3), ("4th-6th", 3, 6), ("7th+", 6, 99)):
            xs = [r["target_correct"] for i in full for j, r in enumerate(ana[m][i]["replans"]) if lo <= j < hi]; out.append(f"{lab} {np.mean(xs):.2f} (n={len(xs)})" if xs else f"{lab} -")
        print(f"   {m:4s} " + " | ".join(out))
    json.dump(res, open(OUT / "g5_occlusion_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
