#!/usr/bin/env python
"""G7 ORACLE-ANCHOR causal diagnostic (poker seeds 3000-3009, paired with the G5 raw G4d logs and the G6 stabilised logs). The oracle replaces ONLY the decoder-conditioning anchor by the GT grasp keypoint; everything else is identical.
Chunk geometry: a chunk starts at a replan decided from record i-1 and ends at min(next replan - 1, i+49, episode end). XY geometry w.r.t. the instructed card's position at decision time (the GT keypoint has the same XY).
`cos` = cosine between the chunk's EE XY displacement and the direction from the decision-time EE to the GT anchor."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g5_analysis import analyse, OUT  # noqa: E402

RUNS = {"raw": "g4d", "stab": "g4d_stab", "oracle": "oracle"}; BIG = 0.12


def chunks(ep):
    S = ep["steps"]; n = len(S); ee = np.array([s["ee_pos"] for s in S]); rep = [i for i in range(1, n) if S[i]["planner_replanned_this_step"]]; out = []
    for j, i in enumerate(rep):
        d = i - 1; end = min((rep[j + 1] - 1) if j + 1 < len(rep) else n - 1, i + 49, n - 1); gt = np.array(S[d]["target_pos"])[:2]; e0 = ee[d][:2]; e1 = ee[end][:2]; mv = e1 - e0; dg = gt - e0; dp = np.array(S[i]["planner_target_world"])[:2] - e0
        cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)); phase = S[i]["phase"]
        out.append({"j": j, "i": i, "len": end - d, "phase": phase, "start": float(np.linalg.norm(dg)), "end": float(np.linalg.norm(e1 - gt)), "gain": float(np.linalg.norm(dg) - np.linalg.norm(e1 - gt)), "disp": float(np.linalg.norm(mv)), "cos_gt": cos(mv, dg), "cos_planner": cos(mv, dp), "planner_err": float(np.linalg.norm(np.array(S[i]["planner_target_world"])[:2] - gt))})
    return out


def main():
    raw = {m: json.load(open(OUT / d / "episodes_raw.json")) for m, d in RUNS.items()}; seeds = sorted(e["seed"] for e in raw["raw"]); E = {m: {e["seed"]: e for e in raw[m]} for m in RUNS}; A = {m: {s: analyse(E[m][s]) for s in seeds} for m in RUNS}; C = {m: {s: chunks(E[m][s]) for s in seeds} for m in RUNS}
    print("== sanity: oracle planner target vs GT (xy cm) at replans:", "median %.2f max %.2f" % (np.median([c["planner_err"] * 100 for s in seeds for c in C["oracle"][s]]), max(c["planner_err"] * 100 for s in seeds for c in C["oracle"][s])))
    ge = [r for s in seeds for r in [S_["grounder"]["g4d_err_xy"] for S_ in E["oracle"][s]["steps"] if S_["planner_replanned_this_step"] and S_["grounder"]]]; print("   G4d's own prediction error inside the oracle run (logged, not used): xy median %.1f cm mean %.1f cm | >5cm %.2f (n=%d replans)" % (np.median(ge) * 100, np.mean(ge) * 100, np.mean(np.array(ge) > .05), len(ge)))
    print("\n== PER-SEED (raw G4d | G4d+stab | ORACLE): minEE-GT xy cm | closest-approached card | close (step, near correct?) | wrong card displaced | success | len ==")
    for s in seeds:
        line = f" seed {s} {E['raw'][s]['meta']['target_entity']:>17s}"
        for m in RUNS:
            a = A[m][s]; c = a["close"]; line += f" | {m:6s} min{a['min_ee_gt_xy'] * 100:5.1f} {'T' if a['closest_card_is_target'] else 'x'} close {('t=%d %s' % (c['step'], 'OK' if c['correct'] else 'wr')) if c else '--'} wd{int(a['wrong_card_displaced'])} {'SUCC' if a['success'] else 'fail'} {a['len']}"
        print(line)
    f = lambda m, k: np.mean([k(A[m][s]) for s in seeds])
    print("\n== PRIMARY METRICS (n=10 paired seeds; SR secondary, no statistical claim) ==")
    rows = [("min EE-GT xy (cm) mean", lambda e: e["min_ee_gt_xy"] * 100), ("min EE-GT xy (cm) median", None), ("min EE-GT 3D (cm) mean", lambda e: e["min_ee_gt_3d"] * 100), ("min EE-GT xy < 3cm", lambda e: e["min_ee_gt_xy"] < .03), ("closest-approached card is target", lambda e: e["closest_card_is_target"]),
            ("target card displaced", lambda e: e["target_displaced"]), ("wrong card displaced (wrong-card interaction)", lambda e: e["wrong_card_displaced"]), ("reactive close occurred (episodes)", lambda e: e["close"] is not None), ("target max lift >3cm", lambda e: e["target_max_lift_z"] > .03),
            ("native is_grasped(target) ever", lambda e: e["target_is_grasped_any(native)"]), ("native success (secondary)", lambda e: e["success"])]
    print(f"   {'':46s}" + "".join(f"{m:>9s}" for m in RUNS))
    for name, k in rows: print(f"   {name:46s}" + "".join(f"{(np.median([A[m][s]['min_ee_gt_xy'] * 100 for s in seeds]) if k is None else f(m, k)):9.3f}" for m in RUNS))
    for m in RUNS: cl = [A[m][s]["close"] for s in seeds if A[m][s]["close"]]; print(f"   {m:6s} closes {len(cl)} | near correct card {sum(c['correct'] for c in cl)} | close ee->GT xy (cm) {[round(c['ee_to_gt_xy'] * 100, 1) for c in cl]} | close steps {[c['step'] for c in cl]}")
    print("\n== first 50-step chunk (chunk 1) ==")
    for m in RUNS:
        c1 = [C[m][s][0] for s in seeds]; print(f"   {m:6s} approach gain {np.mean([c['gain'] for c in c1]) * 100:+.1f} cm | end EE-GT {np.mean([c['end'] for c in c1]) * 100:.1f} cm | cos(disp, GT dir) mean {np.mean([c['cos_gt'] for c in c1]):+.2f} | planner->GT err {np.mean([c['planner_err'] for c in c1]) * 100:.1f} cm")
    print("\n== CHUNK-LEVEL CAUSAL TEST: PRE_GRASP chunks with EE-GT > 12cm at decision ==")
    for m in RUNS:
        cs = [c for s in seeds for c in C[m][s] if c["start"] > BIG and c["phase"] == 0]; ok = [c for c in cs if c["planner_err"] < 0.05]
        g = lambda xs, k: np.mean([x[k] for x in xs]) if xs else float("nan")
        print(f"   {m:6s} n={len(cs):3d} | approach gain {g(cs, 'gain') * 100:+5.1f}cm | approach>2cm {np.mean([c['gain'] > .02 for c in cs]):.2f} | cos(disp,GT dir) {g(cs, 'cos_gt'):+.2f} | chunk-end EE-GT {g(cs, 'end') * 100:5.1f}cm | cos(disp,PLANNER dir) {g(cs, 'cos_planner'):+.2f}"
              + (f" || planner-correct (<5cm) subset n={len(ok)}: gain {g(ok, 'gain') * 100:+.1f} approach>2cm {np.mean([c['gain'] > .02 for c in ok]) if ok else float('nan'):.2f} cos(disp,planner) {g(ok, 'cos_planner'):+.2f}" if ok else ""))
    print("\n== by chunk index (PRE_GRASP, start>12cm): early vs late ==")
    for lab, lo, hi in (("chunk 1", 0, 1), ("chunks 2-3", 1, 3), ("chunks 4+", 3, 99)):
        print("   " + lab + ": " + " | ".join(f"{m} n={len([c for s in seeds for c in C[m][s] if lo <= c['j'] < hi and c['start'] > BIG and c['phase'] == 0])} gain {np.mean([c['gain'] for s in seeds for c in C[m][s] if lo <= c['j'] < hi and c['start'] > BIG and c['phase'] == 0] or [np.nan]) * 100:+.1f} cos {np.mean([c['cos_gt'] for s in seeds for c in C[m][s] if lo <= c['j'] < hi and c['start'] > BIG and c['phase'] == 0] or [np.nan]):+.2f}" for m in RUNS))
    print("\n== Q5: oracle chunks (PRE_GRASP, start>12cm) where the decoder does NOT follow the (correct) anchor ==")
    cs = [c for s in seeds for c in C["oracle"][s] if c["start"] > BIG and c["phase"] == 0]; print(f"   n={len(cs)} | not approaching (gain<=2cm) {np.mean([c['gain'] <= .02 for c in cs]):.2f} | cos<0.3 {np.mean([c['cos_gt'] < .3 for c in cs]):.2f} | moves away (gain<0) {np.mean([c['gain'] < 0 for c in cs]):.2f} | mean |disp| {np.mean([c['disp'] for c in cs]) * 100:.1f}cm")
    for s in seeds: print(f"      seed {s}: " + " ".join(f"[{c['gain'] * 100:+.0f}cm cos{c['cos_gt']:+.1f} st{c['start'] * 100:.0f}]" for c in C['oracle'][s] if c['phase'] == 0 and c['start'] > BIG))
    json.dump({m: {str(s): {"analysis": {k: v for k, v in A[m][s].items() if not k.startswith('_')}, "chunks": C[m][s]} for s in seeds} for m in RUNS}, open(OUT / "g7_analysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
