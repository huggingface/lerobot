#!/usr/bin/env python
"""G8 analysis of the capture-only audit (fixed observations, decoder-only counterfactuals). Reference chunk = A1 (GT anchor, V2's own grounded feature). Chunk = 50 x (x,y,z) absolute action positions in the robot frame (m).
Strata (disjoint, per frame): EARLY t<=20 | NEAR/CONTACT EE-GT_xy<8cm or any card displaced >1.5cm | LATE 8-12cm | MID otherwise (>12cm, t>20).
Anchor-following gain on axis a for segment S: mean over S of  d(action_a) / d(anchor_a)  (signed; 1.0 = chunk moves one-for-one with the anchor). Segments: 1-5, 6-10, 11-20, 21-30, 31-50 (chunk index, 1-based)."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np

OUT = Path("outputs/eval/safediff_vla_g8_audit"); SEG = {"1-5": (0, 5), "6-10": (5, 10), "11-20": (10, 20), "21-30": (20, 30), "31-50": (30, 50)}; ORDER = ("EARLY", "MID", "LATE", "NEAR/CONTACT"); THR = 0.5
AX = {"x": (("A3", +1), ("A4", -1)), "y": (("A5", +1), ("A6", -1)), "z": (("A7", +1), ("A8", -1))}; AI = {"x": 0, "y": 1, "z": 2}


def stratum(f):
    if f["ee_gt_xy"] < 0.08 or f["any_card_displaced_cm"] > 1.5: return "NEAR/CONTACT"
    if f["t"] <= 20: return "EARLY"
    return "LATE" if f["ee_gt_xy"] <= 0.12 else "MID"


def gain(f, axis, seg):
    a, b = SEG[seg]; ref = f["cond"]["A1"]["xyz"]; gs = []
    for c, sgn in AX[axis]:
        d_anchor = f["cond"][c]["anchor_robot"] - f["cond"]["A1"]["anchor_robot"]; d_act = f["cond"][c]["xyz"][a:b] - ref[a:b]; ia = AI[axis]; gs.append(float(d_act[:, ia].mean() / d_anchor[ia]))
    return float(np.mean(gs))


def cross_leak(f, axis, seg):
    a, b = SEG[seg]; ref = f["cond"]["A1"]["xyz"]; tot = []
    for c, _ in AX[axis]:
        d_anchor = f["cond"][c]["anchor_robot"] - f["cond"]["A1"]["anchor_robot"]; d_act = (f["cond"][c]["xyz"][a:b] - ref[a:b]).mean(0); ia = AI[axis]; others = [i for i in range(3) if i != ia]; tot.append(float(np.linalg.norm(d_act[others]) / abs(d_anchor[ia])))
    return float(np.mean(tot))


def close_idx(f, key): g = f["cond"][key]["grip"]; return int(np.argmax(g <= THR)) if (g <= THR).any() else 50


def main():
    F = []
    for src in ("g4d", "oracle"):
        for fr in pickle.load(open(OUT / src / "audit.pkl", "rb")): fr["src"] = src; fr["stratum"] = stratum(fr); F.append(fr)
    print("frames:", len(F), {s: sum(1 for f in F if f['src'] == s) for s in ('g4d', 'oracle')}, "| strata:", {k: sum(1 for f in F if f['stratum'] == k) for k in ORDER}, "| phase1 frames:", sum(1 for f in F if f['phase'] == 1), "| gripper-closed(state) frames:", sum(1 for f in F if f['state7'][6] > 0.5))
    R = {}
    print("\n== Q1/Q2: ANCHOR-FOLLOWING GAIN (+-5cm anchor perturbation; reference A1). signed gain per axis, by stratum and chunk segment ==")
    for ax in ("x", "y", "z"):
        print(f" axis {ax}:  (gain: mean over frames [25th,75th pct]);  columns = chunk segments " + " ".join(f"{s:>16s}" for s in SEG))
        for st in ORDER + ("ALL",):
            fs = [f for f in F if st == "ALL" or f["stratum"] == st]
            if not fs: continue
            row = []
            for sg in SEG: g = np.array([gain(f, ax, sg) for f in fs]); row.append(f"{g.mean():+.2f}[{np.percentile(g, 25):+.2f},{np.percentile(g, 75):+.2f}]"); R[(ax, st, sg)] = float(g.mean())
            print(f"   {st:13s} n={len(fs):3d} " + " ".join(f"{r:>16s}" for r in row))
    print("\n== cross-axis leakage (|off-axis chunk change| / |anchor change|), full chunk 1-50, by stratum ==")
    for ax in ("x", "y", "z"): print(f"   perturb {ax}: " + " | ".join(f"{st} {np.mean([cross_leak(f, ax, '31-50') for f in F if f['stratum'] == st]):.2f}" for st in ORDER if any(f['stratum'] == st for f in F)))
    print("\n== endpoint (chunk index 50) and first-10 / full-chunk mean response to +-5cm XY (|Delta action_xy| in cm; ideal 5.0), by stratum ==")
    for st in ORDER:
        fs = [f for f in F if f["stratum"] == st]
        if not fs: continue
        def mag(sl): return np.mean([np.linalg.norm((f["cond"][c]["xyz"][sl][:, :2] - f["cond"]["A1"]["xyz"][sl][:, :2]).mean(0)) for f in fs for c in ("A3", "A4", "A5", "A6")]) * 100
        def magz(sl): return np.mean([abs((f["cond"][c]["xyz"][sl][:, 2] - f["cond"]["A1"]["xyz"][sl][:, 2]).mean()) for f in fs for c in ("A7", "A8")]) * 100
        print(f"   {st:13s} n={len(fs):3d} | xy: first1 {mag(slice(0, 1)):.2f} first5 {mag(slice(0, 5)):.2f} first10 {mag(slice(0, 10)):.2f} full {mag(slice(0, 50)):.2f} endpoint {mag(slice(49, 50)):.2f} | z: first5 {magz(slice(0, 5)):.2f} first10 {magz(slice(0, 10)):.2f} full {magz(slice(0, 50)):.2f} endpoint {magz(slice(49, 50)):.2f}")
    print("\n== Q3: WRONG-CARD COUNTERFACTUAL (A1 GT anchor vs A2 nearest wrong-card anchor, same observation) ==")
    for st in ORDER + ("ALL",):
        fs = [f for f in F if st == "ALL" or f["stratum"] == st]
        if not fs: continue
        sep_end = np.array([np.linalg.norm(f["cond"]["A1"]["xyz"][49, :2] - f["cond"]["A2"]["xyz"][49, :2]) * 100 for f in fs]); sep10 = np.array([np.linalg.norm((f["cond"]["A1"]["xyz"][:10, :2] - f["cond"]["A2"]["xyz"][:10, :2]).mean(0)) * 100 for f in fs]); anc = np.array([np.linalg.norm(f["cond"]["A1"]["anchor_robot"][:2] - f["cond"]["A2"]["anchor_robot"][:2]) * 100 for f in fs])
        def cosv(f, c, tgtkey):
            ee = f["ee_robot"][:2]; v = f["cond"][c]["xyz"][:10, :2].mean(0) - ee; d = f["cond"][c]["anchor_robot"][:2] - ee; return float(v @ d / (np.linalg.norm(v) * np.linalg.norm(d) + 1e-9))
        cg = np.array([cosv(f, "A1", None) for f in fs]); cw = np.array([cosv(f, "A2", None) for f in fs]); ce = np.array([float((f["cond"]["A1"]["xyz"][49, :2] - f["ee_robot"][:2]) @ (f["cond"]["A1"]["anchor_robot"][:2] - f["ee_robot"][:2]) / (np.linalg.norm(f["cond"]["A1"]["xyz"][49, :2] - f["ee_robot"][:2]) * np.linalg.norm(f["cond"]["A1"]["anchor_robot"][:2] - f["ee_robot"][:2]) + 1e-9)) for f in fs])
        print(f"   {st:13s} n={len(fs):3d} | anchor separation {anc.mean():.1f}cm -> action separation endpoint {sep_end.mean():.1f}cm ({sep_end.mean() / anc.mean():.2f}x) first10 {sep10.mean():.1f}cm ({sep10.mean() / anc.mean():.2f}x) | cos(first10 action disp, direction to anchor): GT {cg.mean():+.2f} wrong {cw.mean():+.2f} | cos(endpoint disp, GT dir) {ce.mean():+.2f}")
    print("\n== closing-time sensitivity: first chunk index with gripper <= open threshold (50 = no close predicted); A1 vs A2 vs F-conditions ==")
    for st in ORDER:
        fs = [f for f in F if st == "ALL" or f["stratum"] == st]
        if fs: print(f"   {st:13s} n={len(fs):3d} | mean close idx A1 {np.mean([close_idx(f, 'A1') for f in fs]):.1f} A2 {np.mean([close_idx(f, 'A2') for f in fs]):.1f} A0 {np.mean([close_idx(f, 'A0') for f in fs]):.1f} | frac frames predicting a close within chunk: A1 {np.mean([close_idx(f, 'A1') < 50 for f in fs]):.2f} | mean |close idx change| A1->A2 {np.mean([abs(close_idx(f, 'A1') - close_idx(f, 'A2')) for f in fs]):.1f} | gripper chunk |Delta| A1-A2 {np.mean([np.abs(f['cond']['A1']['grip'] - f['cond']['A2']['grip']).mean() for f in fs]):.3f}")
    print("\n== extension: WHICH CONDITIONING PATH MOVES THE CHUNK? mean |Delta chunk xyz| (cm, full chunk) relative to A1 ==")
    for st in ORDER:
        fs = [f for f in F if f["stratum"] == st]
        if not fs: continue
        d = lambda c: np.mean([np.linalg.norm(f["cond"][c]["xyz"] - f["cond"]["A1"]["xyz"], axis=1).mean() for f in fs]) * 100
        print(f"   {st:13s} n={len(fs):3d} | anchor->wrong (A2) {d('A2'):.2f} | anchor->G4d pred (A0) {d('A0'):.2f} | feature zeroed (F1) {d('F1'):.2f} | feature->GT-card oracle (F2) {d('F2'):.2f} | feature->wrong card, anchor GT (F4) {d('F4'):.2f} | BOTH wrong (F3) {d('F3'):.2f} | wrong anchor+GT feature (F5) {d('F5'):.2f}")
    print("\n== STATE-CONDITIONING: anchor gain (xy, full chunk, mean of x/y gains) vs state ==")
    gxy = lambda f: (gain(f, "x", "31-50") + gain(f, "y", "31-50")) / 2; gz = lambda f: gain(f, "z", "31-50"); g10 = lambda f: (gain(f, "x", "6-10") + gain(f, "y", "6-10")) / 2
    for lab, sel in (("EE-GT xy >30cm", lambda f: f["ee_gt_xy"] > .30), ("15-30cm", lambda f: .15 < f["ee_gt_xy"] <= .30), ("8-15cm", lambda f: .08 < f["ee_gt_xy"] <= .15), ("<8cm", lambda f: f["ee_gt_xy"] <= .08), ("PRE_GRASP", lambda f: f["phase"] == 0), ("POST_GRASP", lambda f: f["phase"] == 1),
                     ("gripper state open", lambda f: f["state7"][6] <= 0.5), ("gripper state closed", lambda f: f["state7"][6] > 0.5), ("no card displaced", lambda f: f["any_card_displaced_cm"] <= 1.5), ("card contact/displaced", lambda f: f["any_card_displaced_cm"] > 1.5), ("EE z above anchor >8cm", lambda f: f["ee_gt_z"] > .08), ("EE z within 4cm of anchor", lambda f: abs(f["ee_gt_z"]) <= .04)):
        fs = [f for f in F if sel(f)]
        if len(fs) >= 3: print(f"   {lab:26s} n={len(fs):3d} | xy gain 6-10: {np.mean([g10(f) for f in fs]):+.2f}  31-50: {np.mean([gxy(f) for f in fs]):+.2f} | z gain 31-50: {np.mean([gz(f) for f in fs]):+.2f}")
    json.dump({"|".join(k): v for k, v in R.items()}, open(OUT / "g8_gain_table.json", "w"), indent=1)


if __name__ == "__main__":
    main()
