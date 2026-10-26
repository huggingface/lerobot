#!/usr/bin/env python
"""G13 analysis of the visual-source interventions (see g13_interventions.py). Effects are measured against the SAME frame's original chunk (GT anchor, original images). Variants: F = grounded feature of the original image held fixed (isolates the decoder's visual-memory path), R = grounded feature recomputed (deployed pipeline).
Reference from G10 (full image-token mask, F-style): |dz chunk| 7.19cm, |dxy| 6.98cm, close index +21.96 steps. dz = chunk z - EE z (cm); gripper action convention: closed if <= 0.5."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
OUT = Path("outputs/eval/safediff_vla_g13_audit"); THR = 0.5
CONDS1 = ["cam_front", "cam_right", "cam_wrist", "cam_all"]; CONDS2 = ["robot_all", "scene_all", "cards_all", "robot_wrist", "robot_front"]; CONDS3 = ["swapR_diff", "swapR_same", "swapR_diff_wrist", "swapS_diff", "swapS_same"]


def ci(g): c = g <= THR; return int(np.argmax(c)) if c.any() else 50


def main():
    D = pickle.load(open(OUT / "interventions.pkl", "rb")); C0, R, donors, B = D["C0"], D["R"], D["donors"], D["bank"]; N = len(B); ee = [e["ee_robot"] for e in B]; prim = np.array([e["ee_gt_z"] > 0.06 for e in B])
    print(f"frames {N} | primary subset (target >=6cm below EE) {prim.sum()} | donors found: " + ", ".join(f"{k} {int((v >= 0).sum())}" for k, v in donors.items()))
    def eff(name, var, idxs):
        r = {k: [] for k in ("dz", "dxy", "dz30", "dend", "ci", "cfrac", "l1", "reopen")}
        for i in idxs:
            if (i, name) not in R: continue
            a = R[(i, name)][var]; b = C0[i]
            r["dz"].append(np.abs(a["xyz"][:, 2] - b["xyz"][:, 2]).mean() * 100); r["dxy"].append(np.linalg.norm(a["xyz"][:, :2] - b["xyz"][:, :2], axis=1).mean() * 100); r["dz30"].append((a["xyz"][29, 2] - b["xyz"][29, 2]) * 100); r["dend"].append((a["xyz"][49, 2] - b["xyz"][49, 2]) * 100)
            r["ci"].append(ci(a["grip"]) - ci(b["grip"])); r["cfrac"].append((a["grip"] <= THR).mean() - (b["grip"] <= THR).mean()); r["l1"].append(np.abs(a["grip"] - b["grip"]).mean()); r["reopen"].append(((a["grip"] <= THR)[:-1] & ~(a["grip"] <= THR)[1:]).sum() - ((b["grip"] <= THR)[:-1] & ~(b["grip"] <= THR)[1:]).sum())
        return {k: np.array(v) for k, v in r.items()}
    def line(name, var, idxs, lab=None):
        r = eff(name, var, idxs); n = len(r["dz"])
        if n == 0: return f"   {lab or name:18s} n=0"
        return f"   {lab or name:18s} n={n:3d} | |dz| {r['dz'].mean():5.2f} |dxy| {r['dxy'].mean():5.2f} cm | signed dz@30 {r['dz30'].mean():+5.2f} endpoint z {r['dend'].mean():+5.2f} cm | close idx shift {r['ci'].mean():+6.2f} (|.| {np.abs(r['ci']).mean():5.2f}) | close frac {r['cfrac'].mean():+.3f} | gripper L1 {r['l1'].mean():.3f} | reopen {r['reopen'].mean():+.2f}"
    ALL = list(range(N)); PR = [i for i in range(N) if prim[i]]
    for title, conds in (("STEP 1: camera-wise replacement (mean image)", CONDS1), ("STEP 2: robot vs scene region masking (segmentation, diagnostic only)", CONDS2), ("STEP 3/4: matched donor region swaps (diff = different gripper state / contact state; same = procedure-artefact control)", CONDS3)):
        print(f"\n== {title} ==")
        for var in ("F", "R"):
            print(f"  variant {var} ({'grounded feature held at the original image' if var == 'F' else 'grounded feature recomputed'}) , all frames with a valid intervention:")
            for c in conds: print(line(c, var, ALL))
    print("   G10 reference (full image-token mask, variant F-like): |dz| 7.19 |dxy| 6.98 | close idx +21.96 (|.| 21.96) | gripper |diff| 0.456")
    print("\n== STEP 3/4 transfer ratio: how much of the DONOR's behaviour moves into the receiver? (swapped - receiver) / (donor own chunk - receiver), variant F; pairs with |donor-receiver| > 3cm in dz@30 (or > 5 steps in close idx) ==")
    for name, dk in (("swapR_diff", "Dg"), ("swapR_same", "Ds"), ("swapR_diff_wrist", "Dg"), ("swapS_diff", "Sd"), ("swapS_same", "Ss")):
        tz, tc = [], []
        for i in ALL:
            j = donors[dk][i]
            if (i, name) not in R or j < 0: continue
            a = R[(i, name)]["F"]; rz = (C0[i]["xyz"][29, 2] - ee[i][2]) * 100; dzd = (C0[j]["xyz"][29, 2] - ee[j][2]) * 100; sz = (a["xyz"][29, 2] - ee[i][2]) * 100
            if abs(dzd - rz) > 3: tz.append((sz - rz) / (dzd - rz))
            cd, cr, cs = ci(C0[j]["grip"]), ci(C0[i]["grip"]), ci(a["grip"])
            if abs(cd - cr) > 5: tc.append((cs - cr) / (cd - cr))
        print(f"   {name:18s} dz@30 transfer: n={len(tz):3d} median {np.median(tz) if tz else float('nan'):+.2f} mean {np.mean(tz) if tz else float('nan'):+.2f} | close-idx transfer: n={len(tc):3d} median {np.median(tc) if tc else float('nan'):+.2f} mean {np.mean(tc) if tc else float('nan'):+.2f}")
    print("\n== STEP 4: physical gripper appearance: donor with the OPPOSITE gripper state, receiver split by its own state (variant F, robot swap, all cameras) ==")
    for lab, m in (("receiver gripper CLOSED (donor open)", lambda e: e["grip_raw"] > .5), ("receiver gripper OPEN (donor closed)", lambda e: e["grip_raw"] <= .5)):
        idxs = [i for i in ALL if m(B[i])]; print(line("swapR_diff", "F", idxs, lab)); print(line("swapR_same", "F", idxs, lab.split(" (")[0] + " [control]"))
    print("\n== STEP 5: contact / card-motion strata (exclusive: under-gripper (min EE-card XY < 4cm) > clutter (>=2 cards displaced) > one card displaced > no contact), variant F ==")
    def strat(e): return "under gripper" if e["min_ee_card_xy"] < 0.04 else ("clutter (>=2 displaced)" if e["n_displaced"] >= 2 else ("one card displaced" if e["n_displaced"] == 1 else "no contact"))
    S = {s: [i for i in ALL if strat(B[i]) == s] for s in ("no contact", "one card displaced", "clutter (>=2 displaced)", "under gripper")}; print("   counts:", {k: len(v) for k, v in S.items()})
    for s, idxs in S.items():
        print(f"  [{s}]")
        for c in ("cam_wrist", "cam_all", "robot_all", "scene_all", "cards_all"): print(line(c, "F", idxs))
    print("\n  same-EE-height observational check (original chunks, EE z in 0.20-0.30 m, target >= 6cm below): median rho_30 and dz30 by contact stratum and gripper state")
    for s, idxs in S.items():
        for gl, gm in (("grip open", lambda e: e["grip_raw"] <= .5), ("grip closed", lambda e: e["grip_raw"] > .5)):
            sel = [i for i in idxs if prim[i] and 0.20 <= ee[i][2] <= 0.30 and gm(B[i])]
            if len(sel) >= 5: print(f"   {s:24s} {gl:11s} n={len(sel):3d} rho30 {np.median([(C0[i]['xyz'][29, 2] - ee[i][2]) / (B[i]['gt_anchor_robot'][2] - ee[i][2]) for i in sel]):+.2f} dz30 {np.mean([(C0[i]['xyz'][29, 2] - ee[i][2]) * 100 for i in sel]):+.1f}cm")
    print("\n== STEP 6: close timing under each intervention (variant F): mean first-close idx / fraction of chunk closed / mean reopen events (C0 reference first) ==")
    print(f"   C0 (original)       first close {np.mean([ci(C0[i]['grip']) for i in ALL]):5.1f} | closed frac {np.mean([(C0[i]['grip'] <= THR).mean() for i in ALL]):.3f} | reopen {np.mean([((C0[i]['grip'] <= THR)[:-1] & ~(C0[i]['grip'] <= THR)[1:]).sum() for i in ALL]):.2f}")
    for c in CONDS1 + CONDS2 + CONDS3:
        idxs = [i for i in ALL if (i, c) in R]
        if idxs: print(f"   {c:18s} first close {np.mean([ci(R[(i, c)]['F']['grip']) for i in idxs]):5.1f} (C0 on the same frames {np.mean([ci(C0[i]['grip']) for i in idxs]):5.1f}) | closed frac {np.mean([(R[(i, c)]['F']['grip'] <= THR).mean() for i in idxs]):.3f} | reopen {np.mean([((R[(i, c)]['F']['grip'] <= THR)[:-1] & ~(R[(i, c)]['F']['grip'] <= THR)[1:]).sum() for i in idxs]):.2f}")


if __name__ == "__main__":
    main()
