#!/usr/bin/env python
"""G11-D: fixed-observation comparison V2 decoder (residual OFF) vs residual-conditioned decoder (RES) on the SAME 797 observations as G8/G9/G10. Gains, wrong-card separation, vertical closing ratio, and the action-quality guard.
Gain definitions as in g8/g9 (+-5cm anchor perturbation, mean over a chunk segment, signed, 1.0 = one-for-one). Closing ratio rho_k = (chunk_z[k] - EE_z) / e_z with e_z = anchor_z - EE_z (frames with e_z < -6cm, i.e. target well below the EE); demonstration values by task are in g11_audit."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g8_analysis import SEG, ORDER, AX, AI, THR  # noqa: E402
from g9_analysis import quality  # noqa: E402

O11 = Path("outputs/eval/safediff_vla_g11_audit"); M = ("V2", "RES")


def stratum(f):
    if f["ee_gt_xy"] < 0.08 or f["any_card_displaced_cm"] > 1.5: return "NEAR/CONTACT"
    if f["t"] <= 20: return "EARLY"
    return "LATE" if f["ee_gt_xy"] <= 0.12 else "MID"


def gain(f, m, axis, seg):
    a, b = SEG[seg]; ref = f["cond"][(m, "A1")]["xyz"]; ia = AI[axis]; out = []
    for c, _ in AX[axis]: da = f["cond"][(m, c)]["anchor_robot"] - f["cond"][(m, "A1")]["anchor_robot"]; out.append(float((f["cond"][(m, c)]["xyz"][a:b] - ref[a:b])[:, ia].mean() / da[ia]))
    return float(np.mean(out))


def sgain(f, m, seg): a, b = SEG[seg]; return float((f["cond"][(m, "Sp2")]["xyz"][a:b, 2] - f["cond"][(m, "Sm2")]["xyz"][a:b, 2]).mean() / 0.04)


def main():
    F = []
    for src in ("g4d", "oracle"):
        for fr in pickle.load(open(O11 / src / "audit11.pkl", "rb")): fr["src"] = src; fr["stratum"] = stratum(fr); F.append(fr)
    print("frames", len(F), {s: sum(1 for f in F if f["stratum"] == s) for s in ORDER})
    print("\n== anchor-following gain (+-5cm), by chunk segment: V2 | RES ==")
    for ax in ("x", "y", "z"):
        print(f" axis {ax}:  segments " + "".join(f"{s:>17s}" for s in SEG))
        for st in ORDER + ("ALL",):
            fs = [f for f in F if st == "ALL" or f["stratum"] == st]
            if fs: print(f"   {st:13s} n={len(fs):3d} " + "".join(f"{np.mean([gain(f, 'V2', ax, s) for f in fs]):+7.2f}|{np.mean([gain(f, 'RES', ax, s) for f in fs]):+6.2f} " for s in SEG))
    print("\n== response magnitude (cm; ideal 5.0) to +-5cm anchor: ALL frames ==")
    for m in M:
        r = lambda axis, sl: np.mean([np.mean([abs((f["cond"][(m, c)]["xyz"][sl][:, AI[axis]] - f["cond"][(m, "A1")]["xyz"][sl][:, AI[axis]]).mean()) for c, _ in AX[axis]]) for f in F]) * 100
        print(f"   {m:4s} xy first10 {(r('x', slice(0, 10)) + r('y', slice(0, 10))) / 2:.2f} endpoint {(r('x', slice(49, 50)) + r('y', slice(49, 50))) / 2:.2f} | z first10 {r('z', slice(0, 10)):.2f} endpoint {r('z', slice(49, 50)):.2f}")
    print("\n== relative-error intervention: EE-z state +-2cm (target fixed) -> d(chunk z)/d(EE z), V2 | RES ==")
    for st in ORDER + ("ALL",):
        fs = [f for f in F if st == "ALL" or f["stratum"] == st]
        if fs: print(f"   {st:13s} n={len(fs):3d} " + "".join(f"{np.mean([sgain(f, 'V2', s) for f in fs]):+7.2f}|{np.mean([sgain(f, 'RES', s) for f in fs]):+6.2f} " for s in SEG))
    print("\n== wrong-card counterfactual (A1 GT vs A2 wrong card): V2 -> RES ==")
    for lab, fs in [("ALL", F)] + [(s, [f for f in F if f["stratum"] == s]) for s in ORDER]:
        if not fs: continue
        r = {}
        for m in M:
            end = np.array([np.linalg.norm(f["cond"][(m, "A1")]["xyz"][49, :2] - f["cond"][(m, "A2")]["xyz"][49, :2]) for f in fs]) * 100; f10 = np.array([np.linalg.norm((f["cond"][(m, "A1")]["xyz"][:10, :2] - f["cond"][(m, "A2")]["xyz"][:10, :2]).mean(0)) for f in fs]) * 100
            def cosv(f, k): ee = f["ee_robot"][:2]; v = f["cond"][(m, k)]["xyz"][:10, :2].mean(0) - ee; d = f["cond"][(m, k)]["anchor_robot"][:2] - ee; return float(v @ d / (np.linalg.norm(v) * np.linalg.norm(d) + 1e-9))
            r[m] = (end.mean(), f10.mean(), np.mean([cosv(f, "A1") for f in fs]), np.mean([cosv(f, "A2") for f in fs]), np.mean(end > 8))
        print(f"   {lab:13s} n={len(fs):3d} | endpoint sep {r['V2'][0]:.1f}->{r['RES'][0]:.1f}cm (frac>8cm {r['V2'][4]:.2f}->{r['RES'][4]:.2f}) | first10 sep {r['V2'][1]:.1f}->{r['RES'][1]:.1f}cm | cos(first10, dir to anchor) GT {r['V2'][2]:+.2f}->{r['RES'][2]:+.2f} wrong {r['V2'][3]:+.2f}->{r['RES'][3]:+.2f}")
    print("\n== vertical closing ratio rho_k = (chunk z_k - EE z)/e_z (GT anchor A1) for frames with e_z < -6cm (target well below the EE) ==")
    for lab, fs in [("ALL", [f for f in F if f["ee_gt_z"] > .06])] + [(s, [f for f in F if f["stratum"] == s and f["ee_gt_z"] > .06]) for s in ORDER]:
        if len(fs) < 3: continue
        g = lambda m, k: np.median([(f["cond"][(m, "A1")]["xyz"][k, 2] - f["ee_robot"][2]) / (f["cond"][(m, "A1")]["anchor_robot"][2] - f["ee_robot"][2]) for f in fs]); zs = lambda m, k: np.mean([(f["cond"][(m, "A1")]["xyz"][k, 2] - f["ee_robot"][2]) * 100 for f in fs])
        print(f"   {lab:13s} n={len(fs):3d} median rho at k=10/20/30/50: V2 " + " ".join(f"{g('V2', k):+.2f}" for k in (9, 19, 29, 49)) + " | RES " + " ".join(f"{g('RES', k):+.2f}" for k in (9, 19, 29, 49)) + " || mean (z_k - EE z) cm V2 " + " ".join(f"{zs('V2', k):+.1f}" for k in (9, 29, 49)) + " | RES " + " ".join(f"{zs('RES', k):+.1f}" for k in (9, 29, 49)) + f" || mean e_z {np.mean([-f['ee_gt_z'] for f in fs]) * 100:.1f}cm")
    print("   demonstration reference (G11 audit, median rho): EARLY k=10/20/30/49 ~ 0.07/0.35/0.55/0.95 ; MID ~ 0.17/0.78/0.94/0.69 (task-independent)")
    print("\n== ACTION-QUALITY GUARD (same observations; A0 = G4d anchor, A1 = GT anchor) ==")
    Q = {m: {k: [quality(f["cond"][(m, k)]["xyz"], f["cond"][(m, k)]["grip"], f["ee_robot"]) for f in F] for k in ("A0", "A1")} for m in M}
    print(f"   {'metric':16s}" + "".join(f"{m + ' ' + k:>11s}" for k in ("A0", "A1") for m in M))
    for mt in ("step", "max_step", "jerk", "max_jerk", "first_offset", "path_len", "net_disp", "grip_transitions"): print(f"   {mt:16s}" + "".join(f"{np.mean([q[mt] for q in Q[m][k]]):11.2f}" for k in ("A0", "A1") for m in M))
    for mt in ("max_step", "jerk", "first_offset"):
        thr = np.percentile([q[mt] for k in ("A0", "A1") for q in Q["V2"][k]], 99); print(f"   RES chunks above the V2 99th percentile of {mt} ({thr:.2f}): {np.mean([q[mt] > thr for k in ('A0', 'A1') for q in Q['RES'][k]]):.3f}")
    print(f"   mean |chunk(RES) - chunk(V2)| same anchor: A0 {np.mean([np.linalg.norm(f['cond'][('RES', 'A0')]['xyz'] - f['cond'][('V2', 'A0')]['xyz'], axis=1).mean() for f in F]) * 100:.2f}cm | A1 {np.mean([np.linalg.norm(f['cond'][('RES', 'A1')]['xyz'] - f['cond'][('V2', 'A1')]['xyz'], axis=1).mean() for f in F]) * 100:.2f}cm | gripper chunk diff A1 {np.mean([np.abs(f['cond'][('RES', 'A1')]['grip'] - f['cond'][('V2', 'A1')]['grip']).mean() for f in F]):.4f} | close-idx change A1 (mean |.|) {np.mean([abs(q2['close_idx'] - q1['close_idx']) for q1, q2 in zip(Q['V2']['A1'], Q['RES']['A1'])]):.2f}")
    print(f"   endpoint distance to the GT anchor (A1): xy V2 {np.mean([np.linalg.norm(f['cond'][('V2', 'A1')]['xyz'][49, :2] - f['cond'][('V2', 'A1')]['anchor_robot'][:2]) for f in F]) * 100:.1f} -> RES {np.mean([np.linalg.norm(f['cond'][('RES', 'A1')]['xyz'][49, :2] - f['cond'][('RES', 'A1')]['anchor_robot'][:2]) for f in F]) * 100:.1f} cm | z V2 {np.mean([abs(f['cond'][('V2', 'A1')]['xyz'][49, 2] - f['cond'][('V2', 'A1')]['anchor_robot'][2]) for f in F]) * 100:.1f} -> RES {np.mean([abs(f['cond'][('RES', 'A1')]['xyz'][49, 2] - f['cond'][('RES', 'A1')]['anchor_robot'][2]) for f in F]) * 100:.1f} cm")


if __name__ == "__main__":
    main()
