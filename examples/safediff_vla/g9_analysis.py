#!/usr/bin/env python
"""G9-A analysis: target-branch gain gamma=1 (current) vs gamma=4 (single causal probe, no sweep) on the SAME fixed observations as G8. Chunks: 50 x (x,y,z) absolute action positions in the robot frame (m). Strata and gain definitions identical to g8_analysis."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g8_analysis import SEG, ORDER, AX, AI, THR  # noqa: E402

O9 = Path("outputs/eval/safediff_vla_g9_audit"); O8 = Path("outputs/eval/safediff_vla_g8_audit"); G = (1.0, 4.0)


def stratum(f):
    if f["ee_gt_xy"] < 0.08 or f["any_card_displaced_cm"] > 1.5: return "NEAR/CONTACT"
    if f["t"] <= 20: return "EARLY"
    return "LATE" if f["ee_gt_xy"] <= 0.12 else "MID"


def gain(f, g, axis, seg):
    a, b = SEG[seg]; ref = f["cond"][(g, "A1")]["xyz"]; ia = AI[axis]; out = []
    for c, _ in AX[axis]: da = f["cond"][(g, c)]["anchor_robot"] - f["cond"][(g, "A1")]["anchor_robot"]; out.append(float((f["cond"][(g, c)]["xyz"][a:b] - ref[a:b])[:, ia].mean() / da[ia]))
    return float(np.mean(out))


def resp(f, g, axis, sl):  # |chunk change| in cm for the +-5cm perturbation, mean over the segment slice, along the perturbed axis
    ia = AI[axis]; return float(np.mean([abs((f["cond"][(g, c)]["xyz"][sl][:, ia] - f["cond"][(g, "A1")]["xyz"][sl][:, ia]).mean()) for c, _ in AX[axis]]) * 100)


def quality(chunk, grip, ee):
    d = np.diff(chunk, axis=0); step = np.linalg.norm(d, axis=1); jerk = np.linalg.norm(np.diff(chunk, n=3, axis=0), axis=1); acc = np.linalg.norm(np.diff(chunk, n=2, axis=0), axis=1); closed = grip <= THR
    return {"step": step.mean() * 100, "max_step": step.max() * 100, "jerk": jerk.mean() * 100, "max_jerk": jerk.max() * 100, "acc": acc.mean() * 100, "first_offset": float(np.linalg.norm(chunk[0] - ee)) * 100, "path_len": step.sum() * 100, "net_disp": float(np.linalg.norm(chunk[-1] - chunk[0])) * 100,
            "grip_transitions": int(np.sum(closed[1:] != closed[:-1])), "close_idx": int(np.argmax(closed)) if closed.any() else 50}


def main():
    F = []
    for src in ("g4d", "oracle"):
        for fr in pickle.load(open(O9 / src / "audit9.pkl", "rb")): fr["src"] = src; fr["stratum"] = stratum(fr); F.append(fr)
    F8 = {}
    for src in ("g4d", "oracle"):
        for fr in pickle.load(open(O8 / src / "audit.pkl", "rb")): F8[(src, fr["ep"], fr["t"])] = fr
    chk = [np.abs(f["cond"][(1.0, k)]["xyz"] - F8[(f["src"], f["ep"], f["t"])]["cond"][k]["xyz"]).max() for f in F for k in ("A0", "A1", "A2", "A5", "A7")]
    print(f"frames {len(F)} | strata {{{', '.join(f'{s}: {sum(1 for f in F if f['stratum'] == s)}' for s in ORDER)}}} | SANITY gamma=1 chunks vs G8 chunks (A0,A1,A2,A5,A7): max |diff| = {max(chk) * 100:.5f} cm over {len(chk)} chunks")
    print("\n== ANCHOR-FOLLOWING GAIN (+-5cm anchor -> action), by chunk segment; gamma=1 | gamma=4 ==")
    for ax in ("x", "y", "z"):
        print(f" axis {ax}:   segments " + "".join(f"{s:>16s}" for s in SEG))
        for st in ORDER + ("ALL",):
            fs = [f for f in F if st == "ALL" or f["stratum"] == st]
            if not fs: continue
            print(f"   {st:13s} n={len(fs):3d}  " + "".join(f"{np.mean([gain(f, 1.0, ax, s) for f in fs]):+6.2f} |{np.mean([gain(f, 4.0, ax, s) for f in fs]):+6.2f}  " for s in SEG))
    print("\n== response magnitude to +-5cm (cm; ideal 5.0): endpoint and first-10 mean, ALL frames and by stratum ==")
    for lab, fs in [("ALL", F)] + [(st, [f for f in F if f["stratum"] == st]) for st in ORDER]:
        if not fs: continue
        print(f"   {lab:13s} n={len(fs):3d} | xy-axis(mean of x,y) first10 {np.mean([(resp(f, 1.0, 'x', slice(0, 10)) + resp(f, 1.0, 'y', slice(0, 10))) / 2 for f in fs]):.2f}->{np.mean([(resp(f, 4.0, 'x', slice(0, 10)) + resp(f, 4.0, 'y', slice(0, 10))) / 2 for f in fs]):.2f} endpoint {np.mean([(resp(f, 1.0, 'x', slice(49, 50)) + resp(f, 1.0, 'y', slice(49, 50))) / 2 for f in fs]):.2f}->{np.mean([(resp(f, 4.0, 'x', slice(49, 50)) + resp(f, 4.0, 'y', slice(49, 50))) / 2 for f in fs]):.2f} | z first10 {np.mean([resp(f, 1.0, 'z', slice(0, 10)) for f in fs]):.2f}->{np.mean([resp(f, 4.0, 'z', slice(0, 10)) for f in fs]):.2f} endpoint {np.mean([resp(f, 1.0, 'z', slice(49, 50)) for f in fs]):.2f}->{np.mean([resp(f, 4.0, 'z', slice(49, 50)) for f in fs]):.2f}")
    print("\n== WRONG-CARD COUNTERFACTUAL: A1 (GT anchor) vs A2 (nearest wrong card), gamma 1 -> 4 ==")
    for lab, fs in [("ALL", F)] + [(st, [f for f in F if f["stratum"] == st]) for st in ORDER]:
        if not fs: continue
        r = {}
        for g in G:
            anc = np.array([np.linalg.norm(f["cond"][(g, "A1")]["anchor_robot"][:2] - f["cond"][(g, "A2")]["anchor_robot"][:2]) for f in fs]) * 100
            end = np.array([np.linalg.norm(f["cond"][(g, "A1")]["xyz"][49, :2] - f["cond"][(g, "A2")]["xyz"][49, :2]) for f in fs]) * 100; f10 = np.array([np.linalg.norm((f["cond"][(g, "A1")]["xyz"][:10, :2] - f["cond"][(g, "A2")]["xyz"][:10, :2]).mean(0)) for f in fs]) * 100
            def proj_sep(f):  # signed endpoint separation along the GT->wrong direction (1 = chunks separate one-for-one along the anchor axis)
                d = f["cond"][(g, "A2")]["anchor_robot"][:2] - f["cond"][(g, "A1")]["anchor_robot"][:2]; return float((f["cond"][(g, "A2")]["xyz"][49, :2] - f["cond"][(g, "A1")]["xyz"][49, :2]) @ d / (d @ d))
            def cosv(f, k): ee = f["ee_robot"][:2]; v = f["cond"][(g, k)]["xyz"][:10, :2].mean(0) - ee; d = f["cond"][(g, k)]["anchor_robot"][:2] - ee; return float(v @ d / (np.linalg.norm(v) * np.linalg.norm(d) + 1e-9))
            r[g] = (anc.mean(), end.mean(), f10.mean(), np.mean([proj_sep(f) for f in fs]), np.mean([cosv(f, "A1") for f in fs]), np.mean([cosv(f, "A2") for f in fs]), np.median(end), np.mean(end > 8))
        print(f"   {lab:13s} n={len(fs):3d} | anchor sep {r[1.0][0]:.1f}cm | endpoint sep {r[1.0][1]:.1f}->{r[4.0][1]:.1f}cm (median {r[1.0][6]:.1f}->{r[4.0][6]:.1f}; frac>8cm {r[1.0][7]:.2f}->{r[4.0][7]:.2f}) | first10 sep {r[1.0][2]:.1f}->{r[4.0][2]:.1f}cm | signed along-axis endpoint gain {r[1.0][3]:+.2f}->{r[4.0][3]:+.2f} | cos(first10, dir to anchor) GT {r[1.0][4]:+.2f}->{r[4.0][4]:+.2f} wrong {r[1.0][5]:+.2f}->{r[4.0][5]:+.2f}")
    print("\n== does the chunk endpoint move toward the GT anchor? |endpoint(A1) - GT anchor| (cm) ==")
    for lab, fs in [("ALL", F)] + [(st, [f for f in F if f["stratum"] == st]) for st in ORDER]:
        if not fs: continue
        e = lambda g, sl: np.mean([np.linalg.norm(f["cond"][(g, "A1")]["xyz"][49][sl] - f["cond"][(g, "A1")]["anchor_robot"][sl]) for f in fs]) * 100
        print(f"   {lab:13s} xy {e(1.0, slice(0, 2)):.1f}->{e(4.0, slice(0, 2)):.1f} cm | z {e(1.0, slice(2, 3)):.1f}->{e(4.0, slice(2, 3)):.1f} cm | (EE-to-anchor xy {np.mean([np.linalg.norm(f['ee_robot'][:2] - f['cond'][(1.0, 'A1')]['anchor_robot'][:2]) for f in fs]) * 100:.1f}, z {np.mean([abs(f['ee_robot'][2] - f['cond'][(1.0, 'A1')]['anchor_robot'][2]) for f in fs]) * 100:.1f})")
    print("\n== ACTION-QUALITY GUARD (same observations; A0 = G4d's own anchor, A1 = GT anchor) ==")
    Q = {g: {k: [quality(f["cond"][(g, k)]["xyz"], f["cond"][(g, k)]["grip"], f["ee_robot"]) for f in F] for k in ("A0", "A1")} for g in G}; keys = ("step", "max_step", "jerk", "max_jerk", "acc", "first_offset", "path_len", "net_disp", "grip_transitions")
    print(f"   {'metric (cm per step / cm)':26s}" + "".join(f"{'g1 ' + k:>11s}{'g4 ' + k:>11s}" for k in ("A0", "A1")))
    for m in keys: print(f"   {m:26s}" + "".join(f"{np.mean([q[m] for q in Q[1.0][k]]):11.2f}{np.mean([q[m] for q in Q[4.0][k]]):11.2f}" for k in ("A0", "A1")))
    for m in ("max_step", "jerk", "max_jerk", "first_offset"):
        thr = np.percentile([q[m] for k in ("A0", "A1") for q in Q[1.0][k]], 99); print(f"   gamma=4 chunks above the gamma=1 99th percentile of {m} ({thr:.2f}): {np.mean([q[m] > thr for k in ('A0', 'A1') for q in Q[4.0][k]]):.3f} (gamma=1 itself: 0.010)")
    dchunk = lambda k: np.mean([np.linalg.norm(f["cond"][(4.0, k)]["xyz"] - f["cond"][(1.0, k)]["xyz"], axis=1).mean() for f in F]) * 100; print(f"   mean |chunk(gamma=4) - chunk(gamma=1)| for the SAME anchor: A0 {dchunk('A0'):.2f}cm | A1 {dchunk('A1'):.2f}cm | gripper chunk diff A1: {np.mean([np.abs(f['cond'][(4.0, 'A1')]['grip'] - f['cond'][(1.0, 'A1')]['grip']).mean() for f in F]):.4f} | close idx change A1 (mean |.|): {np.mean([abs(q4['close_idx'] - q1['close_idx']) for q1, q4 in zip(Q[1.0]['A1'], Q[4.0]['A1'])]):.2f} steps")
    print("   close idx (A1) gamma=1 -> 4 (mean, non-EARLY):", np.mean([q["close_idx"] for q, f in zip(Q[1.0]["A1"], F) if f["stratum"] != "EARLY"]).round(2), "->", np.mean([q["close_idx"] for q, f in zip(Q[4.0]["A1"], F) if f["stratum"] != "EARLY"]).round(2))
    json.dump({"note": "see stdout"}, open(O9 / "g9_analysis_done.json", "w"))


if __name__ == "__main__":
    main()
