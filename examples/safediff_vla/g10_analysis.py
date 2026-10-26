#!/usr/bin/env python
"""G10-Z analysis of the fixed-observation audit (see g10_capture_audit.py). Reference chunk Z0 = decoder with GT anchor + own grounded feature. Chunk = 50 x (x,y,z) absolute action positions (robot frame, m); z in cm below.
Strata as in G8 (EARLY t<=20 | NEAR/CONTACT EE-GT_xy<8cm or card displaced | LATE 8-12cm | MID otherwise)."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g8_analysis import ORDER, THR  # noqa: E402

O10 = Path("outputs/eval/safediff_vla_g10_audit"); O9 = Path("outputs/eval/safediff_vla_g9_audit"); rng = np.random.default_rng(0)
SEG = {"1-5": (0, 5), "6-10": (5, 10), "11-20": (10, 20), "21-30": (20, 30), "31-50": (30, 50)}


def stratum(f):
    if f["ee_gt_xy"] < 0.08 or f["any_card_displaced_cm"] > 1.5: return "NEAR/CONTACT"
    if f["t"] <= 20: return "EARLY"
    return "LATE" if f["ee_gt_xy"] <= 0.12 else "MID"


def r2(y, X):
    X = np.c_[np.ones(len(y)), X]; b, *_ = np.linalg.lstsq(X, y, rcond=None); return float(1 - (y - X @ b).var() / y.var()) if y.var() > 1e-12 else float("nan")


def close_idx(g): c = g <= THR; return int(np.argmax(c)) if c.any() else 50


def main():
    F = []
    for src in ("g4d", "oracle"):
        for fr in pickle.load(open(O10 / src / "audit10.pkl", "rb")): fr["src"] = src; fr["stratum"] = stratum(fr); F.append(fr)
    print("frames", len(F), {s: sum(1 for f in F if f["stratum"] == s) for s in ORDER}, "| phase1:", sum(f["phase"] == 1 for f in F))
    z0 = lambda f: f["cond"]["Z0"]["xyz"]; ee = lambda f: f["ee_robot"]
    # ---------------- STEP 1: natural trajectory regression (captured frames) ----------------
    print("\n== STEP 1: R^2 of the decoder's predicted z (Z0 chunk) on candidate predictors, 797 captured frames ==")
    X = {"EE z": np.array([ee(f)[2] for f in F]), "vertical gap (EE z - GT z)": np.array([f["ee_gt_z"] for f in F]), "phase": np.array([f["phase"] for f in F], float), "frame time t": np.array([f["t"] for f in F], float), "target z (GT)": np.array([f["gt_anchor_robot"][2] for f in F]), "XY dist EE-target": np.array([f["ee_gt_xy"] for f in F]), "gripper state": np.array([f["state7"][6] for f in F])}
    Y = {"first action z": np.array([z0(f)[0, 2] for f in F]), "first-5 mean z": np.array([z0(f)[:5, 2].mean() for f in F]), "first-10 mean z": np.array([z0(f)[:10, 2].mean() for f in F]), "endpoint z": np.array([z0(f)[49, 2] for f in F]), "min z": np.array([z0(f)[:, 2].min() for f in F]), "close idx": np.array([close_idx(f["cond"]["Z0"]["grip"]) for f in F], float)}
    print(f"   {'':16s}" + "".join(f"{k:>16s}" for k in X) + f"{'EEz+t+XYd+grip':>16s}")
    for yk, y in Y.items(): print(f"   {yk:16s}" + "".join(f"{r2(y, x):16.3f}" for x in X.values()) + f"{r2(y, np.c_[X['EE z'], X['frame time t'], X['XY dist EE-target'], X['gripper state']]):16.3f}")
    print(f"   (target z std over these frames: {X['target z (GT)'].std() * 1000:.3f} mm -> its R^2 is numerical noise)   mean first-action z - EE z: {np.mean([z0(f)[0, 2] - ee(f)[2] for f in F]) * 100:+.1f} cm ; first-5: {np.mean([z0(f)[:5, 2].mean() - ee(f)[2] for f in F]) * 100:+.1f} cm")
    # ---------------- STEP 2: in-distribution state-z perturbation ----------------
    print("\n== STEP 2: EE-z STATE perturbation (+-1cm and +-2cm; image/language/target fixed). gain = d(chunk z)/d(state z) (1.0 = chunk z shifts one-for-one with the EE z); S1 = decoder state input only, S2 = whole observation.state (decoder AND VLM prefix, latents recomputed) ==")
    for var in ("S1", "S2"):
        for cm in (1, 2):
            print(f"  {var} +-{cm}cm  segments " + "".join(f"{s:>9s}" for s in SEG) + "   | endpoint | XY leakage(first10 / full cm per cm) | close idx shift (+ vs -)")
            for st in ORDER + ("ALL",):
                fs = [f for f in F if st == "ALL" or f["stratum"] == st]
                if not fs: continue
                g = lambda a, b: np.mean([((f["cond"][f"{var}_p{cm}"]["xyz"][a:b, 2] - f["cond"][f"{var}_m{cm}"]["xyz"][a:b, 2]).mean()) / (2 * cm / 100) for f in fs])
                lk = lambda a, b: np.mean([np.linalg.norm((f["cond"][f"{var}_p{cm}"]["xyz"][a:b, :2] - f["cond"][f"{var}_m{cm}"]["xyz"][a:b, :2]).mean(0)) / (2 * cm / 100) for f in fs])
                ci = np.mean([close_idx(f["cond"][f"{var}_p{cm}"]["grip"]) - close_idx(f["cond"][f"{var}_m{cm}"]["grip"]) for f in fs])
                print(f"     {st:13s} n={len(fs):3d}  " + "".join(f"{g(*SEG[s]):+9.2f}" for s in SEG) + f"  | {g(49, 50):+.2f}  | {lk(0, 10):.2f} / {lk(0, 50):.2f} | {ci:+.1f}")
    # ---------------- STEP 3: branch ablations ----------------
    print("\n== STEP 3: ablations vs Z0 (same observation). mean |Delta chunk| in cm; z = vertical, xy = horizontal; close idx = first chunk index with gripper <= threshold ==")
    conds = ("Z1", "Z2a", "Z2p", "Z2m", "Z3", "Z4", "V1", "V2"); names = {"Z1": "target_xyz branch OFF", "Z2a": "target z := demo mean", "Z2p": "target z +0.5mm", "Z2m": "target z -0.5mm", "Z3": "grounded feature OFF", "Z4": "target branch AND feature OFF", "V1": "image tokens masked", "V2": "text tokens masked"}
    for st in ORDER + ("ALL",):
        fs = [f for f in F if st == "ALL" or f["stratum"] == st]
        if not fs: continue
        print(f"  {st} (n={len(fs)}):   cond                          | z full  z first10  z endpoint  z min-shift | xy full  xy endpoint | close idx change | gripper |diff|")
        for c in conds:
            dz = lambda sl: np.mean([np.abs(f["cond"][c]["xyz"][sl, 2] - z0(f)[sl, 2]).mean() for f in fs]) * 100; dxy = lambda sl: np.mean([np.linalg.norm(f["cond"][c]["xyz"][sl, :2] - z0(f)[sl, :2], axis=1).mean() for f in fs]) * 100
            print(f"        {c:4s} {names[c]:28s} | {dz(slice(0, 50)):5.2f}  {dz(slice(0, 10)):6.2f}  {dz(slice(49, 50)):8.2f}  {np.mean([f['cond'][c]['xyz'][:, 2].min() - z0(f)[:, 2].min() for f in fs]) * 100:+9.2f} | {dxy(slice(0, 50)):5.2f}  {dxy(slice(49, 50)):8.2f} | {np.mean([close_idx(f['cond'][c]['grip']) - close_idx(f['cond']['Z0']['grip']) for f in fs]):+6.2f} (mean |.| {np.mean([abs(close_idx(f['cond'][c]['grip']) - close_idx(f['cond']['Z0']['grip'])) for f in fs]):.2f}) | {np.mean([np.abs(f['cond'][c]['grip'] - f['cond']['Z0']['grip']).mean() for f in fs]):.3f}")
        if st == "ALL": print("        reference scale: mean z extent of the Z0 chunk (max-min) %.1f cm; mean |z(Z0) - EE z| %.1f cm" % (np.mean([np.ptp(z0(f)[:, 2]) for f in F]) * 100, np.mean([np.abs(z0(f)[:, 2] - ee(f)[2]).mean() for f in F]) * 100))
    # ---------------- STEP 4: branch norms ----------------
    print("\n== STEP 4: contribution norms at Z0 (mean per-token L2 norm; no causal reading without the ablations above) ==")
    ks = ["q_base", "q_state", "q_target", "q_feature", "q_phase"] + [f"L{i}_{b}" for i in range(4) for b in ("selfattn", "crossattn", "ffn")] + ["final_hidden"]
    print(f"   {'stratum':13s}" + "".join(f"{k.replace('selfattn', 'SA').replace('crossattn', 'CA'):>10s}" for k in ks))
    for st in ORDER + ("ALL",):
        fs = [f for f in F if st == "ALL" or f["stratum"] == st]
        if fs: print(f"   {st:13s}" + "".join(f"{np.mean([f['norms'][k] for f in fs if k in f['norms']]):10.2f}" for k in ks))
    n = {k: np.mean([f["norms"][k] for f in F]) for k in ks}; qs = n["q_base"] + n["q_state"] + n["q_target"] + n["q_feature"] + n["q_phase"]; print("   query-component norm shares (sum of norms, not vector sum): " + ", ".join(f"{k[2:]} {n[k] / qs:.2f}" for k in ("q_base", "q_state", "q_target", "q_feature", "q_phase")))
    # ---------------- STEP 5: vertical template ----------------
    print("\n== STEP 5: VERTICAL TEMPLATE. dz_k = chunk z_k - current EE z (cm), Z0 chunks ==")
    DZ = np.array([(z0(f)[:, 2] - ee(f)[2]) * 100 for f in F]); ep = np.array([(f["src"], f["ep"]) for f in F], dtype=object); tgt = np.array([f["target_name"] for f in F]); st_arr = np.array([f["stratum"] for f in F]); ph = np.array([f["phase"] for f in F])
    for k in (0, 4, 9, 19, 29, 49): print(f"   chunk idx {k + 1:2d}: mean dz {DZ[:, k].mean():+6.1f} cm, std across frames {DZ[:, k].std():5.1f} cm")
    def pair_corr(mask, diff_ep=False, n=3000):
        idx = np.where(mask)[0]; cs = []
        for _ in range(n):
            i, j = rng.choice(idx, 2, replace=False)
            if diff_ep and tuple(ep[i]) == tuple(ep[j]): continue
            a, b = DZ[i], DZ[j]
            if a.std() > 1e-6 and b.std() > 1e-6: cs.append(np.corrcoef(a, b)[0, 1])
        return np.mean(cs), np.median(cs)
    for lab, mask in [("ALL frames", np.ones(len(F), bool))] + [(s, st_arr == s) for s in ORDER] + [("PRE_GRASP", ph == 0), ("POST_GRASP", ph == 1)]:
        if mask.sum() > 5: m, md = pair_corr(mask, True); mt = DZ[mask].mean(0); res = ((DZ[mask] - mt) ** 2).sum() / ((DZ[mask] - DZ.mean()) ** 2).sum(); print(f"   {lab:13s} n={int(mask.sum()):3d} | pairwise corr of dz curves (different episodes) mean {m:+.2f} median {md:+.2f} | variance NOT explained by this group's mean template: {res:.2f} | template dz at idx 1/10/30/50: {mt[0]:+.1f} {mt[9]:+.1f} {mt[29]:+.1f} {mt[49]:+.1f}")
    # same frame, different target/anchor (G9 audit, gamma=1: A0 G4d anchor, A1 GT, A2 wrong card)
    F9 = []
    for src in ("g4d", "oracle"): F9 += pickle.load(open(O9 / src / "audit9.pkl", "rb"))
    d_anc = np.mean([np.sqrt(((np.array([f["cond"][(1.0, a)]["xyz"][:, 2] for a in ("A0", "A1", "A2")]) - np.array([f["cond"][(1.0, a)]["xyz"][:, 2] for a in ("A0", "A1", "A2")]).mean(0)) ** 2).mean()) for f in F9]) * 100
    print(f"   SAME observation, three different targets (G4d pred / GT / wrong card): RMS spread of z over the chunk = {d_anc:.2f} cm   vs   across-frame spread of dz = {DZ.std(0).mean():.2f} cm")
    print(f"   target identity effect on dz (eta^2 of mean dz over chunk, {len(set(tgt))} referents; chance ~ {len(set(tgt)) / len(tgt):.2f}): {sum((tgt == r).sum() * (DZ.mean(1)[tgt == r].mean() - DZ.mean()) ** 2 for r in set(tgt)) / ((DZ.mean(1) - DZ.mean()) ** 2).sum():.3f}")
    # ---------------- gripper / vertical coupling (observe only) ----------------
    print("\n== GRIPPER TIMING vs VERTICAL TEMPLATE (observe only) ==")
    ci = np.array([close_idx(f["cond"]["Z0"]["grip"]) for f in F]); tm = np.array([int(np.argmin(z0(f)[:, 2])) for f in F]); gtz = np.array([f["gt_anchor_robot"] for f in F]); d3 = []
    for f, c in zip(F, ci):
        p = z0(f)[min(c, 49)]; d3.append(np.linalg.norm(p - f["gt_anchor_robot"]) * 100)
    d3 = np.array(d3); mask = np.array([f["stratum"] != "EARLY" for f in F])
    print(f"   first predicted close idx: mean {ci.mean():.1f} (non-EARLY {ci[mask].mean():.1f}) | idx of the predicted z minimum: mean {tm.mean():.1f} | corr(close idx, z-min idx) {np.corrcoef(ci, tm)[0, 1]:+.2f} | corr(close idx, EE-GT xy dist) {np.corrcoef(ci, X['XY dist EE-target'])[0, 1]:+.2f} | corr(close idx, frame time t) {np.corrcoef(ci, X['frame time t'])[0, 1]:+.2f}")
    print(f"   3D distance between the predicted EE position AT the predicted close and the GT anchor: mean {d3.mean():.1f} cm, median {np.median(d3):.1f} cm, frac <3cm {np.mean(d3 < 3):.2f}")
    print(f"   by stratum (mean close idx / mean 3D dist at predicted close): " + " | ".join(f"{s} {ci[st_arr == s].mean():.1f} / {d3[st_arr == s].mean():.1f}cm" for s in ORDER))
    print(f"   close idx under ablations (mean, all frames): Z0 {np.mean([close_idx(f['cond']['Z0']['grip']) for f in F]):.1f} | " + " | ".join(f"{c} {np.mean([close_idx(f['cond'][c]['grip']) for f in F]):.1f}" for c in conds))


if __name__ == "__main__":
    main()
