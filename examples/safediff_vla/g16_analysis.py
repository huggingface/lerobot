#!/usr/bin/env python
"""G16-F analysis on the fixed observations (797 bank frames; G15 subset = closed & EE-GT>6cm & open donor). Models: v2 baseline | g11 residual | g16 geometry | g4 (V2 + target-branch gamma=4, DIAGNOSTIC UPPER BOUND ONLY). Errors vs the GT grasp keypoint, cm."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g15_analysis import metrics  # noqa: E402
from g9_analysis import quality  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g16"); MODELS = ("v2", "g11", "g16", "g4"); K = {"10": 9, "20": 19, "30": 29, "50": 49}


def load():
    R = {}
    for m, f in (("v2", "capture_v2_g11_g16_g4.pkl"),):
        R = pickle.load(open(OUT / f, "rb"))
    return R


def stratum(f):
    if f["ee_gt_xy"] < 0.08 or f["any_disp_cm"] > 1.5: return "NEAR/CONTACT"
    if f["t"] <= 20: return "EARLY"
    return "LATE" if f["ee_gt_xy"] <= 0.12 else "MID"


def mt(f, m, key):  # metrics via g15.metrics on a view
    v = {"out": {("x",): f["out"][key]}, "gt": f["gt"], "ee": f["ee"]}; r = metrics(v, ("x",)); r["path_len"] = quality(f["out"][key]["xyz"], f["out"][key]["grip"], f["ee"])["path_len"]; return r


def row(lab, vals, fmt="{:8.2f}"): print(f"   {lab:22s}" + "".join(fmt.format(v) for v in vals))


def main():
    R = load(); F = {m: R[m] for m in MODELS}; N = len(F["v2"]); hdr = "".join(f"{m:>8s}" for m in MODELS); sub = [i for i in range(N) if F["v2"][i]["sub"]]
    pre = [i for i in range(N) if F["v2"][i]["phase"] == 0 and F["v2"][i]["grip_raw"] <= 0.5]; wr = [i for i in range(N) if F["v2"][i]["wrong"] is not None]
    print(f"frames {N} | PRE_GRASP & gripper open {len(pre)} | G15 subset (closed & far & donor) {len(sub)} | wrong-card available {len(wr)}")
    # ---- XY steering ----
    print("\n== XY STEERING: GT vs WRONG-card anchor (frames with a wrong card; PRE_GRASP-open frames in the second block) ==")
    for nm, idx in (("all frames", wr), ("PRE_GRASP open", [i for i in wr if i in set(pre)])):
        print(f" [{nm}] n={len(idx)}   metric" + " " * 12 + hdr)
        def sep(m, sl):
            o = []
            for i in idx:
                a, b = F[m][i]["out"][("GT", 0)]["xyz"], F[m][i]["out"][("WR", 0)]["xyz"]; o.append(np.linalg.norm((a[sl, :2] - b[sl, :2]).mean(0)) * 100)
            return np.mean(o)
        row("first10 separation cm", [sep(m, slice(0, 10)) for m in MODELS]); row("endpoint separation cm", [sep(m, slice(49, 50)) for m in MODELS])
        asep = np.mean([np.linalg.norm(F["v2"][i]["gt"][:2] - F["v2"][i]["wrong"][:2]) * 100 for i in idx]); print(f"   (anchor separation GT-vs-wrong: {asep:.1f} cm)")
        def cos(m, k):
            o = []
            for i in idx:
                f = F[m][i]; x = f["out"][(k, 0)]["xyz"]; anc = f["gt"] if k == "GT" else f["wrong"]; v = x[:10, :2].mean(0) - f["ee"][:2]; d = anc[:2] - f["ee"][:2]; o.append(v @ d / (np.linalg.norm(v) * np.linalg.norm(d) + 1e-9))
            return np.mean(o)
        row("cos(first10, ->GT)", [cos(m, "GT") for m in MODELS], "{:8.3f}"); row("cos(first10, ->wrong)", [cos(m, "WR") for m in MODELS], "{:8.3f}")
    print("\n== XY ERROR to the GT anchor (GT-anchor conditioning), cm, PRE_GRASP-open frames and by stratum ==")
    for nm, idx in (("PRE_GRASP open", pre), ("all frames", list(range(N)))) + tuple((s, [i for i in pre if stratum(F["v2"][i]) == s]) for s in ("MID", "LATE", "NEAR/CONTACT")):
        if not idx: continue
        print(f" [{nm}] n={len(idx)}   " + hdr)
        for k, ii in K.items(): row(f"XY error @{k}", [np.mean([np.linalg.norm(F[m][i]["out"][("GT", 0)]["xyz"][ii, :2] - F[m][i]["gt"][:2]) for i in idx]) * 100 for m in MODELS])
        row("start (EE-GT xy)", [np.mean([F[m][i]["ee_gt_xy"] for i in idx]) * 100 for m in MODELS])
    # ---- Z gating ----
    print("\n== Z / READINESS GATING: PRE_GRASP-open frames with e_z < -3cm (target below EE); rho = (mean chunk z over steps 0-9 - EE z)/e_z, rho30 = (z[29]-EE z)/e_z; demonstrations (dataset audit, next10): rho_z 0.11 | 0.20 | 0.49 for r_xy >10 | 5-10 | <5 cm ==")
    bins = [(">10cm", lambda r: r > .10), ("5-10cm", lambda r: (r > .05) & (r <= .10)), ("<5cm", lambda r: r <= .05)]; ez = lambda i: F["v2"][i]["gt"][2] - F["v2"][i]["ee"][2]; rxy = lambda i: F["v2"][i]["ee_gt_xy"]
    for nm, ff in (("first10 rho", lambda f, i: (f["out"][("GT", 0)]["xyz"][:10, 2].mean() - f["ee"][2]) / ez(i)), ("rho30", lambda f, i: (f["out"][("GT", 0)]["xyz"][29, 2] - f["ee"][2]) / ez(i)), ("dz30 cm", lambda f, i: (f["out"][("GT", 0)]["xyz"][29, 2] - f["ee"][2]) * 100)):
        print(f" [{nm}]   r_xy bin     n   " + hdr)
        for bn, bf in bins:
            idx = [i for i in pre if ez(i) < -.03 and bf(rxy(i))]
            if idx: print(f"   {bn:10s} {len(idx):4d}   " + "".join(f"{np.median([ff(F[m][i], i) for i in idx]) if 'rho' in nm else np.mean([ff(F[m][i], i) for i in idx]):8.2f}" for m in MODELS))
    # ---- 3D on G15 subset ----
    for nm, idx, key in (("G15 subset, GT anchor, ORIGINAL wrist (cell A analogue)", sub, ("GT", 0)), ("G15 subset, GT anchor, OPEN-wrist patch (cell C analogue)", sub, ("GT", 1)), ("ALL frames, GT anchor", list(range(N)), ("GT", 0)), ("ALL frames, G4d predicted anchor", list(range(N)), ("G4D", 0)), ("G15 subset, G4d predicted anchor", sub, ("G4D", 0))):
        M = {m: [mt(F[m][i], m, key) for i in idx] for m in MODELS}; print(f"\n== 3D / Z / quality: {nm} (n={len(idx)}) ==\n   {'metric':22s}{hdr}")
        for k in ("xy@10", "xy@20", "xy@30", "xy@end", "z@10", "z@20", "z@30", "z@end", "z_min", "d3@10", "d3@20", "d3@30", "d3@end", "d3_min", "cos3d", "dz@10", "dz@30"): row(k, [np.mean([x[k] for x in M[m]]) for m in MODELS])
        row("U-shape d3_end-d3_min", [np.mean([x["d3@end"] - x["d3_min"] for x in M[m]]) for m in MODELS]); row("frac d3@30 < d3@10", [np.mean([x["d3@30"] < x["d3@10"] for x in M[m]]) for m in MODELS])
        print("   -- quality guard --")
        row("first_off cm", [np.mean([x["first_off"] for x in M[m]]) for m in MODELS]); row("frac first_off>5cm", [np.mean([x["first_off"] > 5 for x in M[m]]) for m in MODELS]); row("jerk", [np.mean([x["jerk"] for x in M[m]]) for m in MODELS], "{:8.3f}"); row("max_jerk", [np.mean([x["max_jerk"] for x in M[m]]) for m in MODELS], "{:8.3f}"); row("max_step", [np.mean([x["max_step"] for x in M[m]]) for m in MODELS], "{:8.3f}"); row("path_len cm", [np.mean([x["path_len"] for x in M[m]]) for m in MODELS])
        print("   -- gripper (not trained) --"); row("first close idx", [np.mean([x["close_idx"] for x in M[m]]) for m in MODELS]); row("closed fraction", [np.mean([x["closed_frac"] for x in M[m]]) for m in MODELS], "{:8.3f}"); row("reopen prob", [np.mean([x["reopen"] for x in M[m]]) for m in MODELS], "{:8.3f}")
        dv = np.mean([np.linalg.norm(F["g16"][i]["out"][key]["xyz"] - F["v2"][i]["out"][key]["xyz"], axis=1).mean() for i in idx]) * 100; dg = np.mean([np.linalg.norm(F["g11"][i]["out"][key]["xyz"] - F["v2"][i]["out"][key]["xyz"], axis=1).mean() for i in idx]) * 100; print(f"   mean |chunk - V2 chunk| cm: g11 {dg:.2f} | g16 {dv:.2f}")
    # ---- G15 reference ----
    try:
        G15 = pickle.load(open("outputs/eval/safediff_vla_g15/g15.pkl", "rb"))["R"]; print("\n== reference G15 (same subset, GT anchor, recomputed feature) mean d3@30 / d3_min / d3@end: " + " | ".join(f"{c}: {np.mean([metrics(f, ('GT', 'R', c))['d3@30'] for f in G15]):.1f}/{np.mean([metrics(f, ('GT', 'R', c))['d3_min'] for f in G15]):.1f}/{np.mean([metrics(f, ('GT', 'R', c))['d3@end'] for f in G15]):.1f}" for c in 'ABCD'))
    except Exception as ex: print("G15 ref unavailable", ex)


if __name__ == "__main__":
    main()
