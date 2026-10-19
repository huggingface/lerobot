#!/usr/bin/env python
"""G15 analysis: 2x2 (gamma x open-wrist) on fixed closed-and-far observations. Errors are against the GT grasp keypoint (cm). Effects: gamma main = B-A, D-C; wrist main = C-A, D-B; interaction = D-B-C+A (negative = synergy for error metrics). Paired over frames; 95% CI by bootstrap over frames."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g9_analysis import quality, stratum  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g15"); THR = 0.5; K = {"10": 9, "20": 19, "30": 29, "end": 49}; CELLS = "ABCD"


def metrics(f, c):
    x = f["out"][c]["xyz"]; g = f["out"][c]["grip"]; gt, ee = f["gt"], f["ee"]; err = x - gt; xy = np.linalg.norm(err[:, :2], axis=1) * 100; z = np.abs(err[:, 2]) * 100; d3 = np.linalg.norm(err, axis=1) * 100; q = quality(x, g, ee); cl = g <= THR; v = x[29] - ee; u = gt - ee; m = {}
    for k, i in K.items(): m[f"xy@{k}"] = xy[i]; m[f"z@{k}"] = z[i]; m[f"d3@{k}"] = d3[i]
    for k, i in (("10", 9), ("20", 19), ("30", 29)): m[f"dz@{k}"] = (x[i, 2] - ee[2]) * 100
    m.update({"xy_min": xy.min(), "z_min": z.min(), "d3_min": d3.min(), "cos3d": float(v @ u / (np.linalg.norm(v) * np.linalg.norm(u) + 1e-9)), "first_off": q["first_offset"], "jerk": q["jerk"], "max_jerk": q["max_jerk"], "max_step": q["max_step"], "close_idx": q["close_idx"], "closed_frac": cl.mean(), "reopen": float((cl[:-1] & ~cl[1:]).any()), "d3_start": float(np.linalg.norm(u)) * 100})
    return m


def boot(d, n=2000, seed=0):
    r = np.random.default_rng(seed); idx = r.integers(0, len(d), (n, len(d))); b = d[idx].mean(1); return np.percentile(b, 2.5), np.percentile(b, 97.5)


def report(F, key, title):
    M = {c: [metrics(f, (*key, c)) for f in F] for c in CELLS}; col = lambda c, m: np.array([x[m] for x in M[c]])
    print(f"\n######## {title}  (n={len(F)} frames) ########")
    print(f"   {'metric':12s}" + "".join(f"{c:>9s}" for c in CELLS) + f"{'g-main':>10s}{'wrist-main':>12s}{'interact':>10s}   (g-main=mean(B-A,D-C), wrist-main=mean(C-A,D-B))")
    rows = [f"xy@{k}" for k in K] + [f"z@{k}" for k in K] + ["xy_min", "z_min"] + [f"d3@{k}" for k in K] + ["d3_min", "dz@10", "dz@20", "dz@30", "cos3d", "first_off", "jerk", "max_jerk", "max_step", "close_idx", "closed_frac", "reopen"]
    for m in rows:
        a, b, c, d = (col(x, m) for x in CELLS); gm = ((b - a) + (d - c)) / 2; wm = ((c - a) + (d - b)) / 2; it = d - b - c + a; lo, hi = boot(it)
        print(f"   {m:12s}" + "".join(f"{col(x, m).mean():9.2f}" for x in CELLS) + f"{gm.mean():+10.2f}{wm.mean():+12.2f}{it.mean():+10.2f}  [{lo:+.2f},{hi:+.2f}]")
    d3 = {c: col(c, "d3_min") for c in CELLS}; best_other = np.minimum.reduce([d3["A"], d3["B"], d3["C"]])
    print(f"   D min-3D < min(A,B,C) by >=2cm: {np.mean(d3['D'] < best_other - 2):.2f} | D worst (>= min(A,B,C)+2cm): {np.mean(d3['D'] > best_other + 2):.2f} | frac frames min3D<6cm: " + " ".join(f"{c}={np.mean(d3[c] < 6):.2f}" for c in CELLS) + " | <3cm: " + " ".join(f"{c}={np.mean(d3[c] < 3):.2f}" for c in CELLS))
    print(f"   quality guard: first-action offset >5cm: " + " ".join(f"{c}={np.mean(col(c, 'first_off') > 5):.2f}" for c in CELLS) + " | >6cm: " + " ".join(f"{c}={np.mean(col(c, 'first_off') > 6):.2f}" for c in CELLS) + f" | D vs A jump increase {np.mean(col('D', 'first_off') - col('A', 'first_off')):+.2f}cm")
    print("   by stratum (d3_min A/B/C/D | xy@30 | z@30 | first_off):")
    for s in ("EARLY", "MID", "LATE", "NEAR/CONTACT"):
        ix = [i for i, f in enumerate(F) if f["stratum"] == s]
        if ix: print(f"     {s:13s} n={len(ix):3d} | " + "/".join(f"{col(c, 'd3_min')[ix].mean():.1f}" for c in CELLS) + " | " + "/".join(f"{col(c, 'xy@30')[ix].mean():.1f}" for c in CELLS) + " | " + "/".join(f"{col(c, 'z@30')[ix].mean():.1f}" for c in CELLS) + " | " + "/".join(f"{col(c, 'first_off')[ix].mean():.1f}" for c in CELLS))
    return M


def main():
    D = pickle.load(open(OUT / "g15.pkl", "rb")); F = D["R"]
    for f in F: f["stratum"] = stratum({"ee_gt_xy": f["ee_gt_xy"], "t": f["t"], "any_card_displaced_cm": 0.0 if f["n_displaced"] < 1 else 99.0})
    print(f"closed-and-far bank frames {D['n_cf']} | with open donor {len(F)} | EE-GT 3D start mean {np.mean([np.linalg.norm(f['gt'] - f['ee']) for f in F]) * 100:.1f}cm | strata {{{', '.join(f'{s}: {sum(1 for f in F if f['stratum'] == s)}' for s in ('EARLY', 'MID', 'LATE', 'NEAR/CONTACT'))}}}")
    print("   (stratum 'any card displaced' proxied by n_displaced>=1; EARLY=t<=20)")
    M = {}
    for key, title in ((("GT", "R"), "PRIMARY: GT anchor, recomputed grounded feature (deployed pipeline)"), (("GT", "F"), "GT anchor, feature fixed to ORIGINAL image's (decoder visual-memory path only)"), (("G4D", "R"), "SECONDARY: G4d predicted anchor (held fixed across cells), recomputed feature")):
        M[key] = report(F, key, title)
    pickle.dump({"M": M}, open(OUT / "g15_metrics.pkl", "wb"))


if __name__ == "__main__":
    main()
