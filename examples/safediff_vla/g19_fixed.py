#!/usr/bin/env python
"""G19 STEP 1 (fixed observations, no learning): V2 decoder chunk vs deterministic geometric chunk, on the 797 G8-G18 bank frames. Anchors: GT (diagnostic upper bound), G4d prediction (the real pipeline), wrong card (counterfactual). All errors are to the GT grasp keypoint (the real target). Frames: PRE_GRASP-open (phase 0 & gripper open) primary; phase-0 closed-and-far (G15 subset-like) and all frames secondary."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g19_geometry import geo_chunk  # noqa: E402
from g16_analysis import stratum  # noqa: E402

OUT17 = Path("outputs/eval/safediff_vla_g17"); OUT = Path("outputs/eval/safediff_vla_g19")
ARMS = [("V2/GT", "v2", "GT"), ("GEO/GT", "geo", "GT"), ("V2/G4D", "v2", "G4D"), ("GEO/G4D", "geo", "G4D"), ("V2/WR", "v2", "WR"), ("GEO/WR", "geo", "WR")]


def chunk(f, kind, an, G4):
    if kind == "v2": return f["cond"][("V2", an)]["xyz"].astype(float)
    return geo_chunk(f["ee"], {"GT": f["gt"], "WR": f["wr"], "G4D": G4[f["i"]]}[an])


def qual(x, ee):
    st = np.linalg.norm(np.diff(x, axis=0), axis=1); jk = np.linalg.norm(np.diff(x, n=3, axis=0), axis=1); return {"first_off": np.linalg.norm(x[0] - ee) * 100, "max_step": st.max() * 100, "jerk": jk.mean() * 100, "path": st.sum() * 100}


def cosv(x, ee, tgt, k, dims):
    v = x[:k, dims].mean(0) - ee[dims]; d = tgt[dims] - ee[dims]; return float(v @ d / (np.linalg.norm(v) * np.linalg.norm(d) + 1e-9))


def main():
    F = pickle.load(open(OUT17 / "g17_bank.pkl", "rb"))["R"]; G4 = np.load(OUT / "g4d_anchors.npy")
    for f in F: f["stratum"] = stratum(f)
    open_pre = [f for f in F if f["phase"] == 0 and f["grip_raw"] <= 0.5]; closed_far = [f for f in F if f["phase"] == 0 and f["grip_raw"] > 0.5 and np.linalg.norm(f["gt"] - f["ee"]) > 0.06]
    e4 = np.array([np.linalg.norm(G4[f["i"]] - f["gt"]) for f in F]) * 100; e4xy = np.array([np.linalg.norm(G4[f["i"]][:2] - f["gt"][:2]) for f in F]) * 100
    print(f"G4d anchor error vs GT keypoint (all {len(F)} frames): XY mean {e4xy.mean():.1f} median {np.median(e4xy):.1f} cm | 3D mean {e4.mean():.1f} | frac XY<3cm {np.mean(e4xy < 3):.2f}, <5cm {np.mean(e4xy < 5):.2f}, >10cm {np.mean(e4xy > 10):.2f} (raw-G4d bank frames: {np.mean(e4xy[[f['src'] == 'g4d' for f in F]] > 10):.2f} >10cm; oracle-episode frames: {np.mean(e4xy[[f['src'] == 'oracle' for f in F]] > 10):.2f})")
    for scope, S in (("PRE_GRASP-open (primary)", open_pre), ("phase-0 closed-and-far", closed_far), ("ALL frames", F)):
        print(f"\n== {scope} n={len(S)}: arms = decoder / geometry x anchor (GT | G4d | wrong card). errors cm to the GT keypoint ==\n   {'metric':16s}" + "".join(f"{a[0]:>10s}" for a in ARMS))
        R = {a[0]: [] for a in ARMS}
        for f in S:
            for nm, kind, an in ARMS:
                x = chunk(f, kind, an, G4); gt = f["gt"]; ee = f["ee"]; er = x - gt; q = qual(x, ee); tgt = {"GT": gt, "WR": f["wr"], "G4D": G4[f["i"]]}[an]
                R[nm].append({"cos5_3d": cosv(x, ee, tgt, 5, [0, 1, 2]), "cos10_3d": cosv(x, ee, tgt, 10, [0, 1, 2]), "cos10_xy": cosv(x, ee, tgt, 10, [0, 1]), "xy@30": np.linalg.norm(er[29, :2]) * 100, "xy@end": np.linalg.norm(er[49, :2]) * 100, "z@30": abs(er[29, 2]) * 100, "z@end": abs(er[49, 2]) * 100, "d3@30": np.linalg.norm(er[29]) * 100, "d3@end": np.linalg.norm(er[49]) * 100, "d3_min": np.linalg.norm(er, axis=1).min() * 100,
                                  "dz@29": (x[29, 2] - ee[2]) * 100, "desc_B": float(x[49, 2] - x[29, 2]) * 100, "U(end-min)": (np.linalg.norm(er[49]) - np.linalg.norm(er, axis=1).min()) * 100, "to_own_tgt_end": np.linalg.norm(x[49] - tgt) * 100, **q})
        for m in ("cos5_3d", "cos10_3d", "cos10_xy", "xy@30", "xy@end", "z@30", "z@end", "d3@30", "d3@end", "d3_min", "U(end-min)", "dz@29", "desc_B", "to_own_tgt_end", "first_off", "max_step", "jerk", "path"): print(f"   {m:16s}" + "".join(f"{np.mean([r[m] for r in R[a[0]]]):10.2f}" for a in ARMS))
        print("   frac first_off>5cm" + "".join(f"{np.mean([r['first_off'] > 5 for r in R[a[0]]]):10.2f}" for a in ARMS)); print("   frac 3D@end<3cm   " + "".join(f"{np.mean([r['d3@end'] < 3 for r in R[a[0]]]):10.2f}" for a in ARMS))
        # counterfactual GT vs wrong-card
        print("   -- counterfactual GT vs WRONG-card anchor (anchor separation %.1f cm) --" % (np.mean([np.linalg.norm(f["gt"][:2] - f["wr"][:2]) for f in S]) * 100))
        for kind in ("v2", "geo"):
            s10 = np.mean([np.linalg.norm((chunk(f, kind, "GT", G4)[:10, :2] - chunk(f, kind, "WR", G4)[:10, :2]).mean(0)) for f in S]) * 100; se = np.mean([np.linalg.norm(chunk(f, kind, "GT", G4)[49, :2] - chunk(f, kind, "WR", G4)[49, :2]) for f in S]) * 100
            gain = np.mean([float((chunk(f, kind, "WR", G4)[49, :2] - chunk(f, kind, "GT", G4)[49, :2]) @ (f["wr"][:2] - f["gt"][:2]) / ((f["wr"][:2] - f["gt"][:2]) @ (f["wr"][:2] - f["gt"][:2]) + 1e-12)) for f in S]); gain10 = np.mean([float((chunk(f, kind, "WR", G4)[:10, :2].mean(0) - chunk(f, kind, "GT", G4)[:10, :2].mean(0)) @ (f["wr"][:2] - f["gt"][:2]) / ((f["wr"][:2] - f["gt"][:2]) @ (f["wr"][:2] - f["gt"][:2]) + 1e-12)) for f in S])
            print(f"   {kind:4s}: first10 sep {s10:6.2f} cm | endpoint sep {se:6.2f} cm | signed along-axis gain first10 {gain10:+.2f} endpoint {gain:+.2f} (1 = chunk follows the anchor one-for-one)")
        if scope.startswith("PRE"):
            print("   by stratum (3D@end GT-anchor V2 | GEO, first_off V2 | GEO):")
            for s in ("MID", "LATE", "NEAR/CONTACT"):
                ix = [j for j, f in enumerate(S) if f["stratum"] == s]
                if ix: print(f"     {s:13s} n={len(ix):3d} | {np.mean([R['V2/GT'][j]['d3@end'] for j in ix]):6.1f} | {np.mean([R['GEO/GT'][j]['d3@end'] for j in ix]):6.1f} || {np.mean([R['V2/GT'][j]['first_off'] for j in ix]):5.2f} | {np.mean([R['GEO/GT'][j]['first_off'] for j in ix]):5.2f}")
    pickle.dump({"ok": True}, open(OUT / "g19_fixed_done.pkl", "wb"))


if __name__ == "__main__":
    main()
