#!/usr/bin/env python
"""G20 analysis (no learning). Every existing G4d 2D signal is projected to the robot frame through the FIXED camera calibration and the single global grasp-plane height (poker grasp keypoint z = 0.0710 m in the robot frame = the constant value at episode start in every scene, NOT per-scene GT) and compared with the GT grasp keypoint (diagnostic only).
P1 fine alpha-weighted centroid (front px) | P2 final-map argmax cell centre (front) | P2r final-map argmax cell centre (right camera) | P3 centroid of the final-map mass inside the selected crop (front) | P4 selected crop centre (front). 'anchor' = the current learned xyz anchor head.
Sets: FIRST REPLAN of seeds 3000-3009 (bank t=0 frames of the raw-G4d episodes == the closed-loop first replans; fine_px/selected/anchor verified identical to the saved cl_geo/g4d records) | PRE_GRASP-open bank frames | all 797 bank frames."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g4b_geometry import CAL, ray, proj  # noqa: E402
from g17_capture import load_bank  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g20"); ZP = 0.0710; BASE = np.array([0.0, -0.4, 0.78]); NAMES = ["anchor", "P1 fine centroid", "P2 final argmax", "P2r right argmax", "P3 local map centroid", "P4 crop centre"]


def to_robot(cam, u, v):
    o, d = ray(CAL[cam], u, v); t = (ZP + BASE[2] - o[2]) / d[2]; return (o + t * d) - BASE


def signals(r):
    I = r["info"]; w2 = np.array(I["w2"]); sel = I["selected"]; bx, by, sd = I["boxes"][sel]; cc = lambda c: ((c % 8 + .5) * 60, (c // 8 + .5) * 60); P = {}
    P["P1 fine centroid"] = tuple(I["fine_px_front"]); P["P2 final argmax"] = cc(I["final_front_argmax"]); P["P4 crop centre"] = (bx + sd / 2, by + sd / 2)
    cells = [(c, cc(c)) for c in range(64) if bx <= cc(c)[0] <= bx + sd and by <= cc(c)[1] <= by + sd]; ws = np.array([w2[c] for c, _ in cells]); P["P3 local map centroid"] = tuple((np.array([p for _, p in cells]) * ws[:, None]).sum(0) / ws.sum()) if ws.sum() > 0 else P["P2 final argmax"]
    xyz = {k: to_robot(0, *v) for k, v in P.items()}; xyz["P2r right argmax"] = to_robot(1, *cc(I["final_right_argmax"])); xyz["anchor"] = r["anchor"]; P["P2r right argmax"] = None; P["anchor"] = tuple(proj(CAL[0], r["anchor"] + BASE)); return P, xyz


def main():
    D = pickle.load(open(OUT / "g20_signals.pkl", "rb")); R = D["R"]; B, _, _ = load_bank(); M = {}
    for r, e in zip(R, B):
        P, X = signals(r); cs = e["cards_xyz_robot"]; tn = e["target_name"]; gt = np.asarray(e["gt_anchor_robot"]); near = {k: min(cs, key=lambda n: np.linalg.norm(np.asarray(cs[n])[:2] - v[:2])) == tn for k, v in X.items()}
        M[r["i"]] = {"P": P, "X": X, "gt": gt, "ctr": np.asarray(cs[tn]), "near_ok": near, "src": r["src"], "ep": r["ep"], "t": r["t"], "phase": e["phase"], "grip": e["grip_raw"], "sel_ok": None}
    for i, e in enumerate(B): M[i]["flat"] = abs(e["gt_anchor_robot"][2] - ZP) < 0.005
    print(f"frames whose GT keypoint is on the grasp plane (|z-{ZP}|<5mm; others = card lifted/pushed, the fixed plane is wrong for them and they are EXCLUDED from the bank-wide sets): {sum(m['flat'] for m in M.values())}/{len(M)}")
    pre = [i for i, m in M.items() if m["phase"] == 0 and m["grip"] <= 0.5 and m["flat"]]; first = [next(i for i, m in M.items() if m["src"] == "g4d" and m["ep"] == k and m["t"] == 0) for k in range(10)]
    off = {k: np.median([M[i]["X"][k][:2] - M[i]["gt"][:2] for i in M], axis=0) for k in NAMES}; print("systematic offset of each signal vs the GT grasp keypoint (median over 797 frames, cm, x/y): " + " | ".join(f"{k}: {off[k][0] * 100:+.1f}/{off[k][1] * 100:+.1f}" for k in NAMES))
    def stat(S, k, ref="gt", corr=False):
        e = np.array([np.linalg.norm(M[i]["X"][k][:2] - (M[i]["gt"][:2] if ref == "gt" else M[i]["ctr"][:2]) - (off[k] if corr else 0)) for i in S]) * 100; return e
    for nm, S in (("FIRST REPLAN seeds 3000-3009 (n=10)", first), ("PRE_GRASP-open, plane-valid bank frames", pre), ("ALL plane-valid bank frames", [i for i in M if M[i]["flat"]])):
        print(f"\n== {nm} n={len(S)}: XY error of the projected signal vs GT grasp keypoint (cm) ==\n   {'signal':24s}{'mean':>7s}{'median':>8s}{'<1cm':>7s}{'<2cm':>7s}{'<3cm':>7s}{'<5cm':>7s}{'nearest card = target':>24s}{'| after removing median offset: median, <2cm, <3cm':>52s}")
        for k in NAMES:
            e = stat(S, k); ec = stat(S, k, corr=True) if k != "anchor" else e
            print(f"   {k:24s}{e.mean():7.1f}{np.median(e):8.1f}{np.mean(e < 1):7.2f}{np.mean(e < 2):7.2f}{np.mean(e < 3):7.2f}{np.mean(e < 5):7.2f}{np.mean([M[i]['near_ok'][k] for i in S]):24.2f}{np.median(ec):14.1f}{np.mean(ec < 2):7.2f}{np.mean(ec < 3):7.2f}")
        e = stat(S, "P1 fine centroid", "ctr"); print(f"   (P1 vs the target CARD CENTRE: median {np.median(e):.1f} cm, <2cm {np.mean(e < 2):.2f}, <3cm {np.mean(e < 3):.2f}; GT keypoint vs card centre median {np.median([np.linalg.norm(M[i]['gt'][:2] - M[i]['ctr'][:2]) for i in S]) * 100:.1f} cm)")
    print("\n== 2D (image-space) error on the FIRST REPLANS: pixel distance to the projected GT keypoint (px; 1 cm on the table ~ 4.2 px along x, 2.4 px along y) ==")
    for nm in ("P1 fine centroid", "P2 final argmax", "P3 local map centroid", "P4 crop centre", "anchor"): d = [np.hypot(*(np.array(M[i]["P"][nm]) - np.array(proj(CAL[0], M[i]["gt"] + BASE)))) for i in first]; print(f"   {nm:24s} median {np.median(d):5.1f} px | mean {np.mean(d):5.1f} | per seed " + " ".join(f"{x:5.1f}" for x in d))
    print("\n== FIRST REPLAN per seed: XY error (cm) to the GT grasp keypoint; * = nearest card to the signal is NOT the target ==\n   seed  selected-crop-card? | " + " ".join(f"{k.split()[0]:>9s}" for k in NAMES))
    for k, i in enumerate(first):
        m = M[i]; print(f"   {3000 + k}  " + " " * 18 + "| " + " ".join(f"{np.linalg.norm(m['X'][n][:2] - m['gt'][:2]) * 100:8.1f}{'*' if not m['near_ok'][n] else ' '}" for n in NAMES))
    print("\n== 2D-vs-3D decomposition (first replans & PRE-open): is the learned xyz head worse than its own 2D signal? ==")
    for nm, S in (("first replan", first), ("PRE-open", pre), ("ALL flat", [i for i in M if M[i]["flat"]])):
        a = stat(S, "anchor"); p = stat(S, "P1 fine centroid"); print(f"   {nm:13s} anchor median {np.median(a):5.1f} vs P1 median {np.median(p):5.1f} | anchor card-correct {np.mean([M[i]['near_ok']['anchor'] for i in S]):.2f} vs P1 {np.mean([M[i]['near_ok']['P1 fine centroid'] for i in S]):.2f} | frames P1 better than anchor by >2cm {np.mean(a - p > 2):.2f}, worse by >2cm {np.mean(p - a > 2):.2f}")
    # overlays
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    IM, MK = {}, {}
    for src in ("g4d",):
        im = np.load(Path("outputs/eval/safediff_vla_g13_bank") / src / "imgs.npy", mmap_mode="r"); mk = np.load(Path("outputs/eval/safediff_vla_g13_bank") / src / "masks_packed.npy")
        idx = {(m["ep"], m["t"]): i for i, m in M.items() if m["src"] == src}
        for k in range(10): j = idx[(k, 0)]; jj = j - sum(1 for m in M.values() if False); IM[k] = (j, im, mk)
    fig, ax = plt.subplots(2, 5, figsize=(26, 11)); gi = {}
    n_g4 = 0
    for k in range(10):
        i = first[k]; m = M[i]; ib = i  # bank index of g4d episodes is the same ordering as R (g4d first)
        im = np.load(Path("outputs/eval/safediff_vla_g13_bank/g4d/imgs.npy"), mmap_mode="r")[sum(1 for x in R[:i] if x["src"] == "g4d")][0] if True else None
        mk = np.unpackbits(np.load(Path("outputs/eval/safediff_vla_g13_bank/g4d/masks_packed.npy"))[sum(1 for x in R[:i] if x["src"] == "g4d")])[:3 * 2 * 480 * 480].reshape(3, 2, 480, 480)[0, 1].astype(bool)
        a = ax[k // 5, k % 5]; a.imshow(im); a.contour(mk, levels=[0.5], colors="w", linewidths=0.6)
        for nm, c, mkr in (("P1 fine centroid", "red", "o"), ("P2 final argmax", "orange", "s"), ("P3 local map centroid", "yellow", "^"), ("P4 crop centre", "cyan", "D"), ("anchor", "magenta", "x")): a.plot(*m["P"][nm], mkr, color=c, ms=9, mfc="none" if mkr != "x" else None, mew=2, label=nm.split()[0] if k == 0 else None)
        a.plot(*proj(CAL[0], m["gt"] + BASE), "*", color="lime", ms=16, label="GT keypoint" if k == 0 else None); a.plot(*proj(CAL[0], m["ctr"] + BASE), "+", color="lime", ms=12, mew=2, label="target card centre" if k == 0 else None)
        I = R[i]["info"]; bx, by, sd = I["boxes"][I["selected"]]; a.add_patch(plt.Rectangle((bx, by), sd, sd, fill=False, ec="cyan", lw=1))
        a.set_title(f"seed {3000 + k}: anchor err {np.linalg.norm(m['X']['anchor'][:2] - m['gt'][:2]) * 100:.1f}cm | P1 {np.linalg.norm(m['X']['P1 fine centroid'][:2] - m['gt'][:2]) * 100:.1f}cm | P3 {np.linalg.norm(m['X']['P3 local map centroid'][:2] - m['gt'][:2]) * 100:.1f}cm", fontsize=9); a.axis("off")
    ax[0, 0].legend(fontsize=7, loc="lower left"); fig.tight_layout(); fig.savefig(OUT / "g20_first_replan_overlays.png", dpi=48); print("saved overlays")


if __name__ == "__main__":
    main()
