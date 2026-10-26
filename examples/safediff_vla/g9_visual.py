#!/usr/bin/env python
"""G9-A visuals on the SAME fixed observations as G8's five representative frames (+ summary). Solid = gamma 1, thick dashed = gamma 4. green = GT-anchor chunk, red = wrong-card-anchor chunk. EE black dot, GT star, wrong-card x. Squares mark the chunk END, circles the chunk START (first action).
Summary: anchor-following gain profile gamma 1 vs 4 (x, y, z) and distributions of wrong-card endpoint separation and first-action offset from the EE."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from g9_analysis import O9, SEG, gain, stratum, quality  # noqa: E402

PICK = [("1 EARLY", "oracle", 3, 20), ("2 MID", "g4d", 0, 210), ("3 LATE (G8: anchor-insensitive)", "g4d", 8, 80), ("4 NEAR+contact", "oracle", 9, 60), ("5 z-misalignment (XY aligned, EE high)", "oracle", 3, 210)]


def main():
    F = {}
    for src in ("g4d", "oracle"):
        for fr in pickle.load(open(O9 / src / "audit9.pkl", "rb")): fr["src"] = src; fr["stratum"] = stratum(fr); F[(src, fr["ep"], fr["t"])] = fr
    fig, ax = plt.subplots(2, 5, figsize=(21, 8.5))
    for c, (title, src, ep, t) in enumerate(PICK):
        f = F[(src, ep, t)]; gt, wr, ee = f["cond"][(1.0, "A1")]["anchor_robot"], f["cond"][(1.0, "A2")]["anchor_robot"], f["ee_robot"]
        for g, ls, lw in ((1.0, "-", 1.4), (4.0, "--", 2.4)):
            for k, col in (("A1", "green"), ("A2", "red")):
                ch = f["cond"][(g, k)]["xyz"]
                for a, (i, j) in zip((ax[0, c], ax[1, c]), ((0, 1), (1, 2))): a.plot(ch[:, i], ch[:, j], ls, color=col, lw=lw); a.plot(ch[0, i], ch[0, j], "o", color=col, ms=4); a.plot(ch[-1, i], ch[-1, j], "s", color=col, ms=5)
        for a, (i, j) in zip((ax[0, c], ax[1, c]), ((0, 1), (1, 2))): a.plot(gt[i], gt[j], "*", color="green", ms=16, mec="k"); a.plot(wr[i], wr[j], "x", color="red", ms=11, mew=3); a.plot(ee[i], ee[j], "ko", ms=7); a.set_aspect("equal"); a.tick_params(labelsize=6)
        ax[0, c].set_title(f"{title}\n{src} ep{ep} t={t} | xy gain(31-50) {np.mean([gain(f, 1.0, 'x', '31-50'), gain(f, 1.0, 'y', '31-50')]):+.2f}->{np.mean([gain(f, 4.0, 'x', '31-50'), gain(f, 4.0, 'y', '31-50')]):+.2f} z {gain(f, 1.0, 'z', '31-50'):+.2f}->{gain(f, 4.0, 'z', '31-50'):+.2f}", fontsize=7)
    ax[0, 0].plot([], [], "-", color="gray", label="gamma=1 (solid)"); ax[0, 0].plot([], [], "--", color="gray", lw=2.4, label="gamma=4 (dashed)"); ax[0, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(O9 / "g9_chunks_gamma.png", dpi=55); plt.close(fig)
    fl = list(F.values()); fig, ax = plt.subplots(1, 5, figsize=(22, 4.2)); xs = np.arange(len(SEG))
    for a, axn in zip(ax[:3], ("x", "y", "z")):
        for g, ls in ((1.0, "-"), (4.0, "--")): a.plot(xs, [np.mean([gain(f, g, axn, s) for f in fl]) for s in SEG], "o" + ls, label=f"gamma={g:g}")
        a.set_title(f"{axn} gain (all 797 frames)", fontsize=9); a.set_xticks(xs); a.set_xticklabels(list(SEG)); a.axhline(0, color="k", lw=.5); a.axhline(1, color="gray", lw=.5, ls=":"); a.set_ylim(-0.2, 1.05); a.legend(fontsize=7)
    for g, col in ((1.0, "tab:blue"), (4.0, "tab:orange")): ax[3].hist([np.linalg.norm(f["cond"][(g, "A1")]["xyz"][49, :2] - f["cond"][(g, "A2")]["xyz"][49, :2]) * 100 for f in fl], bins=30, range=(0, 30), alpha=.55, color=col, label=f"gamma={g:g}")
    ax[3].set_title("wrong-card endpoint separation (cm)", fontsize=9); ax[3].legend(fontsize=7)
    for g, col in ((1.0, "tab:blue"), (4.0, "tab:orange")): ax[4].hist([quality(f["cond"][(g, "A0")]["xyz"], f["cond"][(g, "A0")]["grip"], f["ee_robot"])["first_offset"] for f in fl], bins=30, range=(0, 20), alpha=.55, color=col, label=f"gamma={g:g}")
    ax[4].set_title("first-action offset from EE (cm), G4d anchor", fontsize=9); ax[4].legend(fontsize=7); fig.tight_layout(); fig.savefig(O9 / "g9_summary.png", dpi=55); print("saved")


if __name__ == "__main__":
    main()
