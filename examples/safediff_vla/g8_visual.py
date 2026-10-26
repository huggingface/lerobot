#!/usr/bin/env python
"""G8 visuals on FIXED observations (robot frame, metres). Per frame: top-down (x,y) and side (y,z): EE (black dot), GT anchor (green star), nearest-wrong-card anchor (red x), G4d prediction (grey +), decoder chunk under GT anchor (green line, start->end marked by circle->square),
under wrong-card anchor (red line), under G4d prediction (grey dashed). Plus a summary figure: anchor-following gain vs chunk segment per stratum (xy and z) and the gripper chunk under GT vs wrong anchor."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from g8_analysis import OUT, SEG, ORDER, stratum, gain, close_idx  # noqa: E402


def draw(ax_top, ax_side, f, title):
    for key, col, ls, lab in (("A1", "green", "-", "chunk | GT anchor"), ("A2", "red", "-", "chunk | wrong-card anchor"), ("A0", "gray", "--", "chunk | G4d pred")):
        c = f["cond"][key]["xyz"]; ax_top.plot(c[:, 0], c[:, 1], ls, color=col, lw=1.6, label=lab); ax_top.plot(*c[0, :2], "o", color=col, ms=4); ax_top.plot(*c[-1, :2], "s", color=col, ms=5); ax_side.plot(c[:, 1], c[:, 2], ls, color=col, lw=1.6); ax_side.plot(c[0, 1], c[0, 2], "o", color=col, ms=4); ax_side.plot(c[-1, 1], c[-1, 2], "s", color=col, ms=5)
    gt, wr, pr, ee = f["cond"]["A1"]["anchor_robot"], f["cond"]["A2"]["anchor_robot"], f["cond"]["A0"]["anchor_robot"], f["ee_robot"]
    for ax, (i, j) in ((ax_top, (0, 1)), (ax_side, (1, 2))):
        ax.plot(gt[i], gt[j], "*", color="green", ms=16, mec="k"); ax.plot(wr[i], wr[j], "x", color="red", ms=11, mew=3); ax.plot(pr[i], pr[j], "+", color="gray", ms=11, mew=2); ax.plot(ee[i], ee[j], "ko", ms=7); ax.set_aspect("equal"); ax.tick_params(labelsize=6)
    gx = (gain(f, "x", "31-50") + gain(f, "y", "31-50")) / 2; ax_top.set_title(f"{title}\nt={f['t']} {f['src']} ep{f['ep']} {f['stratum']} | EE-GT xy {f['ee_gt_xy'] * 100:.0f}cm dz {f['ee_gt_z'] * 100:+.0f}cm | xy-gain(31-50) {gx:+.2f} z-gain {gain(f, 'z', '31-50'):+.2f}", fontsize=7)
    ax_side.set_xlabel("y (m) -> side view (y,z)", fontsize=6); ax_top.set_xlabel("x (m): top-down (x,y)", fontsize=6)


def main():
    F = []
    for src in ("g4d", "oracle"):
        for fr in pickle.load(open(OUT / src / "audit.pkl", "rb")): fr["src"] = src; fr["stratum"] = stratum(fr); F.append(fr)
    gxy = lambda f: (gain(f, "x", "31-50") + gain(f, "y", "31-50")) / 2; sep = lambda f: np.linalg.norm(f["cond"]["A1"]["xyz"][49, :2] - f["cond"]["A2"]["xyz"][49, :2])
    early = max((f for f in F if f["stratum"] == "EARLY"), key=gxy); mid = sorted((f for f in F if f["stratum"] == "MID"), key=gxy)[len([f for f in F if f["stratum"] == "MID"]) // 2]
    late = min((f for f in F if f["stratum"] == "LATE"), key=lambda f: abs(gxy(f))); near = min((f for f in F if f["stratum"] == "NEAR/CONTACT" and f["ee_gt_xy"] < .08 and f["any_card_displaced_cm"] > 1.5), key=lambda f: abs(gxy(f)))
    zmis = max((f for f in F if f["ee_gt_xy"] < 0.04 and f["ee_gt_z"] > 0.06), key=lambda f: f["ee_gt_z"]); picks = [(early, "1 EARLY (highest xy gain)"), (mid, "2 MID (median xy gain)"), (late, "3 LATE (lowest |gain|: anchor-insensitive)"), (near, "4 NEAR+contact (lowest |gain|)"), (zmis, "5 z-misalignment (XY aligned, EE above anchor)")]
    fig, ax = plt.subplots(2, 5, figsize=(21, 8.5))
    for c, (f, t) in enumerate(picks): draw(ax[0, c], ax[1, c], f, t)
    ax[0, 0].legend(fontsize=6, loc="best"); fig.tight_layout(); fig.savefig(OUT / "g8_chunks_fixed_obs.png", dpi=55); plt.close(fig)
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2)); xs = np.arange(len(SEG))
    for st, col in zip(ORDER, ("tab:green", "tab:blue", "tab:orange", "tab:red")):
        fs = [f for f in F if f["stratum"] == st]
        for a, axn, ttl in ((ax[0], "x", "x gain (+5cm anchor -> action)"), (ax[1], "y", "y gain"), (ax[2], "z", "z gain")): a.plot(xs, [np.mean([gain(f, axn, s) for f in fs]) for s in SEG], "o-", color=col, label=f"{st} n={len(fs)}"); a.set_title(ttl, fontsize=9); a.set_xticks(xs); a.set_xticklabels(list(SEG)); a.axhline(0, color="k", lw=.5); a.axhline(1, color="gray", lw=.5, ls=":"); a.set_ylim(-0.1, 1.05)
    ax[0].legend(fontsize=7); ax[0].set_xlabel("chunk index segment"); ax[3].hist([close_idx(f, "A1") for f in F if f["stratum"] != "EARLY"], bins=range(0, 52, 2), color="tab:purple", alpha=.7, label="GT anchor"); ax[3].hist([close_idx(f, "A2") for f in F if f["stratum"] != "EARLY"], bins=range(0, 52, 2), color="tab:red", alpha=.4, label="wrong-card anchor"); ax[3].set_title("first chunk index where the decoder commands CLOSE (non-EARLY frames)", fontsize=8); ax[3].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "g8_gain_profile.png", dpi=55); print("saved", [(t, f["src"], f["ep"], f["t"]) for f, t in picks])


if __name__ == "__main__":
    main()
