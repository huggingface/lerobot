#!/usr/bin/env python
"""G15 visuals (GT anchor, recomputed feature): A/B/C/D chunks of the same observation overlaid. Columns: top-down XY (EE = square, GT = star), side view (chunk index vs z; gray = EE z, green = GT z), 3D distance to GT vs chunk index."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from g15_analysis import OUT, metrics  # noqa: E402
from g9_analysis import stratum  # noqa: E402

COL = {"A": "k", "B": "tab:blue", "C": "tab:orange", "D": "tab:red"}; LAB = {"A": "A g1+orig", "B": "B g4+orig", "C": "C g1+open", "D": "D g4+open"}


def main():
    F = pickle.load(open(OUT / "g15.pkl", "rb"))["R"]; key = ("GT", "R")
    for f in F: f["stratum"] = stratum({"ee_gt_xy": f["ee_gt_xy"], "t": f["t"], "any_card_displaced_cm": 0.0 if f["n_displaced"] < 1 else 99.0}); f["m"] = {c: metrics(f, (*key, c)) for c in "ABCD"}
    big = lambda f: abs(f["gt"][2] - f["ee"][2]) * 100 + f["ee_gt_xy"] * 100
    pick = lambda cond, score: max((f for f in F if cond(f)), key=score)
    cases = [("1 large XY + large Z error", pick(lambda f: True, lambda f: min(f["ee_gt_xy"] * 100, abs(f["ee_gt_z"]) * 100))), ("2 MID", pick(lambda f: f["stratum"] == "MID", lambda f: f["m"]["C"]["d3_min"] - f["m"]["D"]["d3_min"])), ("3 LATE", pick(lambda f: f["stratum"] == "LATE", big)),
             ("4 largest U-shape under C (z@end - z_min)", pick(lambda f: True, lambda f: f["m"]["C"]["z@end"] - f["m"]["C"]["z_min"])), ("5 D fails (largest D min-3D)", pick(lambda f: True, lambda f: f["m"]["D"]["d3_min"]))]
    fig, ax = plt.subplots(len(cases), 3, figsize=(15, 4 * len(cases)))
    for r, (title, f) in enumerate(cases):
        for c in "ABCD":
            x = f["out"][(*key, c)]["xyz"]; ax[r, 0].plot(x[:, 0], x[:, 1], color=COL[c], lw=1.8, label=LAB[c]); ax[r, 0].plot(*x[-1, :2], "o", color=COL[c], ms=4); ax[r, 1].plot(x[:, 2], color=COL[c], lw=1.8); ax[r, 2].plot(np.linalg.norm(x - f["gt"], axis=1) * 100, color=COL[c], lw=1.8)
        ax[r, 0].plot(*f["ee"][:2], "ks", ms=8); ax[r, 0].plot(*f["gt"][:2], "g*", ms=14); ax[r, 0].set_aspect("equal"); ax[r, 0].set_title(f"{title}\n{f['src']} ep{f['ep']} t={f['t']} {f['stratum']} | EE-GT xy {f['ee_gt_xy'] * 100:.0f}cm z {f['ee_gt_z'] * 100:+.0f}cm", fontsize=8); ax[r, 0].legend(fontsize=6)
        ax[r, 1].axhline(f["ee"][2], color="gray", ls=":"); ax[r, 1].axhline(f["gt"][2], color="green", lw=1); ax[r, 1].set_title("chunk z (m)", fontsize=8); ax[r, 2].set_title("3D distance to GT (cm); min: " + " ".join(f"{c}={f['m'][c]['d3_min']:.0f}" for c in "ABCD") + " | 1st-act jump " + " ".join(f"{f['m'][c]['first_off']:.1f}" for c in "ABCD"), fontsize=8)
        for a in ax[r, 1:]: a.set_xlabel("chunk index", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "g15_composition.png", dpi=55); print("saved")


if __name__ == "__main__":
    main()
