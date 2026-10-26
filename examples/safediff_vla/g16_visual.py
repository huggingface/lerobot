#!/usr/bin/env python
"""G16 visuals: V2 / G11 / G16 / G4 (gamma=4, upper bound) chunks on the same observation (GT anchor). Columns: top-down XY, side view z vs chunk index, 3D distance to GT vs chunk index."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from g16_analysis import OUT, MODELS, stratum  # noqa: E402

COL = {"v2": "k", "g11": "tab:green", "g16": "tab:red", "g4": "tab:blue"}


def main():
    R = pickle.load(open(OUT / "capture_v2_g11_g16_g4.pkl", "rb")); F = R["v2"]; N = len(F); d3 = lambda m, i, k: np.linalg.norm(R[m][i]["out"][k]["xyz"] - F[i]["gt"], axis=1) * 100
    pre = [i for i in range(N) if F[i]["phase"] == 0 and F[i]["grip_raw"] <= 0.5]; sub = [i for i in range(N) if F[i]["sub"]]; ez = lambda i: F[i]["gt"][2] - F[i]["ee"][2]
    cases = [("1 large XY + large Z error", max(sub, key=lambda i: min(F[i]["ee_gt_xy"], -ez(i))), (("GT", 0))), ("2 XY far (PRE_GRASP open)", max(pre, key=lambda i: F[i]["ee_gt_xy"]), ("GT", 0)),
             ("3 XY near / Z far (PRE_GRASP open)", min([i for i in pre if F[i]["ee_gt_xy"] < .05 and ez(i) < -.03], key=ez), ("GT", 0)), ("4 G15 U-shape: open wrist patch (cell C/D analogue)", max(sub, key=lambda i: d3("g4", i, ("GT", 1))[-1] - d3("g4", i, ("GT", 1)).min()), ("GT", 1)),
             ("5 G16 failure (largest d3@30 worse than V2)", max(range(N), key=lambda i: d3("g16", i, ("GT", 0))[29] - d3("v2", i, ("GT", 0))[29]), ("GT", 0))]
    fig, ax = plt.subplots(len(cases), 3, figsize=(15, 4 * len(cases)))
    for r, (title, i, k) in enumerate(cases):
        f = F[i]
        for m in MODELS:
            x = R[m][i]["out"][k]["xyz"]; ax[r, 0].plot(x[:, 0], x[:, 1], color=COL[m], lw=1.8, label=m); ax[r, 0].plot(*x[-1, :2], "o", color=COL[m], ms=4); ax[r, 1].plot(x[:, 2], color=COL[m], lw=1.8); ax[r, 2].plot(d3(m, i, k), color=COL[m], lw=1.8)
        ax[r, 0].plot(*f["ee"][:2], "ks", ms=8); ax[r, 0].plot(*f["gt"][:2], "g*", ms=14); ax[r, 0].set_aspect("equal"); ax[r, 0].legend(fontsize=6); ax[r, 0].set_title(f"{title}\n{f['src']} ep{f['ep']} t={f['t']} {stratum(f)} phase{f['phase']} grip={'closed' if f['grip_raw'] > .5 else 'open'} wrist={'open-patch' if k[1] else 'orig'} | xy {f['ee_gt_xy'] * 100:.0f}cm z {ez(i) * 100:+.0f}cm", fontsize=8)
        ax[r, 1].axhline(f["ee"][2], color="gray", ls=":"); ax[r, 1].axhline(f["gt"][2], color="green", lw=1); ax[r, 1].set_title("chunk z (m)", fontsize=8)
        ax[r, 2].set_title("3D dist to GT (cm) | first-action jump " + " ".join(f"{m}={np.linalg.norm(R[m][i]['out'][k]['xyz'][0] - f['ee']) * 100:.1f}" for m in MODELS), fontsize=8)
        for a in ax[r, 1:]: a.set_xlabel("chunk index", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "g16_cases.png", dpi=55); print("saved")


if __name__ == "__main__":
    main()
