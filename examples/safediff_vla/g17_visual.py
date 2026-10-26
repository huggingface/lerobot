#!/usr/bin/env python
"""G17 visuals (GT anchor): V2 baseline | extra_I1 (best early-ish injection under the rule: lowest XY@30 error among conditions with <=20% first-offset>5cm and input-side/early location) | geo_I3 (best late safe injection, same rule) | move_I4 (late, unsafe) | G4 (reference)."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from g17_analysis import OUT  # noqa: E402
from g16_analysis import stratum  # noqa: E402

CS = {"V2": "k", "extra_I1": "tab:green", "geo_I3": "tab:red", "move_I4": "tab:purple", "G4": "tab:blue"}


def main():
    F = pickle.load(open(OUT / "g17_bank.pkl", "rb"))["R"]; N = len(F); x = lambda f, c: f["cond"][(c, "GT")]["xyz"]; d3 = lambda f, c: np.linalg.norm(x(f, c) - f["gt"], axis=1) * 100; ez = lambda f: f["gt"][2] - f["ee"][2]
    pre = [f for f in F if f["phase"] == 0 and f["grip_raw"] <= .5]; far = [f for f in F if f["grip_raw"] > .5 and np.linalg.norm(f["gt"] - f["ee"]) > .06]
    cases = [("1 large XY + large Z error", max(F, key=lambda f: min(f["ee_gt_xy"], -ez(f)))), ("2 MID", sorted([f for f in pre if stratum(f) == "MID"], key=lambda f: d3(f, "V2")[29] - d3(f, "extra_I1")[29])[len([f for f in pre if stratum(f) == "MID"]) // 2]),
             ("3 near target (xy<5cm, Z far)", min([f for f in pre if f["ee_gt_xy"] < .05 and ez(f) < -.03], key=ez)), ("4 G15 composition case (closed & far, largest G4 U-shape)", max(far, key=lambda f: d3(f, "G4")[-1] - d3(f, "G4").min())),
             ("5 injection fails (largest 3D@30 worse than V2, extra_I1)", max(F, key=lambda f: d3(f, "extra_I1")[29] - d3(f, "V2")[29]))]
    fig, ax = plt.subplots(len(cases), 3, figsize=(15, 4 * len(cases)))
    for r, (title, f) in enumerate(cases):
        for c, col in CS.items(): xx = x(f, c); ax[r, 0].plot(xx[:, 0], xx[:, 1], color=col, lw=1.8, label=c); ax[r, 0].plot(*xx[-1, :2], "o", color=col, ms=4); ax[r, 1].plot(xx[:, 2], color=col, lw=1.8); ax[r, 2].plot(d3(f, c), color=col, lw=1.8)
        ax[r, 0].plot(*f["ee"][:2], "ks", ms=8); ax[r, 0].plot(*f["gt"][:2], "g*", ms=14); ax[r, 0].plot(*f["wr"][:2], "mx", ms=9); ax[r, 0].set_aspect("equal"); ax[r, 0].legend(fontsize=6); ax[r, 0].set_title(f"{title}\n{f['src']} ep{f['ep']} t={f['t']} {stratum(f)} grip={'closed' if f['grip_raw'] > .5 else 'open'} | xy {f['ee_gt_xy'] * 100:.0f}cm z {ez(f) * 100:+.0f}cm (x = wrong-card anchor)", fontsize=8)
        ax[r, 1].axhline(f["ee"][2], color="gray", ls=":"); ax[r, 1].axhline(f["gt"][2], color="green", lw=1); ax[r, 1].set_title("chunk z (m)", fontsize=8); ax[r, 2].set_title("3D dist to GT (cm) | first-action jump " + " ".join(f"{c}={np.linalg.norm(x(f, c)[0] - f['ee']) * 100:.1f}" for c in CS), fontsize=7)
        for a in ax[r, 1:]: a.set_xlabel("chunk index", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "g17_cases.png", dpi=55); print("saved")


if __name__ == "__main__":
    main()
