#!/usr/bin/env python
"""G11-D visuals on the fixed observations: V2 decoder (solid) vs residual-conditioned decoder (dashed), chunk under the GT anchor. top = (x,y), bottom = side (y,z). EE black dot, GT anchor star. Columns: EARLY, MID, NEAR, contact, large vertical gap with gripper CLOSED, and a large vertical gap with gripper OPEN (for comparison)."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from g11_analysis import O11, stratum  # noqa: E402


def main():
    F = []
    for src in ("g4d", "oracle"):
        for fr in pickle.load(open(O11 / src / "audit11.pkl", "rb")): fr["src"] = src; fr["stratum"] = stratum(fr); F.append(fr)
    med = lambda fs: sorted(fs, key=lambda f: np.linalg.norm(f["cond"][("RES", "A1")]["xyz"] - f["cond"][("V2", "A1")]["xyz"], axis=1).mean())[len(fs) // 2]
    big = [f for f in F if f["ee_gt_z"] > .15]; picks = [("EARLY", med([f for f in F if f["stratum"] == "EARLY"])), ("MID", med([f for f in F if f["stratum"] == "MID"])), ("NEAR (xy<8cm)", med([f for f in F if f["ee_gt_xy"] < .08 and f["any_card_displaced_cm"] <= 1.5] or [f for f in F if f["stratum"] == "NEAR/CONTACT"])),
             ("contact (card displaced)", med([f for f in F if f["any_card_displaced_cm"] > 1.5 and f["ee_gt_xy"] < .12])), ("large vertical gap, gripper CLOSED", med([f for f in big if f["state7"][6] > .5])), ("large vertical gap, gripper OPEN", med([f for f in big if f["state7"][6] <= .5 and f["stratum"] != "EARLY"] or [f for f in big if f["state7"][6] <= .5]))]
    fig, ax = plt.subplots(2, 6, figsize=(24, 8.6))
    for c, (title, f) in enumerate(picks):
        gt = f["cond"][("V2", "A1")]["anchor_robot"]; ee = f["ee_robot"]
        for m, ls, col in (("V2", "-", "tab:blue"), ("RES", "--", "tab:red")):
            ch = f["cond"][(m, "A1")]["xyz"]
            for a, (i, j) in zip((ax[0, c], ax[1, c]), ((0, 1), (1, 2))): a.plot(ch[:, i], ch[:, j], ls, color=col, lw=2, label=m if c == 0 else None); a.plot(ch[0, i], ch[0, j], "o", color=col, ms=4); a.plot(ch[-1, i], ch[-1, j], "s", color=col, ms=5)
        for a, (i, j) in zip((ax[0, c], ax[1, c]), ((0, 1), (1, 2))): a.plot(gt[i], gt[j], "*", color="green", ms=15, mec="k"); a.plot(ee[i], ee[j], "ko", ms=7); a.tick_params(labelsize=6)
        ax[0, c].set_title(f"{title}\n{f['src']} ep{f['ep']} t={f['t']} grip={'closed' if f['state7'][6] > .5 else 'open'} e_z={-f['ee_gt_z'] * 100:.0f}cm", fontsize=8)
    ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(O11 / "g11_chunks_res_vs_v2.png", dpi=52); print("saved")


if __name__ == "__main__":
    main()
