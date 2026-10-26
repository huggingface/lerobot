#!/usr/bin/env python
"""G12 visuals: SAME observation, chunk under forced OPEN vs forced CLOSED gripper state (B = consistent intervention solid, A = decoder-state-only dashed), original C0 in black. top: side view (y,z); bottom: chunk z vs index with the EE z (gray dotted) and the anchor z (green)."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from g12_analysis import O12, stratum  # noqa: E402


def main():
    F = []
    for src in ("g4d", "oracle"):
        for fr in pickle.load(open(O12 / src / "audit12.pkl", "rb")): fr["src"] = src; fr["stratum"] = stratum(fr); F.append(fr)
    eff = lambda f: (f["cond"]["B_closed"]["xyz"][29, 2] - f["cond"]["B_open"]["xyz"][29, 2])
    big = [f for f in F if f["ee_gt_z"] > .06]; med = lambda fs: sorted(fs, key=eff)[len(fs) // 2]; top = lambda fs: max(fs, key=eff)
    picks = [("large vertical gap (largest effect)", top([f for f in big if f["ee_gt_z"] > .15])), ("MID (median effect)", med([f for f in big if f["stratum"] == "MID"])), ("LATE (median effect)", med([f for f in big if f["stratum"] == "LATE"] or big)),
             ("target directly above (XY<3cm)", med([f for f in big if f["ee_gt_xy"] < .03] or big)), ("after contact (card displaced)", med([f for f in big if f["any_card_displaced_cm"] > 1.5 and f["ee_gt_xy"] < .12] or big))]
    fig, ax = plt.subplots(2, 5, figsize=(22, 8.4))
    for c, (title, f) in enumerate(picks):
        ee = f["ee_robot"]; gt = f["gt_anchor_robot"]
        for key, col, ls, lab in (("C0", "k", "-", f"original (grip {'closed' if f['grip_raw'] > .5 else 'open'})"), ("B_open", "tab:green", "-", "forced OPEN (B)"), ("B_closed", "tab:red", "-", "forced CLOSED (B)"), ("A_open", "tab:green", "--", "forced OPEN (A)"), ("A_closed", "tab:red", "--", "forced CLOSED (A)")):
            ch = f["cond"][key]["xyz"]; ax[0, c].plot(ch[:, 1], ch[:, 2], ls, color=col, lw=1.8); ax[1, c].plot(ch[:, 2], ls, color=col, lw=1.8, label=lab if c == 0 else None)
        ax[0, c].plot(gt[1], gt[2], "*", color="green", ms=16, mec="k"); ax[0, c].plot(ee[1], ee[2], "ko", ms=7); ax[1, c].axhline(ee[2], color="gray", ls=":"); ax[1, c].axhline(gt[2], color="green", lw=.8)
        ax[0, c].set_title(f"{title}\n{f['src']} ep{f['ep']} t={f['t']} e_z={-f['ee_gt_z'] * 100:.0f}cm", fontsize=8); ax[0, c].set_xlabel("y (m)", fontsize=7); ax[1, c].set_xlabel("chunk index", fontsize=7)
    ax[1, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(O12 / "g12_open_vs_closed.png", dpi=52); print("saved")


if __name__ == "__main__":
    main()
