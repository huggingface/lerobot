#!/usr/bin/env python
"""G10-Z visuals. (a) poker demonstrations: EE z vs absolute time (band) and vs time-to-close (band) -> the descent is a schedule in time-to-close; (b) EARLY-frame dz curves (chunk z - EE z) overlaid: a near-identical template;
(c) state-z gain profile (S1/S2) by chunk segment; (d) four representative frames: chunk z under Z0 / target branch OFF / feature OFF / image tokens masked / state z +-2cm."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from g10_analysis import O10, SEG, stratum  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


def main():
    F = []
    for src in ("g4d", "oracle"):
        for fr in pickle.load(open(O10 / src / "audit10.pkl", "rb")): fr["src"] = src; fr["stratum"] = stratum(fr); F.append(fr)
    d = LeRobotDataset("lerobot/vlabench_unified"); fi = d.meta.episodes["dataset_from_index"]; ti = d.meta.episodes["dataset_to_index"]; E = pd.read_json("outputs/eval/safediff_vla_phase_d0_training_data_audit/episodes.json"); P = E[E.task == "select_poker"].iloc[:200]
    Z = []; TC = []
    for _, r in P.iterrows(): ep = int(r.episode); s = np.array(d.hf_dataset[int(fi[ep]):int(ti[ep])]["observation.state"]); Z.append(s[:, 2]); TC.append(int(r.t_close))
    fig, ax = plt.subplots(2, 4, figsize=(22, 8.4))
    L = max(len(z) for z in Z); A = np.full((len(Z), L), np.nan)
    for i, z in enumerate(Z): A[i, :len(z)] = z
    for z in Z[:60]: ax[0, 0].plot(z, color="tab:blue", alpha=.15)
    ax[0, 0].set_title("poker demos: EE z vs ABSOLUTE time (60 of 200 episodes)", fontsize=9); ax[0, 0].set_xlabel("t"); ax[0, 0].set_ylabel("z (m)")
    taus = np.arange(-55, 25); R = np.full((len(Z), len(taus)), np.nan)
    for i, (z, tc) in enumerate(zip(Z, TC)):
        for j, tau in enumerate(taus):
            if 0 <= tc + tau < len(z): R[i, j] = z[tc + tau]
    for i in range(60): ax[0, 1].plot(taus, R[i], color="tab:red", alpha=.15)
    ax[0, 1].set_title("same episodes vs TIME-TO-CLOSE (tau = t - t_close): collapses onto one curve", fontsize=9); ax[0, 1].set_xlabel("tau")
    E_ = [f for f in F if f["stratum"] == "EARLY"]
    for f in E_: ax[0, 2].plot((f["cond"]["Z0"]["xyz"][:, 2] - f["ee_robot"][2]) * 100, color="tab:green", alpha=.35)
    ax[0, 2].set_title(f"EARLY frames (n={len(E_)}): chunk z - EE z (cm), Z0 (GT anchor)", fontsize=9); ax[0, 2].set_xlabel("chunk index")
    xs = np.arange(len(SEG))
    for var, ls in (("S1", "--"), ("S2", "-")):
        for st, col in zip(("EARLY", "MID", "NEAR/CONTACT"), ("tab:green", "tab:blue", "tab:red")):
            fs = [f for f in F if f["stratum"] == st]; ax[0, 3].plot(xs, [np.mean([((f["cond"][f"{var}_p1"]["xyz"][a:b, 2] - f["cond"][f"{var}_m1"]["xyz"][a:b, 2]).mean()) / 0.02 for f in fs]) for a, b in SEG.values()], "o" + ls, color=col, label=f"{var} {st}")
    ax[0, 3].axhline(0, color="k", lw=.5); ax[0, 3].axhline(1, color="gray", ls=":", lw=.5); ax[0, 3].set_xticks(xs); ax[0, 3].set_xticklabels(list(SEG)); ax[0, 3].set_title("d(chunk z)/d(EE z state), +-1cm: follows state early, ~0 / negative late", fontsize=9); ax[0, 3].legend(fontsize=6)
    gz = lambda f: np.mean(f["cond"]["Z0"]["xyz"][:, 2]); picks = []
    for st in ("EARLY", "MID", "NEAR/CONTACT"): fs = sorted([f for f in F if f["stratum"] == st], key=gz); picks.append((st + " (median)", fs[len(fs) // 2]))
    picks.append(("z-misaligned (XY<4cm, EE >6cm above anchor)", max((f for f in F if f["ee_gt_xy"] < .04 and f["ee_gt_z"] > .06), key=lambda f: f["ee_gt_z"])))
    for c, (title, f) in enumerate(picks):
        a = ax[1, c]
        for key, col, ls, lab in (("Z0", "k", "-", "Z0 normal"), ("Z1", "tab:orange", "--", "target branch OFF"), ("Z3", "tab:purple", "--", "grounded feature OFF"), ("V1", "tab:red", "-", "image tokens masked"), ("S1_p2", "tab:green", ":", "state z +2cm"), ("S1_m2", "tab:cyan", ":", "state z -2cm")): a.plot(f["cond"][key]["xyz"][:, 2], ls, color=col, lw=1.6, label=lab)
        a.axhline(f["ee_robot"][2], color="gray", lw=.8, ls=":"); a.axhline(f["gt_anchor_robot"][2], color="green", lw=.8); a.set_title(f"{title}\n{f['src']} ep{f['ep']} t={f['t']} | chunk z (m); gray = EE z, green = anchor z", fontsize=8); a.set_xlabel("chunk index")
    ax[1, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(O10 / "g10_vertical_audit.png", dpi=52); print("saved")


if __name__ == "__main__":
    main()
