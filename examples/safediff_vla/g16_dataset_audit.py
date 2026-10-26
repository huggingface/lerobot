#!/usr/bin/env python
"""G16-D dataset audit (before any training): does the smooth readiness feature r_descend = sigmoid((d0 - r_xy)/tau), d0 = 5cm, tau = 2cm, match the demonstrations? Uses the G11 frame table (6 tasks, every 2nd step up to t_close+15; target anchor = first open->close commanded xyz).
PRE_GRASP = gripper state open (<=0.5) and t <= t_close. Per r_xy bin: next10 XY progress (r_xy minus the XY distance of the mean of the next 10 actions to the anchor), next10 Z descent (EE z minus mean next-10 action z), fraction with close within 10 steps, fraction of frames that are PRE_GRASP, mean r_descend. Also the within-bin relation of descent to e_z (rho = descent / |e_z|, frames with |e_z|>3cm)."""
from __future__ import annotations
import numpy as np

D0, TAU = 0.05, 0.02; Z = np.load("outputs/eval/safediff_vla_g11_audit/frames.npz", allow_pickle=True); TASKS = ["poker", "mahjong", "fruit", "drink", "toy", "book"]


def main():
    s, T, g, t, tc, task = Z["s"], Z["T"], Z["g"], Z["t"], Z["tc"], Z["task"]; e = T - s; rxy = np.linalg.norm(e[:, :2], axis=1); ttc = Z["ttc"]; a10 = Z["abs_first10"]
    pre = (g <= 0.5) & (t <= tc); rd = 1 / (1 + np.exp(-(D0 - rxy) / TAU)); prog = rxy - np.linalg.norm(T[:, :2] - a10[:, :2], axis=1); desc = s[:, 2] - a10[:, 2]
    bins = [("> 10 cm", rxy > .10), ("5-10 cm", (rxy > .05) & (rxy <= .10)), ("< 5 cm", rxy <= .05)]
    print(f"frames {len(t)} | PRE_GRASP {pre.mean():.2f}\n== ALL tasks, all frames ==")
    for nm, scope in (("ALL frames", np.ones_like(pre)), ("PRE_GRASP only", pre)):
        print(f" [{nm}]  {'r_xy bin':10s}{'n':>7s}{'PRE frac':>9s}{'XY prog(cm)':>12s}{'Z desc(cm)':>11s}{'close<=10':>10s}{'|e_z| cm':>9s}{'rho_z':>7s}{'r_descend':>10s}")
        for b, m in bins:
            mm = m & scope; ez = np.abs(e[mm, 2]); big = ez > .03; rho = np.median(desc[mm][big] / ez[big]) if big.sum() > 10 else np.nan
            print(f"       {b:10s}{int(mm.sum()):7d}{pre[m].mean():9.2f}{prog[mm].mean() * 100:12.2f}{desc[mm].mean() * 100:11.2f}{np.mean(ttc[mm] <= 10):10.2f}{ez.mean() * 100:9.1f}{rho:7.2f}{rd[mm].mean():10.2f}")
    print("== per task (PRE_GRASP): Z descent next10 (cm) / rho_z by r_xy bin ==")
    for i, tk in enumerate(TASKS):
        print(f"   {tk:8s} " + " | ".join(f"{b}: n={int((m & pre & (task == i)).sum()):5d} desc {desc[m & pre & (task == i)].mean() * 100:5.2f} prog {prog[m & pre & (task == i)].mean() * 100:5.2f}" for b, m in bins))
    m = pre; c = np.corrcoef(rd[m], desc[m])[0, 1]; c2 = np.corrcoef(rd[m], desc[m] / np.maximum(np.abs(e[m, 2]), .01))[0, 1]; print(f"== PRE_GRASP correlation: r_descend vs Z-descent {c:+.3f}; vs normalised descent (desc/|e_z|) {c2:+.3f}; r_xy vs XY progress {np.corrcoef(rxy[m], prog[m])[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
