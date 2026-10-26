#!/usr/bin/env python
"""G6 paired visuals: for each case a 2-row figure (top: raw G4d = G5 run, bottom: G4d + stabilisation) with top-down trajectory + rollout frames of the policy's front camera.
Overlays: GT cards (white; lime = instructed), planner target cross (green = nearest card is the instructed one, red = other), selected fine location (magenta star), LOCK location (orange diamond, stab only), selected crop (orange box), EE (cyan)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g4b_geometry import CAL, proj  # noqa: E402
from g5_analysis import analyse, nearest, OUT  # noqa: E402
from g5_visual import topdown  # noqa: E402

FIG = OUT / "figures"


def load(m): return {e["seed"]: (k, e) for k, e in enumerate(json.load(open(OUT / m / "episodes_raw.json")))}


def row(axs, m, seed, steps, eps):
    k, ep = eps[seed]; a = analyse(ep); S = ep["steps"]; tgt = ep["meta"]["target_entity"]; topdown(axs[0], ep, a, f"{m} seed {seed} {tgt} {'SUCCESS' if ep['success'] else 'fail'} len {a['len']}\n(* correct plan, x wrong, black sq = close)")
    cap = cv2.VideoCapture(str(OUT / m / "videos" / f"eval_episode_{k}.mp4")); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for q, st in enumerate(steps):
        st = int(min(st, nf - 1, len(S) - 1)); cap.set(cv2.CAP_PROP_POS_FRAMES, st); ok, f = cap.read(); ax = axs[q + 1]; ax.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) if ok else None; s = S[st]
        for kk, v in s["candidates_world"].items(): u, v_ = proj(CAL[0], v); ax.plot(u, v_, "o", mfc="none", mec="lime" if kk == tgt else "white", ms=14 if kk == tgt else 9, mew=2 if kk == tgt else 1)
        eu, ev = proj(CAL[0], s["ee_pos"]); ax.plot(eu, ev, "c.", ms=8)
        if s["planner_target_world"] is not None: pt = s["planner_target_world"]; okp = nearest(s["candidates_world"], pt)[0] == tgt; u, v_ = proj(CAL[0], pt); ax.plot(u, v_, "+", color="lime" if okp else "red", ms=17, mew=3)
        g = s.get("grounder"); act = ""
        if g and "boxes" in g:
            x0, y0, sd = g["boxes"][g["selected"]]; ax.add_patch(plt.Rectangle((x0, y0), sd, sd, fill=False, ec="orange", lw=1.2)); ax.plot(*g["fine_px_front"], "*", color="magenta", ms=12, mec="k")
            st_ = g.get("stab")
            if st_ and st_["locked"] and st_["L"] is not None: ax.plot(*st_["L"], "D", color="orange", ms=9, mec="k")
            act = f" {st_['action']}/{st_['used']}" if st_ and st_["in_plan"] else ""
        ax.set_xlim(60, 420); ax.set_ylim(400, 100); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"t={st} EE-GT {s['ee_to_target_dist'] * 100:.0f}cm{' REPLAN' if s['planner_replanned_this_step'] else ''}{' CLOSE' if s['reactive_close_event'] else ''}{act}", fontsize=7)


def case(name, seed, steps, note, raw, stab):
    fig, ax = plt.subplots(2, len(steps) + 1, figsize=(3.2 * (len(steps) + 1), 7.2)); row(ax[0], "g4d", seed, steps, raw); row(ax[1], "g4d_stab", seed, steps, stab); fig.suptitle(note, fontsize=9); fig.tight_layout(); fig.savefig(FIG / f"g6_case_{name}.png", dpi=52); plt.close(fig)


def main():
    raw, stab = load("g4d"), load("g4d_stab")
    first_move = lambda m, seed: next((i for i, s in enumerate((raw if m == 'g4d' else stab)[seed][1]["steps"]) if np.linalg.norm(np.array(s["candidates_world"][(raw if m == 'g4d' else stab)[seed][1]["meta"]["target_entity"]])[:2] - np.array((raw if m == 'g4d' else stab)[seed][1]["steps"][0]["candidates_world"][(raw if m == 'g4d' else stab)[seed][1]["meta"]["target_entity"]])[:2]) > 0.03), 100)
    mv = first_move("g4d_stab", 3007)
    case("1_late_switching_stabilised_3007", 3007, [1, 100, 200, 300, 400, 499], "CASE 1 - G5 late-stage switching (seed 3007): raw G4d (top) vs stabilised (bottom)", raw, stab)
    case("1b_late_switching_stabilised_3006", 3006, [1, 100, 200, 300, 400, 499], "CASE 1b - seed 3006: raw (top) vs stabilised (bottom)", raw, stab)
    case("2_initial_correct_lock_3003", 3003, [1, 60, 150, 250, 350, 450], "CASE 2 - initial correct lock kept (seed 3003)", raw, stab)
    case("3_wrong_initial_3009_and_3000", 3009, [1, 50, 100, 150, 165, 171], "CASE 3a - initial wrong fine location (seed 3009): lock follows the fine location to the correct card", raw, stab)
    case("3b_wrong_lock_persists_3000", 3000, [1, 30, 50, 70, 83, 84], "CASE 3b - wrong lock persists (seed 3000): stabilised run still SUCCEEDS", raw, stab)
    case("4_contact_motion_3007", 3007, sorted({1, max(1, mv - 10), mv, mv + 30, mv + 80, mv + 150}), f"CASE 4 - cards moved by contact (seed 3007, first target motion at t={mv} in the stabilised run): identity lock (orange diamond) vs anchor (cross)", raw, stab)
    case("5_stab_induced_failure_3001", 3001, [1, 60, 120, 170, 250, 400], "CASE 5 - stabilised run fails where raw succeeded (seed 3001)", raw, stab)
    print("saved")


if __name__ == "__main__":
    main()
