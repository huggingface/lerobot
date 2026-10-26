#!/usr/bin/env python
"""G5 visual verification: (1) top-down trajectory grid (seed x model); (2) case figures with rollout frames (front camera = the policy's view) overlaid with GT cards (white; lime = instructed), planner target (cross: green if nearest card is the instructed one, red otherwise),
per-frame monitor target (yellow dot), EE (cyan), and for G4d the selected fine crop (box) + alpha-weighted fine location (magenta star)."""
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
from g5_analysis import analyse, nearest  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g5_closed_loop"); FIG = OUT / "figures"; MODELS = ("v2", "g3", "g4d")


def load():
    raw = {m: json.load(open(OUT / m / "episodes_raw.json")) for m in MODELS}; ana = {m: [analyse(e) for e in raw[m]] for m in MODELS}; return raw, ana


def topdown(ax, ep, a, title):
    S = ep["steps"]; tgt = ep["meta"]["target_entity"]; c0 = S[0]["candidates_world"]; ee = np.array([s["ee_pos"] for s in S])
    ax.scatter(ee[:, 0], ee[:, 1], c=np.arange(len(ee)), cmap="viridis", s=2, zorder=2)
    for k, v in c0.items(): ax.add_patch(plt.Circle(v[:2], 0.03, fill=False, ec="lime" if k == tgt else "gray", lw=2 if k == tgt else 1)); ax.text(v[0], v[1] + 0.04, k.split("_of_")[0][:3] + k.split("_of_")[1][0], fontsize=6, ha="center")
    for j, r in enumerate(a["replans"]):
        pt = S[r["i"]]["planner_target_world"]; ax.plot(pt[0], pt[1], "*" if r["target_correct"] else "x", color="green" if r["target_correct"] else "red", ms=6, zorder=3)
    ce = [i for i, s in enumerate(S) if s["reactive_close_event"]]
    if ce: ax.plot(ee[ce[0], 0], ee[ce[0], 1], "ks", ms=7, zorder=4)
    ax.set_aspect("equal"); ax.set_xlim(-0.45, 0.45); ax.set_ylim(-0.1, 0.4); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=7)


def frames_fig(m, idx, raw, ana, steps, path, note):
    ep = raw[m][idx]; a = ana[m][idx]; S = ep["steps"]; tgt = ep["meta"]["target_entity"]; cap = cv2.VideoCapture(str(OUT / m / "videos" / f"eval_episode_{idx}.mp4")); n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); steps = [min(s, n_frames - 1, len(S) - 1) for s in steps]
    fig, ax = plt.subplots(1, len(steps) + 1, figsize=(3.3 * (len(steps) + 1), 3.6)); topdown(ax[0], ep, a, f"{m} seed {ep['seed']} target {tgt}\n(* correct plan, x wrong, black sq = close)")
    for q, st in enumerate(steps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, st); ok, f = cap.read(); f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else np.zeros((480, 480, 3), np.uint8); a_ = ax[q + 1]; a_.imshow(f); s = S[st]
        for k, v in s["candidates_world"].items(): u, v_ = proj(CAL[0], v); a_.plot(u, v_, "o", mfc="none", mec="lime" if k == tgt else "white", ms=14 if k == tgt else 9, mew=2 if k == tgt else 1)
        eu, ev = proj(CAL[0], s["ee_pos"]); a_.plot(eu, ev, "c.", ms=8)
        if s["planner_target_world"] is not None:
            pt = s["planner_target_world"]; ok_ = nearest(s["candidates_world"], pt)[0] == tgt; u, v_ = proj(CAL[0], pt); a_.plot(u, v_, "+", color="lime" if ok_ else "red", ms=16, mew=3)
        if s["monitor_target_world"] is not None: u, v_ = proj(CAL[0], s["monitor_target_world"]); a_.plot(u, v_, "y.", ms=7)
        g = s.get("grounder")
        if g and "boxes" in g:
            x0, y0, sd = g["boxes"][g["selected"]]; a_.add_patch(plt.Rectangle((x0, y0), sd, sd, fill=False, ec="orange", lw=1.5)); a_.plot(*g["fine_px_front"], "*", color="magenta", ms=12, mec="k")
        a_.set_xlim(60, 420); a_.set_ylim(400, 100); a_.set_xticks([]); a_.set_yticks([]); a_.set_title(f"t={st} EE-GT {s['ee_to_target_dist'] * 100:.0f}cm phase {s['phase']}{' REPLAN' if s['planner_replanned_this_step'] else ''}{' CLOSE' if s['reactive_close_event'] else ''}", fontsize=7)
    fig.suptitle(note, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=60); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True); raw, ana = load(); n = len(raw["v2"])
    fig, ax = plt.subplots(n, 3, figsize=(10.5, 3.0 * n))
    for i in range(n):
        for j, m in enumerate(MODELS): topdown(ax[i, j], raw[m][i], ana[m][i], f"{m} s{raw[m][i]['seed']} {raw[m][i]['meta']['target_entity']} {'SUCCESS' if raw[m][i]['success'] else ''}")
    fig.tight_layout(); fig.savefig(FIG / "g5_trajectory_grid.png", dpi=55); plt.close(fig)
    seeds = [e["seed"] for e in raw["v2"]]; ix = {s: i for i, s in enumerate(seeds)}
    # case 3: G4d map-correct-but-anchor-wrong replan with largest planner->GT
    c2 = [(r["planner_to_gt_xy"], i, r["i"]) for i, e in enumerate(ana["g4d"]) for r in e["replans"] if r.get("case") == 2]; c2.sort(reverse=True)
    # case 4: G4d late-stage instability: episode with most planner card switches (+ monitor switches) among those that never closed
    inst = sorted(((e["planner_card_switches"] + e["monitor"]["switches"] / 10, i) for i, e in enumerate(ana["g4d"]) if e["len"] >= 400), reverse=True)
    cases = {"1_v2_wrong_card": ("v2", ix[3006], [1, 20, 50, 58, 100, 200], "CASE 1 - V2 wrong-card (seed 3006): V2 plans on the wrong card and closes on it"),
             "2_g4d_recovery": ("g4d", ix[3001], [1, 40, 60, 110, 160, 172], "CASE 2 - G4d correct-card recovery (seed 3001): V2 was 0/11 correct on this seed (see V2 row)"),
             "2b_v2_same_seed": ("v2", ix[3001], [1, 40, 55, 100, 200, 300], "CASE 2b - V2 on the same seed 3001 (reference for case 2)"),
             "3_g4d_map_correct_anchor_wrong": ("g4d", c2[0][1], sorted({1, max(1, c2[0][2] - 1), c2[0][2] + 20, c2[0][2] + 49, c2[0][2] + 100}), f"CASE 3 - G4d map-correct / anchor-wrong replan (seed {seeds[c2[0][1]]}, replan at t={c2[0][2]}, planner->GT {c2[0][0] * 100:.0f} cm)"),
             "4_g4d_late_instability": ("g4d", inst[0][1], [1, 100, 200, 300, 400, 499], f"CASE 4 - G4d late-stage instability suspect (seed {seeds[inst[0][1]]}: most planner/monitor card switches)"),
             "5_g4d_success": ("g4d", ix[3008], [1, 60, 120, 200, 215, 284], "CASE 5 - G4d success (seed 3008)")}
    for k, (m, i, st, note) in cases.items(): frames_fig(m, i, raw, ana, st, FIG / f"g5_case_{k}.png", note)
    print("saved", list(cases), "case3 episode", seeds[c2[0][1]], "case4 episode", seeds[inst[0][1]])


if __name__ == "__main__":
    main()
