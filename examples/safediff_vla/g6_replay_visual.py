#!/usr/bin/env python
"""G6 replay overlay: rollout frames at selected replans. lime ring = instructed card (GT, eval only); magenta star = RAW selected fine location; orange diamond = STABILISED lock location L; orange box = selected crop; caption shows replan idx, raw/stab correctness, stabiliser action."""
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
from g6_replay import OUT, replay_episode  # noqa: E402
from g6_stabilizer import Stabilizer  # noqa: E402


def locs(ep):
    """re-run the replay but also keep the lock location per replan"""
    from g6_replay import card_px, nearest_px  # noqa
    S = ep["steps"]; stab = Stabilizer(); out = []
    for i in [i for i in range(1, len(S)) if S[i]["planner_replanned_this_step"]]:
        g = S[i]["grounder"]; sel = g["selected"]; fp = np.array(g["fine_px_front"]); c = g["final_front_argmax"]; mc = np.array([(c % 8 + .5) * 60, (c // 8 + .5) * 60]); cands = []
        for j, (b, p) in enumerate(zip(g["boxes"], g["p"])): x0, y0, s = b; cands.append({"box": tuple(b), "p": float(p), "fine": tuple(fp) if j == sel else None, "map_ok": bool(j == sel and x0 <= mc[0] <= x0 + s and y0 <= mc[1] <= y0 + s)})
        dec = stab.step(cands); out.append((i, fp, stab.L.copy() if stab.locked else None, g["boxes"][sel], dec["action"]))
    return out


def main():
    eps = {e["seed"]: (k, e) for k, e in enumerate(json.load(open(OUT / "g4d" / "episodes_raw.json")))}; rep = {r["seed"]: r for r in [replay_episode(e) for _, e in eps.values()]}; picks = {3005: [0, 3, 6, 9], 3006: [0, 3, 6, 9], 3007: [0, 3, 6, 9], 3000: [0, 3, 6, 9], 3009: [0, 1, 2, 4]}
    fig, ax = plt.subplots(len(picks), 4, figsize=(15, 3.7 * len(picks)))
    for r, (seed, js) in enumerate(picks.items()):
        k, e = eps[seed]; S = e["steps"]; tgt = e["meta"]["target_entity"]; L = locs(e); cap = cv2.VideoCapture(str(OUT / "g4d" / "videos" / f"eval_episode_{k}.mp4")); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for c, j in enumerate(js):
            j = min(j, len(L) - 1); i, fp, lk, box, act = L[j]; cap.set(cv2.CAP_PROP_POS_FRAMES, min(i - 1, nf - 1)); ok, f = cap.read(); a = ax[r, c]; a.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); d = i - 1
            for kk, v in S[d]["candidates_world"].items(): u, v_ = proj(CAL[0], v); a.plot(u, v_, "o", mfc="none", mec="lime" if kk == tgt else "white", ms=15 if kk == tgt else 9, mew=2 if kk == tgt else 1)
            x0, y0, s = box; a.add_patch(plt.Rectangle((x0, y0), s, s, fill=False, ec="orange", lw=1.5)); a.plot(*fp, "*", color="magenta", ms=15, mec="k")
            if lk is not None: a.plot(*lk, "D", color="orange", ms=9, mec="k")
            row = rep[seed]["rows"][j]; a.set_xlim(60, 420); a.set_ylim(400, 100); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"seed {seed} replan {j + 1} (t={i}) act={act} | raw {'OK' if row['raw_correct'] else 'BAD'} stab {'OK' if row['stab_correct'] else 'BAD'}", fontsize=8)
    fig.suptitle("G6 replay overlay (magenta star = raw fine location, orange diamond = locked location, lime = instructed card)", fontsize=10); fig.tight_layout(); fig.savefig(OUT / "figures" / "g6_replay_overlay.png", dpi=48); print("saved")


if __name__ == "__main__":
    main()
