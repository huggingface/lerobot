#!/usr/bin/env python
"""G25 visuals: per task (toy, fruit), the GEO episodes grouped as stable / immediate slip / transport drop / goal-near held, each as 6 video frames (t_close-5 .. min(end, drop+25)) plus a trace panel (object z, EE z, finger width, is_grasped) with the event markers."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g25_analysis import events  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g25")


def main():
    for t in ("select_toy", "select_fruit"):
        L = pickle.load(open(OUT / t / "g25_trace.pkl", "rb")); vids = sorted((OUT / t / "videos").glob("*.mp4")); items = []
        for i, e in enumerate(L["episodes"]):
            S = [s for s in L["steps"] if s["ep"] == i]; ci = e["close"]
            if ci is None or not S: continue
            ev = events(S, max(ci - 1, 0)); items.append((i, e, S, ev))
        def cat(x):
            i, e, S, ev = x; held = ev["held_steps"]
            if held < 5: return "IMMEDIATE slip / never held"
            if ev["t_drop"] is not None and ev["t_drop"] - ev["t_close"] > 5: return "TRANSPORT drop"
            if ev["t_goal_near"] is not None and ev["t_drop"] is None: return "held to GOAL"
            return "held (no goal reached)"
        pick = {}
        for x in items: pick.setdefault(cat(x), x)
        fig, ax = plt.subplots(len(pick), 7, figsize=(30, 4.3 * len(pick)), squeeze=False)
        for r, (name, (i, e, S, ev)) in enumerate(pick.items()):
            cap = cv2.VideoCapture(str(vids[i])); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); end = min(len(S) - 1, (ev["t_drop"] + 25) if ev["t_drop"] is not None else len(S) - 1); frs = np.linspace(max(ev["t_close"] - 5, 0), end, 6).astype(int)
            for c, f in enumerate(frs): cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(f, nf - 1))); ok, im = cap.read(); ax[r, c].imshow(im[..., ::-1]) if ok else None; ax[r, c].axis("off"); ax[r, c].set_title((f"{name} | seed {e['seed']} succ {int(e['success'])}\n" if c == 0 else "") + f"step {f}", fontsize=8)
            a = ax[r, 6]; tt = np.arange(len(S)); a.plot(tt, [s["obj_p"][2] for s in S], label="object z"); a.plot(tt, [s["ee_p"][2] for s in S], label="EE z"); a.plot(tt, [s["width"] * 3 for s in S], label="finger width x3"); a.plot(tt, [0.9 + 0.05 * s["grasped"] for s in S], "k", lw=0.8, label="is_grasped")
            for k, col in (("t_close", "g"), ("t_grasp", "c"), ("t_slip", "orange"), ("t_drop", "r"), ("t_goal_near", "m")):
                if ev[k] is not None: a.axvline(ev[k], color=col, ls="--", lw=1, label=k)
            a.set_xlim(0, min(len(S), (ev["t_drop"] + 60) if ev["t_drop"] is not None else len(S))); a.legend(fontsize=5); a.set_title("trace (world units, m)", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / f"g25_video_{t}.png", dpi=36); plt.close(fig)
    print("saved")


if __name__ == "__main__":
    main()
