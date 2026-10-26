#!/usr/bin/env python
"""G14 visuals. Per case: (1) side view (y,z) of the EE for A raw (black) and B open-donor patch (red; patched steps as thick dots); target anchor star; (2) EE z over time with the physical gripper state (closed = shaded) for A and B and the patched steps (red ticks) ;
(3) the first logged intervention event of that episode: wrist original / patched / donor images; (4) planned chunk z at that event: unpatched counterfactual (black) vs patched (red), with the EE z (gray)."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
OUT = Path("outputs/eval/safediff_vla_g14"); BASE = np.array([0.0, -0.4, 0.78])


def main():
    L = {m: pickle.load(open(OUT / m / "g4d" / "g14_log.pkl", "rb")) for m in ("log", "open")}; E = {m: json.load(open(OUT / m / "g4d" / "episodes_raw.json")) for m in ("log", "open")}
    cases = [("1 raw closed-and-far HOLD vs patched DESCEND", 3005), ("2 patch with little effect", 3003), ("3 state CLOSED / image OPEN -> decoder descends", 3006), ("4 near target: patch off, normal close kept (no closed-and-far phase)", 3002), ("5 patch turns an A success into a failure", 3009)]
    fig, ax = plt.subplots(len(cases), 5, figsize=(25, 4.3 * len(cases)))
    for r, (title, seed) in enumerate(cases):
        ep = seed - 3000; eps = {m: E[m][ep] for m in E}; rows = {m: [x for x in L[m]["log"] if x["ep"] == ep] for m in E}; ee = {m: np.array([s["ee_pos"] for s in eps[m]["steps"]]) - BASE for m in E}; tgt = np.array(eps["log"]["steps"][0]["target_pos"]) - BASE + np.array([0, 0, 0.02])
        a = ax[r, 0]
        for m, col in (("log", "k"), ("open", "tab:red")):
            a.plot(ee[m][:, 1], ee[m][:, 2], "-", color=col, lw=1.2, alpha=.8, label=("A raw" if m == "log" else "B patched") + (" (success)" if eps[m]["success"] else " (fail)"))
            pt = [i for i, x in enumerate(rows[m]) if x["patched"]]
            if pt: a.plot(ee[m][pt, 1], ee[m][pt, 2], ".", color="orange", ms=3)
        a.plot(tgt[1], tgt[2], "*", color="green", ms=16, mec="k"); a.set_title(f"{title}\nseed {seed} side view (y,z); orange = patched steps", fontsize=8); a.legend(fontsize=6); a.set_xlabel("y (m)", fontsize=7)
        a = ax[r, 1]
        for m, off, col in (("log", 0.0, "k"), ("open", 0.0, "tab:red")):
            a.plot(ee[m][:, 2], color=col, lw=1.2, alpha=.8)
        for m, y0, col in (("log", 0.02, "gray"), ("open", 0.0, "tab:red")):
            g = np.array([x["grip"] > .5 for x in rows[m]]); a.fill_between(np.arange(len(g)), y0, y0 + 0.015, where=g, color=col, alpha=.5, step="mid")
            pt = [x["t"] for x in rows[m] if x["patched"]]; a.plot(pt, [y0 - 0.012] * len(pt), "|", color="orange", ms=4)
        a.axhline(tgt[2], color="green", lw=.8); a.set_title("EE z(t); bars: physical gripper closed (gray = A, red = B)", fontsize=8); a.set_xlabel("step", fontsize=7)
        evs = [e for e in L["open"]["events"] if e["ep"] == ep]
        if evs:
            e0 = evs[0]
            for c, (k, t_) in enumerate((("orig", "wrist original"), ("patched", "wrist patched"), ("donor_img", "open donor"))):
                pass
            im = np.concatenate([e0["orig"], e0["patched"], e0["donor_img"]], axis=1); ax[r, 2].imshow(im); ax[r, 2].set_title(f"t={e0['t']}: wrist original | patched | open-looking donor", fontsize=8); ax[r, 2].axis("off")
            rec = [x for x in rows["open"] if x["t"] == e0["t"]][0]; ax[r, 3].plot(rec["raw_chunk"]["xyz"][:, 2], "k", lw=2, label="unpatched (counterfactual)"); ax[r, 3].plot(rec["chunk"]["xyz"][:, 2], "r", lw=2, label="patched"); ax[r, 3].axhline(rec["ee"][2], color="gray", ls=":"); ax[r, 3].axhline(tgt[2], color="green", lw=.8)
            g0 = np.argmax(rec["raw_chunk"]["grip"] <= .5) if (rec["raw_chunk"]["grip"] <= .5).any() else 50; g1 = np.argmax(rec["chunk"]["grip"] <= .5) if (rec["chunk"]["grip"] <= .5).any() else 50; ax[r, 3].set_title(f"planned chunk z; first close idx {g0} -> {g1}; d3={rec['d3'] * 100:.0f}cm", fontsize=8); ax[r, 3].legend(fontsize=6)
        else: ax[r, 2].set_title("(no intervention event in this episode)", fontsize=8); ax[r, 2].axis("off"); ax[r, 3].axis("off")
        a = ax[r, 4]
        for m, col in (("log", "k"), ("open", "tab:red")): a.plot([x["d3"] * 100 for x in rows[m]], color=col, lw=1.2, label="A" if m == "log" else "B")
        a.axhline(6, color="gray", ls="--", lw=.8); a.axhline(3, color="green", ls=":", lw=.8); a.set_title("EE-to-GT 3D distance (cm); dashed 6cm trigger, dotted 3cm", fontsize=8); a.legend(fontsize=6); a.set_xlabel("step", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "g14_cases.png", dpi=48); print("saved")


if __name__ == "__main__":
    main()
