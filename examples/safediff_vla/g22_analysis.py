#!/usr/bin/env python
"""G22-A analysis (book, drink; seeds 3000-3004): G21 decoder (oracle point + learned decoder) | G21 geo (oracle point + world-z waypoint) | G22 pose (oracle grasp pose + approach-vector). Primary: pre-close displacement of target / other objects (collision proxy), <3cm reach, correct grasp (close at oracle<=3cm and target lifted>3cm or is_grasped), lift. Native success secondary and cross-checked by video."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g21_analysis import episode  # noqa: E402

O21 = Path("outputs/eval/safediff_vla_g21"); O22 = Path("outputs/eval/safediff_vla_g22"); TASKS = ["select_book", "select_drink"]; ARMS = [("G21 decoder", O21, "decoder"), ("G21 geo world-z", O21, "geo"), ("G22 pose", O22, "pose")]


def pre_close_disp(ep):
    S = ep["steps"]; tgt = ep["meta"]["target_entity"]; n = len(S); ci = next((i for i, s in enumerate(S) if s.get("reactive_close_event")), n); c0 = S[0]["candidates_world"]; tp = np.array([s["target_pos"] for s in S])
    dt = np.linalg.norm(tp[:ci] - tp[0], axis=1).max() * 100 if ci > 0 else 0.0; oth = [k for k in c0 if k != tgt]; do = max([np.linalg.norm(np.array(S[i]["candidates_world"][k]) - np.array(c0[k])) for i in range(ci) for k in oth] + [0.0]) * 100
    return dt, do, ci < n


def main():
    R = {}
    for t in TASKS:
        for nm, root, arm in ARMS:
            E = json.load(open(root / t / arm / "episodes_raw.json")); L = pickle.load(open(root / t / arm / "g21_log.pkl", "rb")); by = {i: [r for r in L["log"] if r["ep"] == i] for i in range(len(E))}; R[(t, nm)] = [dict(episode(e, by[i]), ep=i, pre_t=pre_close_disp(e)[0], pre_o=pre_close_disp(e)[1]) for i, e in enumerate(E)]
            R[(t, nm, "chunks")] = L["chunks"]
    print(f"{'task':14s}{'arm':18s}{'n':>3s}{'reach<3cm':>10s}{'closed':>7s}{'correct grasp':>14s}{'lift>3cm':>9s}{'tgt moved>1.5cm pre-close':>26s}{'other moved>1.5cm pre-close':>28s}{'median pre-close tgt/other disp cm':>36s}{'min3D cm':>9s}{'native SR':>10s}")
    for t in TASKS:
        for nm, _, _ in ARMS:
            r = R[(t, nm)]; n = len(r); cg = sum(x["close"] and x["close_d3"] is not None and x["close_d3"] <= 3.5 and (x["lifted"] or x["grasped_ever"]) for x in r)
            print(f"{t[7:]:14s}{nm:18s}{n:3d}{sum(x['reach'] for x in r):>7d}/{n}{sum(x['close'] for x in r):>4d}/{n}{cg:>11d}/{n}{sum(x['lifted'] for x in r):>6d}/{n}{sum(x['pre_t'] > 1.5 for x in r):>23d}/{n}{sum(x['pre_o'] > 1.5 for x in r):>25d}/{n}{np.median([x['pre_t'] for x in r]):>22.1f}/{np.median([x['pre_o'] for x in r]):<5.1f}{np.mean([x['min_3d'] for x in r]):9.1f}{sum(x['success'] for x in r):>8d}/{n}")
    print("\n== per-episode (min3D cm | close d3 cm | lift cm | pre-close tgt/other displacement cm | native success | category) ==")
    for t in TASKS:
        for nm, _, _ in ARMS: print(f" {t[7:]:7s}{nm:18s}" + " ; ".join(f"{x['seed']}: {x['min_3d']:.1f}|{('%.1f' % x['close_d3']) if x['close_d3'] is not None else '-'}|{x['lift_cm']:.1f}|{x['pre_t']:.1f}/{x['pre_o']:.1f}|{int(x['success'])}|{x['cat']}" for x in R[(t, nm)]))
    for t in TASKS:
        c = R[(t, "G22 pose", "chunks")]
        if c: print(f"  G22 pose chunk {t[7:]}: replans {len(c)} | first-action position offset mean {np.mean([x['first_off'] for x in c]) * 100:.2f} cm | first-step rotation change mean {np.mean([x['rot_first_deg'] for x in c]):.2f} deg")
    pickle.dump(R, open(O22 / "g22_summary.pkl", "wb"))
    import cv2, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for t in TASKS:
        sel = []
        p = R[(t, "G22 pose")]; g = R[(t, "G21 geo world-z")]
        for nm, arm, root, pool in (("pose SUCCESS", "pose", O22, [x for x in p if x["success"]]), ("pose FAILURE", "pose", O22, [x for x in p if not x["success"]]), ("pose (same seed 3003)", "pose", O22, [x for x in p if x["seed"] == 3003]), ("G21 geo (same seed 3003)", "geo", O21, [x for x in g if x["seed"] == 3003])):
            if pool: sel.append((nm, arm, root, pool[0]))
        fig, ax = plt.subplots(len(sel), 6, figsize=(24, 4 * len(sel)), squeeze=False)
        for r, (nm, arm, root, d) in enumerate(sel):
            vids = sorted((root / t / arm / "videos").glob("*.mp4")); cap = cv2.VideoCapture(str(vids[d["ep"]])); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for c, f in enumerate(np.linspace(0, nf - 1, 6).astype(int)): cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, im = cap.read(); ax[r, c].imshow(im[..., ::-1]) if ok else None; ax[r, c].axis("off"); ax[r, c].set_title(f"{nm} seed {d['seed']} [{d['cat']}] fr {f}/{nf}" if c == 0 else f"fr {f}", fontsize=8)
        fig.tight_layout(); fig.savefig(O22 / f"g22_video_{t}.png", dpi=40); plt.close(fig)
    print("saved sheets")


if __name__ == "__main__":
    main()
