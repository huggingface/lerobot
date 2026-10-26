#!/usr/bin/env python
"""G1 visual verification. Per task, held-out TEST scenes (K>=3, deterministic order): SAME RGB, each valid referent instruction.
Rows = arms (V2 reference, A0..A3, seed 0 selected checkpoints), columns = instructions. Each cell: the camera image with the referent
score map (A1-A3: softmax weights over the 192 image tokens; V2/A0: head-averaged cross-attention weights) overlaid,
GREEN ring = GT grasp anchor of the instructed referent, thin white circles = other candidates' GT anchors, RED x = predicted anchor.
Prediction is projected with the static-camera calibration (robot frame + robot base -> world -> pixel)."""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from g1_train import ARMS, Bottleneck, D, OUTT, TaskData, V2Modules  # noqa: E402

OUTF = Path("outputs/eval/safediff_vla_g1_offline/figures")


def proj(cam, xyz_world):
    R = np.array(cam["xmat"]).reshape(3, 3); p = R.T @ (np.asarray(xyz_world) - np.array(cam["xpos"])); f = 240 / np.tan(np.deg2rad(cam["fovy"]) / 2)
    return f * p[0] / -p[2] + 240, 240 - f * p[1] / -p[2]


def main():
    OUTF.mkdir(parents=True, exist_ok=True); cal = json.load(open(D / "camera_calibration.json")); dev = "cuda"
    for task in ("select_poker", "select_fruit"):
        data = TaskData(task, dev); cams = cal[task]["cams_policy_order"]
        models = {"V2": V2Modules().to(dev)}
        for arm in ("A0", "A1", "A2", "A3"):
            m = (V2Modules(warm=False) if ARMS[arm]["model"] == "v2" else Bottleneck()).to(dev); m.load_state_dict(torch.load(OUTT / task / f"{arm}_s0.pt")); models[arm] = m
        test_ids = [i for i in data.idx("test") if int(data.valid[i].sum()) >= 3][:3]
        for s in test_ids:
            sid = data.sids[s]; rec = json.load(open(D / task / f"{sid}.json")); z = np.load(D / task / f"{sid}.npz"); base = np.array(rec["robot_base_world"])
            K = len(rec["candidates"]); gt = [np.array(c["keypoint_world"]) for c in rec["candidates"]]
            tok, mod, anc, valid, cf, held = data.batch([s])
            for ci in (0, 1):
                fig, ax = plt.subplots(len(models), K, figsize=(3.6 * K, 3.5 * len(models)))
                for r, (name, m) in enumerate(models.items()):
                    m.eval()
                    with torch.no_grad():
                        xyz_n, w = m(tok[:K * 0 + tok.shape[0]], mod)
                        if w is None: w = m.score_map(tok, mod)
                    pred = (xyz_n * data.std + data.mean).cpu().numpy(); w = w.cpu().numpy()
                    for k in range(K):
                        a = ax[r, k]; a.imshow(z["rgb"][ci]); wm = w[k, ci * 64:(ci + 1) * 64].reshape(8, 8)
                        a.imshow(wm, extent=(0, 480, 480, 0), cmap="jet", alpha=0.45, interpolation="nearest", vmin=0, vmax=max(w[k].max(), 1e-3))
                        for j in range(K):
                            u, v = proj(cams[ci], gt[j]); a.plot(u, v, "o", mfc="none", mec="white", ms=7, mew=1) if j != k else a.plot(u, v, "o", mfc="none", mec="lime", ms=15, mew=2.5)
                        u, v = proj(cams[ci], pred[k] + base); a.plot(u, v, "x", color="red", ms=12, mew=3)
                        err = np.linalg.norm((pred[k] - anc[0, k].cpu().numpy())[:2]) * 100; a.set_xticks([]); a.set_yticks([])
                        a.set_title(f"{name} | {rec['candidates'][k]['name']}\nerr {err:.1f} cm" if r else f"{name} | {rec['candidates'][k]['name']}  (err {err:.1f} cm)", fontsize=8)
                        if r == 0 and k == 0: a.set_ylabel(sid, fontsize=7)
                fig.suptitle(f"{task} {sid} camera{ci + 1}: same RGB, instruction varies | green ring = GT anchor of instructed referent, white = other candidates, red x = predicted, heat = referent score", fontsize=9)
                fig.tight_layout(); fig.savefig(OUTF / f"{task}_{sid}_cam{ci + 1}.png", dpi=75); plt.close(fig)
            print("fig", task, sid, flush=True)


if __name__ == "__main__":
    main()
