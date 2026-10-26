#!/usr/bin/env python
"""G2 visual verification. SIM: same 3 held-out fruit test scenes as the G1 figures; DATASET: held-out episodes at f0/k45/k30/k10 with the true
instruction and a swapped referent. Rows = V2, G1-A3, G2 M1/M2/M3 (seed 0). green ring = GT anchor (instructed referent; sim: all candidates in white),
red x = predicted anchor, heat = referent score map (camera 1 tokens). Dataset frames are 224px: the sim camera-1 calibration is rescaled to
check that the GT ring lands on the fruit (a sanity check of the shared camera, not an assumption)."""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from g1_train import Bottleneck, D, TaskData, V2Modules  # noqa: E402
from g2_train import DataCache  # noqa: E402

OUTE = Path("outputs/eval/safediff_vla_g3_offline"); OUTF = OUTE / "figures"; G1T = Path("outputs/train/safediff_vla_g1/select_poker"); G2T = Path("outputs/train/safediff_vla_g3")


def proj(cam, xyz, size):
    R = np.array(cam["xmat"]).reshape(3, 3); p = R.T @ (np.asarray(xyz) - np.array(cam["xpos"])); f = 240 / np.tan(np.deg2rad(cam["fovy"]) / 2)
    return (f * p[0] / -p[2] + 240) * size / 480, (240 - f * p[1] / -p[2]) * size / 480


def load_models(dev):
    ms = {"V2": V2Modules().to(dev)}
    for name, path in (("G1-A3", G1T / "A3_s0.pt"), ("G3-M1", G2T / "M1_s0.pt")):
        m = Bottleneck().to(dev); m.load_state_dict(torch.load(path)); ms[name] = m.eval()
    return ms


def fwd(m, tok, mod, mean, std):
    with torch.no_grad():
        x, w = m(tok, mod)
        if w is None: w = m.score_map(tok, mod)
    return (x * std + mean).cpu().numpy(), w.cpu().numpy()


def main():
    OUTF.mkdir(parents=True, exist_ok=True); dev = "cuda"; cal = json.load(open(D / "camera_calibration.json"))["select_poker"]["cams_policy_order"][0]; models = load_models(dev)
    sim = TaskData("select_poker", dev); mean, std = sim.mean, sim.std
    TESTSC = [sim.sids[i] for i in sim.idx("test") if int(sim.valid[i].sum()) >= 3][:3] + [sim.sids[i] for i in sim.idx("test_heldout_identity") if int(sim.valid[i].sum()) >= 3][:1]
    for sid in TESTSC:
        s = sim.sids.index(sid); rec = json.load(open(D / "select_poker" / f"{sid}.json")); z = np.load(D / "select_poker" / f"{sid}.npz"); base = np.array(rec["robot_base_world"])
        K = len(rec["candidates"]); tok, mod, anc, *_ = sim.batch([s]); gt = [np.array(c["keypoint_world"]) for c in rec["candidates"]]
        fig, ax = plt.subplots(len(models), K, figsize=(3.6 * K, 3.5 * len(models)))
        for r, (name, m) in enumerate(models.items()):
            pred, w = fwd(m, tok, mod, mean, std)
            for k in range(K):
                a = ax[r, k]; a.imshow(z["rgb"][0]); a.imshow(w[k, :64].reshape(8, 8), extent=(0, 480, 480, 0), cmap="jet", alpha=0.45, interpolation="nearest", vmin=0, vmax=max(w[k].max(), 1e-3))
                for j in range(K):
                    u, v = proj(cal, gt[j], 480); a.plot(u, v, "o", mfc="none", mec="lime" if j == k else "white", ms=15 if j == k else 7, mew=2.5 if j == k else 1)
                u, v = proj(cal, pred[k] + base, 480); a.plot(u, v, "x", color="red", ms=12, mew=3)
                e = np.linalg.norm((pred[k] - anc[0, k].cpu().numpy())[:2]) * 100; held = rec["candidates"][k]["name"].startswith("9_of_")
                a.set_title(f"{name} | {rec['candidates'][k]['name']}{' (held-out rank 9)' if held else ''}\nerr {e:.1f} cm", fontsize=8); a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"SIM {sid} (camera1) - same RGB, instruction varies", fontsize=9); fig.tight_layout(); fig.savefig(OUTF / f"sim_{sid}.png", dpi=70); plt.close(fig); print("sim fig", sid, flush=True)
    # ---- dataset frames ----
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from _multitask_common import REPO_ID
    ds = LeRobotDataset(REPO_ID); from_idx = ds.meta.episodes["dataset_from_index"]; cam_key = ds.meta.camera_keys[0]
    h = DataCache(OUTE / "g3_held_cache.pt", dev); import pandas as pd
    eps = pd.read_json(Path("outputs/eval/safediff_vla_phase_d0_training_data_audit/episodes.json")).set_index("episode")
    chosen = []
    for e in dict.fromkeys(m["episode"] for m in h.meta):
        if len(chosen) == 3: break
        chosen.append(e)
    for e in chosen:
        cols = [(f, 0) for f in ("f0", "k45", "k30", "k10")] + [("k30", 1), ("f0", 1)]
        fig, ax = plt.subplots(len(models), len(cols), figsize=(3.3 * len(cols), 3.3 * len(models))); anc_w = np.array(eps.loc[e, "anchor"])
        for c, (fname, vi) in enumerate(cols):
            i = next(i for i, m in enumerate(h.meta) if m["episode"] == e and m["frame"] == fname); meta = h.meta[i]
            item = ds[int(from_idx[int(e)]) + meta["image_frame"]]; img = item[cam_key].permute(1, 2, 0).numpy() if item[cam_key].shape[0] == 3 else item[cam_key].numpy()
            if img.dtype != np.uint8: img = (img * 255).astype(np.uint8)
            tok = h.tok[i, vi:vi + 1].float(); mod = h.mod[i, vi:vi + 1]
            for r, (name, m) in enumerate(models.items()):
                pred, w = fwd(m, tok, mod, mean, std); a = ax[r, c]; a.imshow(img); a.imshow(w[0, :64].reshape(8, 8), extent=(0, 224, 224, 0), cmap="jet", alpha=0.45, interpolation="nearest", vmin=0, vmax=max(w.max(), 1e-3))
                if vi == 0: u, v = proj(cal, anc_w + np.array([0, -0.4, 0.78]), 224); a.plot(u, v, "o", mfc="none", mec="lime", ms=14, mew=2.5)
                u, v = proj(cal, pred[0] + np.array([0, -0.4, 0.78]), 224); a.plot(u, v, "x", color="red", ms=11, mew=3)
                err = np.linalg.norm((pred[0] - anc_w)[:2]) * 100
                a.set_title(f"{name}\n{fname} {'TRUE' if vi == 0 else 'SWAPPED'}: " + (f"err {err:.1f}cm" if vi == 0 else meta["texts"][1].replace('primitive: Put the ', '').split(' into')[0]), fontsize=8); a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"DATASET held-out episode {e}: '{meta['texts'][0]}' (green = demonstrated anchor, true text only)", fontsize=9); fig.tight_layout(); fig.savefig(OUTF / f"dataset_ep{e}.png", dpi=65); plt.close(fig); print("dataset fig", e, flush=True)


if __name__ == "__main__":
    main()
