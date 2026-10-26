#!/usr/bin/env python
"""G4d 3-seed visual check. (1) per seed: 4 sim queries (easy 3-card, hard 4-card, same-rank/suit, rank9) -> [front crops(colour=fine p)+star=alpha-weighted fine location] [coarse map] [fine-only map] [G4d final map + pred x vs GT ring] [G4d final (right)].
(2) per seed: ORIGINAL-dataset instruction swap at f0 / k30 / k10 (2 frames each): front frame + predicted anchors for the true instruction (red x; green ring = dataset GT anchor of the true instruction) and the two swapped referents (orange, cyan),
then the final map for each text. There is no GT for swapped texts (the swapped card may not be in the scene)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
import g4c_matcher_train as gc  # noqa: E402
import g4b_proposals as P  # noqa: E402
from _multitask_common import REPO_ID  # noqa: E402
from g1_train import D  # noqa: E402
from g3_identity_diagnostic import rs  # noqa: E402

PD = D / "select_poker"; OD = Path("outputs/eval/safediff_vla_g4d_offline"); BASE = np.array([0.0, -0.4, 0.78])


def proj(cam, xyz):
    R = np.array(cam["xmat"]).reshape(3, 3); p = R.T @ (np.asarray(xyz) - np.array(cam["xpos"])); f = 240 / np.tan(np.deg2rad(cam["fovy"]) / 2); return f * p[0] / -p[2] + 240, 240 - f * p[1] / -p[2]


def main():
    d = gc.Data(); cal = json.load(open(D / "camera_calibration.json"))["select_poker"]["cams_policy_order"]; names = d.names; sids = d.sim["sids"]; split = d.split; valid = d.sim["valid"].numpy(); held = d.sim["held"].numpy(); Bx = {}
    for p in sorted(PD.glob("bboxes_*.json")): Bx.update(json.load(open(p)))
    def find(sp, cond, used):
        for s in np.where(split == sp)[0]:
            ks = [k for k in range(4) if valid[s, k]]
            for k in ks:
                if (s, k) not in used and cond(s, k, ks): return (s, k)
    used = set(); rows = []
    cats = [("easy 3-card", "test", lambda s, k, ks: len(ks) == 3 and not held[s, k]), ("hard 4-card", "test", lambda s, k, ks: len(ks) == 4 and not held[s, k]),
            ("same-rank/suit distractor", "test", lambda s, k, ks: not held[s, k] and len(ks) == 4 and any(rs(names[s][c])[0] == rs(names[s][k])[0] or rs(names[s][c])[1] == rs(names[s][k])[1] for c in ks if c != k)),
            ("held-out rank 9", "test_heldout_identity", lambda s, k, ks: held[s, k] and len(ks) == 4)]
    for title, sp, cond in cats: x = find(sp, cond, used); used.add(x); rows.append((title, *x))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dset = LeRobotDataset(REPO_ID); from_idx = dset.meta.episodes["dataset_from_index"]; cam0 = dset.meta.camera_keys[0]; meta = d.d["dheld"]["meta"]; pick = []
    for fr in ("f0", "k30", "k10"):
        seen = set()
        for i, m in enumerate(meta):
            if m["frame"] == fr and m["episode"] not in seen and len(seen) < 2: seen.add(m["episode"]); pick.append((fr, i))
    for seed in (0, 1, 2):
        mdl = gc.G4cModel("add", 0.2).to("cuda"); mdl.load_state_dict(torch.load(OD / f"g4d_matcher_s{seed}.pt")); mdl.eval()
        fig, ax = plt.subplots(len(rows), 5, figsize=(20, 4.2 * len(rows)))
        for r, (title, s, k) in enumerate(rows):
            n = d.pair_of[(s, k)]; sid = sids[s]; rgb = np.load(PD / f"{sid}.npz")["rgb"]; base = np.array(json.load(open(PD / f"{sid}.json"))["robot_base_world"]); gt = d.anc[n].cpu().numpy()
            with torch.no_grad(): xd, w2d, p, sc, al = d.fwd_sim(mdl, [n])
            xd = gc.to_m(xd, d)[0].cpu().numpy(); w2d = w2d[0].cpu().numpy(); p = p[0].cpu().numpy(); sc = sc[0].cpu().numpy(); al = al[0].cpu().numpy(); w0 = d.w[n].cpu().numpy(); cv = d.cvalid[n].cpu().numpy(); pm = d.proj[n].float().cpu().numpy()
            fo = (p[:, None] * np.einsum("jt,jtc->jc", al, pm)).sum(0); fo = fo / fo.sum(); cands = P.make(w0.astype(np.float64)); top = int(np.where(cv, sc, -1e9).argmax()); gtb = {c: Bx[sid]["boxes_xyxy_per_camera"][names[s][c]][0] for c in range(len(names[s])) if valid[s, c]}
            a = ax[r, 0]; a.imshow(rgb[0])
            for c, bb in gtb.items(): a.add_patch(plt.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="lime" if c == k else "white", lw=2 if c == k else 1))
            for j, c in enumerate(cands): x0, y0, sd = c["box"]; a.add_patch(plt.Rectangle((x0, y0), sd, sd, fill=False, ec=plt.cm.autumn(1 - float(p[j])), lw=3 if j == top else 1.5, ls="-" if j == top else "--")); a.text(x0 + 3, y0 + 12, f"{j + 1}:{c['src'][0].upper()} p={p[j]:.2f}", color="yellow", fontsize=8)
            x0, y0, sd = cands[top]["box"]; a.plot(*(al[top][:, None] * np.array([[x0 + (t % 8 + .5) * sd / 8, y0 + (t // 8 + .5) * sd / 8] for t in range(64)])).sum(0), "*", color="magenta", ms=18, mec="k")
            a.set_xlim(60, 420); a.set_ylim(380, 120); a.set_xticks([]); a.set_yticks([]); a.set_title(f"seed {seed} | {title}\nquery {names[s][k]}", fontsize=8)
            for col, (m, x_, ttl) in zip((1, 2, 3, 4), ((w0, None, "coarse map (front)"), (fo, None, "fine-only map"), (w2d, xd, "G4d final (front)"), (w2d, xd, "G4d final (right)"))):
                a = ax[r, col]; cam = 1 if col == 4 else 0; a.imshow(rgb[cam]); a.imshow(m[cam * 64:(cam + 1) * 64].reshape(8, 8), extent=(0, 480, 480, 0), cmap="jet", alpha=0.45, interpolation="nearest"); u, v_ = proj(cal[cam], gt + base); a.plot(u, v_, "o", mfc="none", mec="lime", ms=16, mew=2.5)
                if x_ is not None: u, v_ = proj(cal[cam], x_ + base); a.plot(u, v_, "x", color="red", ms=12, mew=3)
                if cam == 0: a.set_xlim(60, 420); a.set_ylim(380, 120)
                a.set_xticks([]); a.set_yticks([]); a.set_title(ttl + (f" | err {np.linalg.norm((x_ - gt)[:2]) * 100:.1f} cm" if x_ is not None and col == 3 else ""), fontsize=8)
        fig.tight_layout(); fig.savefig(OD / f"g4d_s{seed}_sim_visual.png", dpi=45); plt.close(fig)
        fig, ax = plt.subplots(len(pick), 4, figsize=(16, 4.0 * len(pick)))
        for r, (fr, i) in enumerate(pick):
            m = meta[i]; item = dset[int(from_idx[int(m["episode"])]) + int(m["image_frame"])]; img = item[cam0].permute(1, 2, 0).numpy(); img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img; sc_ = img.shape[0] / 480; gt = d.d["dheld"]["anc"][i].cpu().numpy(); out = []
            with torch.no_grad():
                for t in range(3): x, w2, *_ = d.fwd_data(mdl, "dheld", [i], t); out.append((gc.to_m(x, d)[0].cpu().numpy(), w2[0].cpu().numpy()))
            a = ax[r, 0]; a.imshow(img); a.set_xticks([]); a.set_yticks([]); u, v_ = proj(cal[0], gt + BASE); a.plot(u * sc_, v_ * sc_, "o", mfc="none", mec="lime", ms=14, mew=2.5)
            for (x, _), c, lab in zip(out, ("red", "orange", "cyan"), ("true", "swap1", "swap2")): u, v_ = proj(cal[0], x + BASE); a.plot(u * sc_, v_ * sc_, "x", color=c, ms=11, mew=3, label=lab)
            a.set_title(f"seed {seed} | {fr} ep{m['episode']} | true: {m['texts'][0].split('poker ')[-1]} (green ring = GT)", fontsize=8); a.legend(fontsize=6, loc="lower right")
            for t in range(3):
                a = ax[r, 1 + t]; a.imshow(img); a.imshow(out[t][1][:64].reshape(8, 8), extent=(0, img.shape[1], img.shape[0], 0), cmap="jet", alpha=0.45, interpolation="nearest"); u, v_ = proj(cal[0], out[t][0] + BASE); a.plot(u * sc_, v_ * sc_, "x", color="white", ms=11, mew=3); a.set_xticks([]); a.set_yticks([])
                a.set_title(f"{'TRUE' if t == 0 else 'SWAP'} text: {m['texts'][t].split('poker ')[-1]}" + ("" if t == 0 else f" | shift {np.linalg.norm((out[t][0] - out[0][0])[:2]) * 100:.1f} cm"), fontsize=8)
        fig.tight_layout(); fig.savefig(OD / f"g4d_s{seed}_dataset_swap_visual.png", dpi=45); plt.close(fig)
    print("saved")


if __name__ == "__main__":
    main()
