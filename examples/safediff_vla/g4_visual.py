#!/usr/bin/env python
"""G4 smoke visual verification: per scene/query -> coarse map + candidate boxes, the 4 crop thumbnails with purity info and fine probability, the FINAL spatial weights w',
predicted anchor (red x) vs GT anchor (green ring). GT is used only for the overlay/diagnostics. Queries: the first two valid queries of each chosen scene (no cherry-picking)."""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g4_matcher_train as g  # noqa: E402
from g1_train import D  # noqa: E402
from g3_identity_diagnostic import rs  # noqa: E402
from g4_build_cache import crop_geom  # noqa: E402

PD = D / "select_poker"


def proj(cam, xyz):
    R = np.array(cam["xmat"]).reshape(3, 3); p = R.T @ (np.asarray(xyz) - np.array(cam["xpos"])); f = 240 / np.tan(np.deg2rad(cam["fovy"]) / 2); return f * p[0] / -p[2] + 240, 240 - f * p[1] / -p[2]


def main():
    data = g.Data(); model = g.G4Model().to("cuda"); model.load_state_dict(torch.load(g.OUT / "g4_matcher_s0.pt")); model.eval(); cal = json.load(open(D / "camera_calibration.json"))["select_poker"]["cams_policy_order"]
    names = data.names; sids = data.sim["sids"]; split = data.split; valid = data.sim["valid"].numpy(); held = data.sim["held"].numpy(); cache = {}
    def pick(sp, cond):
        for s in np.where(split == sp)[0]:
            ks = [k for k in range(4) if valid[s, k]]
            if cond(s, ks): return s
    S = {"easy 3-card": pick("test", lambda s, ks: len(ks) == 3),
         "hard 4-card": pick("test", lambda s, ks: len(ks) == 4),
         "same-rank/suit distractor": pick("test", lambda s, ks: len(ks) == 4 and any(rs(names[s][a])[0] == rs(names[s][b])[0] or rs(names[s][a])[1] == rs(names[s][b])[1] for a in ks for b in ks if a < b) and s != pick("test", lambda s2, k2: len(k2) == 4)),
         "held-out rank 9": pick("test_heldout_identity", lambda s, ks: len(ks) == 4 and any(held[s, k] for k in ks))}
    Bx = {}
    for p in sorted(PD.glob("bboxes_*.json")): Bx.update(json.load(open(p)))
    rows = []
    for title, s in S.items():
        ks = [k for k in range(4) if valid[s, k]]; qs = [k for k in ks if (title == "held-out rank 9") == bool(held[s, k])][:2] or ks[:2]
        for k in qs: rows.append((title, s, k))
    fig, ax = plt.subplots(len(rows), 7, figsize=(26, 3.7 * len(rows)))
    for r, (title, s, k) in enumerate(rows):
        n = data.pair_of[(s, k)]; sid = sids[s]; z = np.load(PD / f"{sid}.npz"); rgb = z["rgb"]
        with torch.no_grad(): xyz, w2, p, sc = data.fwd_sim(model, [n])
        xyz = g.to_m(xyz, data)[0].cpu().numpy(); w2 = w2[0].cpu().numpy(); p = p[0].cpu().numpy(); sc = sc[0].cpu().numpy(); w0 = data.w[n].cpu().numpy(); cells = data.cell[n].cpu().numpy(); base = np.array(json.load(open(PD / f"{sid}.json"))["robot_base_world"])
        pur = {kk: v[n].cpu().numpy() for kk, v in data.pur.items()}; gt = data.anc[n].cpu().numpy(); err = np.linalg.norm((xyz - gt)[:2]) * 100
        a = ax[r, 0]; a.imshow(rgb[0]); a.imshow(w0[:64].reshape(8, 8), extent=(0, 480, 480, 0), cmap="jet", alpha=0.4, interpolation="nearest")
        for c in range(len(names[s])):
            if valid[s, c]: bb = Bx[sid]["boxes_xyxy_per_camera"][names[s][c]][0]; a.add_patch(plt.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="lime" if c == k else "white", lw=2 if c == k else 1))
        for j in range(2):
            x0, y0, sd = crop_geom(int(cells[j]), 0, 480); a.add_patch(plt.Rectangle((x0, y0), sd, sd, fill=False, ec="red" if p[j] == p[:2].max() else "orange", lw=2, ls="--")); a.text(x0, y0 - 3, f"F{j + 1} p={p[j]:.2f}", color="yellow", fontsize=8)
        a.set_xlim(100, 380); a.set_ylim(340, 150); a.set_xticks([]); a.set_yticks([]); a.set_title(f"{title}\nquery: {names[s][k]}  (coarse map + front candidate boxes)", fontsize=8)
        for j in range(4):
            a = ax[r, 1 + j]; cam = j // 2; x0, y0, sd = crop_geom(int(cells[j]), cam, 480); a.imshow(rgb[cam][y0:y0 + sd, x0:x0 + sd]); a.axis("off")
            cen = "centered" if pur["target_nearest_to_center"][j] else "off-ctr/absent"; a.set_title(f"{'front' if cam == 0 else 'right'} #{j % 2 + 1}  p={p[j]:.2f} (coarse mass {data.mass0[n, j].item():.2f})\ncontains target: {bool(pur['contains_target'][j])} | cards {int(pur['n_cards_inside'][j])} | tgt {pur['target_frac_of_crop'][j]:.2f} dis {pur['distractor_frac_of_crop'][j]:.2f} | {cen}", fontsize=7, color="green" if pur["contains_target"][j] else "black")
        a = ax[r, 5]; a.imshow(rgb[0]); a.imshow(w2[:64].reshape(8, 8), extent=(0, 480, 480, 0), cmap="jet", alpha=0.45, interpolation="nearest"); u, v_ = proj(cal[0], gt + base); a.plot(u, v_, "o", mfc="none", mec="lime", ms=16, mew=2.5); u, v_ = proj(cal[0], xyz + base); a.plot(u, v_, "x", color="red", ms=12, mew=3)
        a.set_xlim(100, 380); a.set_ylim(340, 150); a.set_xticks([]); a.set_yticks([]); a.set_title(f"FINAL w' (front), pred x vs GT ring | err {err:.1f} cm", fontsize=8)
        a = ax[r, 6]; a.imshow(rgb[1]); a.imshow(w2[64:128].reshape(8, 8), extent=(0, 480, 480, 0), cmap="jet", alpha=0.45, interpolation="nearest"); u, v_ = proj(cal[1], gt + base); a.plot(u, v_, "o", mfc="none", mec="lime", ms=16, mew=2.5); u, v_ = proj(cal[1], xyz + base); a.plot(u, v_, "x", color="red", ms=12, mew=3)
        a.set_xticks([]); a.set_yticks([]); a.set_title("FINAL w' (right camera)", fontsize=8)
    fig.tight_layout(); (g.OUT / "figures").mkdir(exist_ok=True); fig.savefig(g.OUT / "figures" / "g4_smoke_visual.png", dpi=55); print("saved", len(rows), "rows")


if __name__ == "__main__":
    main()
