#!/usr/bin/env python
"""G4b visual verification: 4 test scenes x first 2 valid queries (no cherry-picking). Columns: front coarse map + front top-2 proposals (orange) + right->front projected proposals (cyan); right coarse map + right top-2 peaks (centroids marked);
the final deduplicated FRONT crops (thumbnails with fine p, source, #cards, contains-target); final map w' (front) with predicted (red x) vs GT (green ring) anchor; w' (right). Also prints the coarse->fine flip statistics."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g4b_matcher_train as g  # noqa: E402
import g4b_proposals as P  # noqa: E402
from g1_train import D  # noqa: E402
from g3_identity_diagnostic import rs  # noqa: E402
from g4_build_cache import nms_peaks  # noqa: E402

PD = D / "select_poker"; OUT = g.OUT


def proj(cam, xyz):
    R = np.array(cam["xmat"]).reshape(3, 3); p = R.T @ (np.asarray(xyz) - np.array(cam["xpos"])); f = 240 / np.tan(np.deg2rad(cam["fovy"]) / 2); return f * p[0] / -p[2] + 240, 240 - f * p[1] / -p[2]


def main():
    data = g.Data(); model = g.G4bModel().to("cuda"); model.load_state_dict(torch.load(OUT / "g4b_matcher_s0.pt")); model.eval(); cal = json.load(open(D / "camera_calibration.json"))["select_poker"]["cams_policy_order"]
    names = data.names; sids = data.sim["sids"]; split = data.split; valid = data.sim["valid"].numpy(); held = data.sim["held"].numpy(); Bx = {}
    for p in sorted(PD.glob("bboxes_*.json")): Bx.update(json.load(open(p)))
    def pick(sp, cond, skip=()):
        for s in np.where(split == sp)[0]:
            ks = [k for k in range(4) if valid[s, k]]
            if s not in skip and cond(s, ks): return s
    easy = pick("test", lambda s, ks: len(ks) == 3); hard = pick("test", lambda s, ks: len(ks) == 4, (easy,))
    same = pick("test", lambda s, ks: len(ks) == 4 and any(rs(names[s][a])[0] == rs(names[s][b])[0] or rs(names[s][a])[1] == rs(names[s][b])[1] for a in ks for b in ks if a < b), (easy, hard))
    S = {"easy 3-card": easy, "hard 4-card": hard, "same-rank/suit distractor": same, "held-out rank 9": pick("test_heldout_identity", lambda s, ks: len(ks) == 4 and any(held[s, k] for k in ks))}; rows = []
    for title, s in S.items():
        ks = [k for k in range(4) if valid[s, k]]; qs = [k for k in ks if (title == "held-out rank 9") == bool(held[s, k])][:2] or ks[:2]
        for k in qs: rows.append((title, s, k))
    fig, ax = plt.subplots(len(rows), 8, figsize=(30, 3.8 * len(rows)))
    for r, (title, s, k) in enumerate(rows):
        n = data.pair_of[(s, k)]; sid = sids[s]; rgb = np.load(PD / f"{sid}.npz")["rgb"]; base = np.array(json.load(open(PD / f"{sid}.json"))["robot_base_world"])
        with torch.no_grad(): xyz, w2, p, sc = data.fwd_sim(model, [n])
        xyz = g.to_m(xyz, data)[0].cpu().numpy(); w2 = w2[0].cpu().numpy(); p = p[0].cpu().numpy(); sc = sc[0].cpu().numpy(); w0 = data.w[n].cpu().numpy(); gt = data.anc[n].cpu().numpy(); err = np.linalg.norm((xyz - gt)[:2]) * 100
        cands = P.make(w0.astype(np.float64)); cv = data.cvalid[n].cpu().numpy(); pur = {kk: v[n].cpu().numpy() for kk, v in data.pur.items()}; mf, mr = w0[:64].reshape(8, 8), w0[64:128].reshape(8, 8)
        a = ax[r, 0]; a.imshow(rgb[0]); a.imshow(mf, extent=(0, 480, 480, 0), cmap="jet", alpha=0.4, interpolation="nearest")
        for c in range(len(names[s])):
            if valid[s, c]: bb = Bx[sid]["boxes_xyxy_per_camera"][names[s][c]][0]; a.add_patch(plt.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="lime" if c == k else "white", lw=2 if c == k else 1))
        for j, c in enumerate(cands): x0, y0, sd = c["box"]; a.add_patch(plt.Rectangle((x0, y0), sd, sd, fill=False, ec="orange" if c["src"] == "front" else "cyan", lw=2, ls="--")); a.text(x0, y0 - 3, f"{j + 1}:{c['src'][0].upper()}{c['rank'] + 1} p={p[j]:.2f}", color="yellow", fontsize=8)
        a.set_xlim(60, 420); a.set_ylim(380, 120); a.set_xticks([]); a.set_yticks([]); a.set_title(f"{title}\nquery {names[s][k]} | FRONT coarse map; orange=front-proposed, cyan=right->front", fontsize=8)
        a = ax[r, 1]; a.imshow(rgb[1]); a.imshow(mr, extent=(0, 480, 480, 0), cmap="jet", alpha=0.4, interpolation="nearest")
        for q, c in enumerate(nms_peaks(mr, 2)): u, v_ = P.centroid(mr, c); a.plot(u, v_, "w+", ms=14, mew=2); a.text(u + 5, v_, f"R{q + 1}", color="yellow", fontsize=9)
        bb = Bx[sid]["boxes_xyxy_per_camera"][names[s][k]][1]; a.add_patch(plt.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="lime", lw=2)); a.set_xticks([]); a.set_yticks([]); a.set_title("RIGHT coarse map (proposal only), centroids", fontsize=8)
        for j in range(4):
            a = ax[r, 2 + j]; a.axis("off")
            if cv[j]: x0, y0, sd = [int(v) for v in data.sim["box"][s, k, j]] if False else cands[j]["box"]; a.imshow(rgb[0][y0:y0 + sd, x0:x0 + sd]); a.set_title(f"cand {j + 1} ({cands[j]['src']}) fine p={p[j]:.2f}\ncontains target: {bool(pur['contains_target'][j])} | cards {int(pur['n_cards'][j])} | centre-card is target: {bool(pur['target_nearest'][j])}", fontsize=7, color="green" if pur["contains_target"][j] else "black")
        for cam, col in ((0, 6), (1, 7)):
            a = ax[r, col]; a.imshow(rgb[cam]); a.imshow(w2[cam * 64:(cam + 1) * 64].reshape(8, 8), extent=(0, 480, 480, 0), cmap="jet", alpha=0.45, interpolation="nearest"); u, v_ = proj(cal[cam], gt + base); a.plot(u, v_, "o", mfc="none", mec="lime", ms=16, mew=2.5); u, v_ = proj(cal[cam], xyz + base); a.plot(u, v_, "x", color="red", ms=12, mew=3)
            if cam == 0: a.set_xlim(60, 420); a.set_ylim(380, 120)
            a.set_xticks([]); a.set_yticks([]); a.set_title(f"FINAL w' ({'front' if cam == 0 else 'right'}) pred x vs GT ring | err {err:.1f} cm" if cam == 0 else "FINAL w' (right)", fontsize=8)
    fig.tight_layout(); (OUT / "figures").mkdir(exist_ok=True); fig.savefig(OUT / "figures" / "g4b_smoke_visual.png", dpi=50); print("saved", len(rows), "rows")
    # ---- flip statistics (test + heldout) ----
    A = torch.load(OUT / "g4b_matcher_s0_arrays.pt", weights_only=False)
    for sp in ("val", "test", "test_heldout_identity"):
        a = A[sp]; t = torch.as_tensor(a["idx"], device="cuda"); cvv = data.cvalid[t].cpu().numpy(); cont = data.pur["contains_target"][t].cpu().numpy() > .5; nbr = data.nbr[t].cpu().numpy().astype(np.float32); cm = (data.w[t].cpu().numpy()[:, None, :] * nbr).sum(-1)
        top_c = np.where(cvv, cm, -1e9).argmax(1); top_f = np.where(cvv, a["s"], -1e9).argmax(1); r_ = np.arange(len(cm)); c_ok = cont[r_, top_c]; f_ok = cont[r_, top_f]
        print(f"{sp}: coarse-top1 crop has target {c_ok.mean():.3f} | fine-top1 {f_ok.mean():.3f} | flips wrong->right {(~c_ok & f_ok).mean():.3f} ({int((~c_ok & f_ok).sum())}) right->wrong {(c_ok & ~f_ok).mean():.3f} ({int((c_ok & ~f_ok).sum())}); of coarse-wrong queries (n={int((~c_ok).sum())}) fine fixes {((~c_ok & f_ok).sum() / max(1, (~c_ok).sum())):.3f}")


if __name__ == "__main__":
    main()
