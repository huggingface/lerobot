#!/usr/bin/env python
"""G4c visual verification (test split; held-out rank9 from the held-out split). One query per category (no cherry-picking: first matching query in scene order), categories: easy 3-card, hard 4-card, same-rank distractor, same-suit distractor, held-out rank9.
Per row: [front view: coarse top proposals' crops (colour = fine p), GT card boxes (diagnostic: lime=target) and the projected fine location = alpha-weighted pixel centre of the fine-selected crop (magenta star)] [top-2 crops by fine score with GT card boxes (diagnostic overlay) + intra-crop alpha heat]
[FINAL global map w' (front) with predicted anchor x vs GT ring] [diagnostic only: fine-only map sum_j p_j M_j (front)] [FINAL w' (right)]."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g4c_matcher_train as gc  # noqa: E402
import g4b_proposals as P  # noqa: E402
from g1_train import D  # noqa: E402
from g3_identity_diagnostic import rs  # noqa: E402

PD = D / "select_poker"; OC = gc.OUTC


def proj(cam, xyz):
    R = np.array(cam["xmat"]).reshape(3, 3); p = R.T @ (np.asarray(xyz) - np.array(cam["xpos"])); f = 240 / np.tan(np.deg2rad(cam["fovy"]) / 2); return f * p[0] / -p[2] + 240, 240 - f * p[1] / -p[2]


def main():
    d = gc.Data(); model = gc.G4cModel().to("cuda"); model.load_state_dict(torch.load(OC / "g4c_matcher_s0.pt")); model.eval(); cal = json.load(open(D / "camera_calibration.json"))["select_poker"]["cams_policy_order"]
    names = d.names; sids = d.sim["sids"]; split = d.split; valid = d.sim["valid"].numpy(); held = d.sim["held"].numpy(); Bx = {}
    for p in sorted(PD.glob("bboxes_*.json")): Bx.update(json.load(open(p)))
    def find(sp, cond, used):
        for s in np.where(split == sp)[0]:
            ks = [k for k in range(4) if valid[s, k]]
            for k in ks:
                if (s, k) not in used and cond(s, k, ks): return (s, k)
    used = set(); rows = []
    cats = [("easy 3-card", "test", lambda s, k, ks: len(ks) == 3 and not held[s, k]), ("hard 4-card", "test", lambda s, k, ks: len(ks) == 4 and not held[s, k]),
            ("same-rank distractor", "test", lambda s, k, ks: not held[s, k] and any(rs(names[s][c])[0] == rs(names[s][k])[0] for c in ks if c != k) and not any(rs(names[s][c])[1] == rs(names[s][k])[1] for c in ks if c != k)),
            ("same-suit distractor", "test", lambda s, k, ks: not held[s, k] and any(rs(names[s][c])[1] == rs(names[s][k])[1] for c in ks if c != k) and not any(rs(names[s][c])[0] == rs(names[s][k])[0] for c in ks if c != k)),
            ("held-out rank 9", "test_heldout_identity", lambda s, k, ks: held[s, k] and len(ks) == 4)]
    for title, sp, cond in cats:
        x = find(sp, cond, used); used.add(x); rows.append((title, *x))
    fig, ax = plt.subplots(len(rows), 6, figsize=(24, 4.2 * len(rows)))
    for r, (title, s, k) in enumerate(rows):
        n = d.pair_of[(s, k)]; sid = sids[s]; rgb = np.load(PD / f"{sid}.npz")["rgb"]; base = np.array(json.load(open(PD / f"{sid}.json"))["robot_base_world"])
        with torch.no_grad(): xyz, w2, p, sc, al = d.fwd_sim(model, [n])
        xyz = gc.to_m(xyz, d)[0].cpu().numpy(); w2 = w2[0].cpu().numpy(); p = p[0].cpu().numpy(); sc = sc[0].cpu().numpy(); al = al[0].cpu().numpy(); gt = d.anc[n].cpu().numpy(); err = np.linalg.norm((xyz - gt)[:2]) * 100
        cv = d.cvalid[n].cpu().numpy(); pm = d.proj[n].float().cpu().numpy(); fine_only = (p[:, None] * np.einsum("jt,jtc->jc", al, pm)).sum(0); box = d.sim["box"] if False else None
        cands = P.make(d.w[n].cpu().numpy().astype(np.float64)); top = int(np.where(cv, sc, -1e9).argmax()); gtb = {c: Bx[sid]["boxes_xyxy_per_camera"][names[s][c]][0] for c in range(len(names[s])) if valid[s, c]}
        a = ax[r, 0]; a.imshow(rgb[0])
        for c, bb in gtb.items(): a.add_patch(plt.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="lime" if c == k else "white", lw=2 if c == k else 1))
        for j, c in enumerate(cands): x0, y0, sd = c["box"]; a.add_patch(plt.Rectangle((x0, y0), sd, sd, fill=False, ec=plt.cm.autumn(1 - float(p[j])), lw=3 if j == top else 1.5, ls="-" if j == top else "--")); a.text(x0 + 3, y0 + 12, f"{j + 1}:{c['src'][0].upper()} p={p[j]:.2f}", color="yellow", fontsize=8)
        x0, y0, sd = cands[top]["box"]; tt = (al[top][:, None] * np.array([[x0 + (t % 8 + .5) * sd / 8, y0 + (t // 8 + .5) * sd / 8] for t in range(64)])).sum(0); a.plot(*tt, "*", color="magenta", ms=18, mec="k")
        a.set_xlim(60, 420); a.set_ylim(380, 120); a.set_xticks([]); a.set_yticks([]); a.set_title(f"{title}\nquery {names[s][k]} | crops (colour=fine p; solid=selected), magenta *=alpha-weighted fine location", fontsize=8)
        order = [top] + [j for j in np.argsort(-np.where(cv, sc, -1e9)) if j != top and cv[j]]
        for q in range(2):
            a = ax[r, 1 + q]; a.axis("off")
            if q < len(order):
                j = order[q]; x0, y0, sd = cands[j]["box"]; a.imshow(rgb[0][y0:y0 + sd, x0:x0 + sd], extent=(0, 8, 8, 0)); heat = al[j].reshape(8, 8); a.imshow(heat, extent=(0, 8, 8, 0), cmap="jet", alpha=0.45, interpolation="nearest")
                for c, bb in gtb.items(): a.add_patch(plt.Rectangle(((bb[0] - x0) / sd * 8, (bb[1] - y0) / sd * 8), (bb[2] - bb[0]) / sd * 8, (bb[3] - bb[1]) / sd * 8, fill=False, ec="lime" if c == k else "white", lw=2 if c == k else 1))
                a.set_xlim(0, 8); a.set_ylim(8, 0); a.set_title(f"{'SELECTED' if q == 0 else 'runner-up'} crop {j + 1} fine p={p[j]:.2f}\nalpha on target card tokens {float((al[j] * (d.tgt_tok[n, j].cpu().numpy() >= .25)).sum()):.2f} (cards GT overlay: lime=target)", fontsize=8)
        for col, (m, ttl) in zip((3, 4, 5), ((w2, "FINAL global map w' (front)"), (fine_only, "diagnostic: fine-only map (front)"), (w2, "FINAL w' (right)"))):
            a = ax[r, col]; cam = 1 if col == 5 else 0; a.imshow(rgb[cam]); a.imshow(m[cam * 64:(cam + 1) * 64].reshape(8, 8), extent=(0, 480, 480, 0), cmap="jet", alpha=0.45, interpolation="nearest"); u, v_ = proj(cal[cam], gt + base); a.plot(u, v_, "o", mfc="none", mec="lime", ms=16, mew=2.5)
            if col != 4: u, v_ = proj(cal[cam], xyz + base); a.plot(u, v_, "x", color="red", ms=12, mew=3)
            if cam == 0: a.set_xlim(60, 420); a.set_ylim(380, 120)
            a.set_xticks([]); a.set_yticks([]); a.set_title(ttl + (f" | pred x vs GT ring, err {err:.1f} cm" if col == 3 else ""), fontsize=8)
    fig.tight_layout(); (OC / "figures").mkdir(exist_ok=True); fig.savefig(OC / "figures" / "g4c_smoke_visual.png", dpi=50); print("saved", [(t, names[s][k]) for t, s, k in rows])


if __name__ == "__main__":
    main()
