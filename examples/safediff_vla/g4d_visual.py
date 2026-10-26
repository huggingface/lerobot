#!/usr/bin/env python
"""G4d visual verification: the SAME 5 queries as the G4c figure. Columns: [front crops (colour=G4d fine p, solid=selected) + GT card boxes (lime=target, diagnostic) + magenta * alpha-weighted fine location] [coarse map w (front)]
[fine-only map norm(sum_j p_j M_j) (G4d)] [G4c final map (multiplicative) + G4c predicted anchor x] [G4d final map (0.8*fine + 0.2*coarse) + G4d predicted anchor x] [G4d final map (right)]. green ring = GT anchor.
Also prints the recovery statistic: queries whose fine-selected crop contains the target but whose COARSE argmax card is wrong -> fraction with correct final-map card under G4c vs G4d."""
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

PD = D / "select_poker"; OD = Path("outputs/eval/safediff_vla_g4d_offline")


def proj(cam, xyz):
    R = np.array(cam["xmat"]).reshape(3, 3); p = R.T @ (np.asarray(xyz) - np.array(cam["xpos"])); f = 240 / np.tan(np.deg2rad(cam["fovy"]) / 2); return f * p[0] / -p[2] + 240, 240 - f * p[1] / -p[2]


def main():
    d = gc.Data(); mc = gc.G4cModel("mult").to("cuda"); mc.load_state_dict(torch.load(gc.OUTC / "g4c_matcher_s0.pt")); mc.eval(); md = gc.G4cModel("add", 0.2).to("cuda"); md.load_state_dict(torch.load(OD / "g4d_matcher_s0.pt")); md.eval()
    cal = json.load(open(D / "camera_calibration.json"))["select_poker"]["cams_policy_order"]; names = d.names; sids = d.sim["sids"]; split = d.split; valid = d.sim["valid"].numpy(); held = d.sim["held"].numpy(); Bx = {}
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
    for title, sp, cond in cats: x = find(sp, cond, used); used.add(x); rows.append((title, *x))
    fig, ax = plt.subplots(len(rows), 6, figsize=(24, 4.2 * len(rows)))
    for r, (title, s, k) in enumerate(rows):
        n = d.pair_of[(s, k)]; sid = sids[s]; rgb = np.load(PD / f"{sid}.npz")["rgb"]; base = np.array(json.load(open(PD / f"{sid}.json"))["robot_base_world"]); gt = d.anc[n].cpu().numpy()
        with torch.no_grad(): xc, w2c, *_ = d.fwd_sim(mc, [n]); xd, w2d, p, sc, al = d.fwd_sim(md, [n])
        xc = gc.to_m(xc, d)[0].cpu().numpy(); xd = gc.to_m(xd, d)[0].cpu().numpy(); w2c = w2c[0].cpu().numpy(); w2d = w2d[0].cpu().numpy(); p = p[0].cpu().numpy(); sc = sc[0].cpu().numpy(); al = al[0].cpu().numpy(); w0 = d.w[n].cpu().numpy()
        cv = d.cvalid[n].cpu().numpy(); pm = d.proj[n].float().cpu().numpy(); fo = (p[:, None] * np.einsum("jt,jtc->jc", al, pm)).sum(0); fo = fo / fo.sum(); cands = P.make(w0.astype(np.float64)); top = int(np.where(cv, sc, -1e9).argmax())
        gtb = {c: Bx[sid]["boxes_xyxy_per_camera"][names[s][c]][0] for c in range(len(names[s])) if valid[s, c]}
        a = ax[r, 0]; a.imshow(rgb[0])
        for c, bb in gtb.items(): a.add_patch(plt.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="lime" if c == k else "white", lw=2 if c == k else 1))
        for j, c in enumerate(cands): x0, y0, sd = c["box"]; a.add_patch(plt.Rectangle((x0, y0), sd, sd, fill=False, ec=plt.cm.autumn(1 - float(p[j])), lw=3 if j == top else 1.5, ls="-" if j == top else "--")); a.text(x0 + 3, y0 + 12, f"{j + 1}:{c['src'][0].upper()} p={p[j]:.2f}", color="yellow", fontsize=8)
        x0, y0, sd = cands[top]["box"]; a.plot(*(al[top][:, None] * np.array([[x0 + (t % 8 + .5) * sd / 8, y0 + (t // 8 + .5) * sd / 8] for t in range(64)])).sum(0), "*", color="magenta", ms=18, mec="k")
        a.set_xlim(60, 420); a.set_ylim(380, 120); a.set_xticks([]); a.set_yticks([]); a.set_title(f"{title}\nquery {names[s][k]} | G4d crops (solid=selected), magenta *=fine location", fontsize=8)
        for col, (m, x_, ttl) in zip((1, 2, 3, 4, 5), ((w0, None, "coarse map w (front)"), (fo, None, "fine-only map (G4d)"), (w2c, xc, "G4c final (multiplicative)"), (w2d, xd, "G4d final (0.8 fine + 0.2 coarse)"), (w2d, xd, "G4d final (right)"))):
            a = ax[r, col]; cam = 1 if col == 5 else 0; a.imshow(rgb[cam]); a.imshow(m[cam * 64:(cam + 1) * 64].reshape(8, 8), extent=(0, 480, 480, 0), cmap="jet", alpha=0.45, interpolation="nearest"); u, v_ = proj(cal[cam], gt + base); a.plot(u, v_, "o", mfc="none", mec="lime", ms=16, mew=2.5)
            if x_ is not None: u, v_ = proj(cal[cam], x_ + base); a.plot(u, v_, "x", color="red", ms=12, mew=3)
            if cam == 0: a.set_xlim(60, 420); a.set_ylim(380, 120)
            a.set_xticks([]); a.set_yticks([]); a.set_title(ttl + (f" | err {np.linalg.norm((x_ - gt)[:2]) * 100:.1f} cm" if x_ is not None else ""), fontsize=8)
    fig.tight_layout(); fig.savefig(OD / "g4d_smoke_visual.png", dpi=50)
    cf = d.sim["cellfrac"].numpy()
    for sp in ("val", "test", "test_heldout_identity"):
        ar_c = torch.load(gc.OUTC / "g4c_matcher_s0_arrays.pt", weights_only=False)[sp]; ar_d = torch.load(OD / "g4d_matcher_s0_arrays.pt", weights_only=False)[sp]; idx = ar_d["idx"]; t = torch.as_tensor(idx, device="cuda"); cv = d.cvalid[t].cpu().numpy(); cont = d.pur["contains_target"][t].cpu().numpy() > .5; k = d.pairs[idx, 1]
        def ok(m): return np.array([max([c for c in range(4) if valid[d.pairs[i, 0], c]], key=lambda c: float((m[nn] * cf[d.pairs[i, 0], c]).sum())) == k[nn] for nn, i in enumerate(idx)])
        c0 = ok(d.w[t].cpu().numpy()); cc = ok(ar_c["w2"]); cd = ok(ar_d["w2"]); sel = cont[np.arange(len(idx)), np.where(cv, ar_d["s"], -1e9).argmax(1)]; m = sel & ~c0
        print(f"{sp}: fine-selected crop has target & coarse argmax card WRONG: n={int(m.sum())} -> final map correct G4c {cc[m].mean():.3f} | G4d {cd[m].mean():.3f}; (coarse-correct n={int((sel & c0).sum())}: G4c {cc[sel & c0].mean():.3f} G4d {cd[sel & c0].mean():.3f})")


if __name__ == "__main__":
    main()
