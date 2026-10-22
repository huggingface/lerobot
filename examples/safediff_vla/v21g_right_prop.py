"""[v2.1 anchor phase] RIGHT-camera class-agnostic proposal head (needed only for calibrated triangulation; the FRONT head stays frozen). Same recipe as the frozen front head:
Head(grid 64), sigma 1.5 cells, focal+offset loss, seed 1000, 4000 steps, batch 16, AdamW 1e-3/1e-4, final step, decode thr 0.10, K cap 10. Train = expansion TRAIN + block A; dev = B; test = C (not used for any choice)."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import cv2, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import v21hr_candidate as H  # noqa: E402
import v21q_prop_iter as Q  # noqa: E402
from v21_common import OUT as V21, load_policy  # noqa: E402
from v21g_geometry import CAMS, project  # noqa: E402

X = V21 / "x"; G, THR = 64, 0.10; TASKS = H.TASKS


def centers_cam(r, ci): return np.array([project(CAMS[ci], c["obj_center"] + r["base"])[:2] for c in r["candidates"] if c["valid"]])


@torch.no_grad()
def feats_cam(pol, rows, ci):
    vm = pol.backbone.model.vlm_with_expert.get_vlm_model().vision_model; out = {}
    for r in rows:
        x = torch.from_numpy(cv2.resize(r["imgs480"][ci], (512, 512), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).float().div(255)[None].cuda() * 2 - 1
        out[r["scene_id"]] = vm(pixel_values=x.to(vm.dtype)).last_hidden_state[0].float().reshape(32, 32, -1).permute(2, 0, 1).half().cpu()
    return out


def targets_cam(r, ci, g=G):
    px, sig = 480 / g, 1.5; hm = np.zeros((g, g), np.float32); off = np.zeros((2, g, g), np.float32); msk = np.zeros((g, g), np.float32); yy, xx = np.mgrid[0:g, 0:g]
    for u, v in centers_cam(r, ci):
        cu, cv_ = u / px, v / px; hm = np.maximum(hm, np.exp(-((xx + 0.5 - cu) ** 2 + (yy + 0.5 - cv_) ** 2) / (2 * sig ** 2))); iu, iv = int(cu), int(cv_)
        if 0 <= iu < g and 0 <= iv < g: hm[iv, iu] = 1.0; off[:, iv, iu] = (cu - iu - 0.5, cv_ - iv - 0.5); msk[iv, iu] = 1
    return hm, off, msk


def cov_eval(outs, rows, ci):
    res = {}
    for t in TASKS:
        cov, err, dup, tot, fp = [], [], 0, 0, 0
        for r in [r for r in rows if r["task"] == t]:
            Gc = centers_cam(r, ci); Pp = Q.decode(outs[r["scene_id"]], G, THR); used = {}
            for p in Pp:
                d = [np.hypot(p[0] - q[0], p[1] - q[1]) for q in Gc]; j = int(np.argmin(d))
                if d[j] <= Q.MATCH_PX: used[j] = used.get(j, 0) + 1; err.append(d[j])
                else: fp += 1
            cov += [float(j in used) for j in range(len(Gc))]; dup += sum(v - 1 for v in used.values()); tot += len(Pp)
        res[t] = {"coverage": float(np.mean(cov)), "duplicate_rate": dup / max(tot, 1), "false_pos_rate": fp / max(tot, 1), "centre_err_px": float(np.mean(err)), "n_gt": len(cov)}
    return res


def main():
    pol, _ = load_policy(); xr = pickle.load(open(X / "referent_table_x.pkl", "rb")); split = json.load(open(X / "split_x.json")); A, B, C = Q.build_rows("A"), Q.build_rows("B"), Q.build_rows("C")
    tr = [r for r in xr if split[r["scene_id"]] == "train" and r["scene_ok"]] + [r for r in A if r["scene_ok"]]; ev = {"dev": [r for r in B if r["scene_ok"]], "test": [r for r in C if r["scene_ok"]], "expansion_val_secondary": [r for r in xr if split[r["scene_id"]] == "val" and r["scene_ok"]]}
    f = feats_cam(pol, tr + sum(ev.values(), []), 1); T = {r["scene_id"]: targets_cam(r, 1) for r in tr}; torch.manual_seed(H.SEED); rng = np.random.default_rng(H.SEED); m = Q.Head(G).cuda(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(1, Q.STEPS + 1):
        m.train(); b = [tr[i] for i in rng.choice(len(tr), Q.BATCH, replace=False)]; x = torch.stack([f[r["scene_id"]] for r in b]).float().cuda(); hm, off, mk = [torch.from_numpy(np.stack([T[r["scene_id"]][k] for r in b])).cuda() for k in range(3)]
        o = m(x); loss = __import__("v21p_proposals").focal(o[:, 0], hm) + F.l1_loss(o[:, 1:] * mk[:, None], off * mk[:, None], reduction="sum") / mk.sum().clamp_min(1); opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0: print("right step", step, float(loss), flush=True)
    torch.save(m.state_dict(), X / "proposal_head_right_g64.pt"); res = {}
    for k, R in ev.items(): res[k] = cov_eval(Q.infer(m, f, R), R, 1); print("RIGHT", k, json.dumps(res[k]), flush=True)
    json.dump(res, open(X / "right_proposal_results.json", "w"), indent=1)

if __name__ == "__main__": main()
