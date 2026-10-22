#!/usr/bin/env python
"""[v2.1HR] Deployable-proposal study (poker/mahjong, source-domain only; NO simulator candidate coordinates at inference).
 stage 1  GT-geometry AUDIT (candidate counts, object size in px, nearest-neighbour spacing, 8x8-token cell collisions) -> required proposal recall / K.
 stage 2  MINIMAL class-agnostic proposal baseline: frozen SigLIP patch features of the FULL-RES front image (480->512, 32x32 patches of ~15 px) -> 2 small convs ->
          centre heat-map (CenterNet focal loss, sigma 1 cell) + sub-cell offset. Language-independent, identity-free (every object is the same class). Train: expansion TRAIN scenes only.
          Decode: 3x3 max-pool NMS, score>=0.25 (fixed a priori), top-K. Metrics on expansion VAL: recall@K, centre error, duplicate rate, missed-object rate.
 stage 3  COMPOSITION: learned proposals -> 72 px crops (same extractor) -> the frozen-trained HR scorer (scorer_expanded.pt) -> referent. GT-proposal vs learned-proposal referent accuracy.
Fixed recipe (not tuned on any eval seed): seed 1000, 3000 steps, batch 16, AdamW 1e-3 wd 1e-4, final step. MATCH_PX = 10 (a proposal within 10 px of the GT centre keeps the 72 px crop centred on the object).
Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/v21p_proposals.py"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import v21hr_candidate as H  # noqa: E402
from v21_common import OUT as V21, load_policy, make_batch  # noqa: E402
from v21_referent_table import cam_params, project  # noqa: E402

X = V21 / "x"; TASKS = H.TASKS; PATCH_PX = 480 / 32; MATCH_PX = 10.0; THR = 0.25; KMAX = 8; STEPS, BATCH = 3000, 16


def centers(r):
    cam0 = cam_params("select_poker")[0]; return np.array([project(cam0, c["obj_center"] + r["base"])[:2] for c in r["candidates"] if c["valid"]]), [i for i, c in enumerate(r["candidates"]) if c["valid"]]


def audit(rows):
    cam0 = cam_params("select_poker")[0]; res = {}
    for t in TASKS:
        R = [r for r in rows if r["task"] == t and r["scene_ok"]]; n, nn_px, nn_cm, cell, patch, wpx, hpx = [], [], [], [], [], [], []
        for r in R:
            C = [c for c in r["candidates"] if c["valid"]]; P = np.array([project(cam0, c["obj_center"] + r["base"]) for c in C]); A = np.array([c["obj_center"] for c in C]); n.append(len(C))
            for i, c in enumerate(C):
                u, v, z = P[i]; wpx.append(0.064 * (480 / 2) / np.tan(np.deg2rad(22.5)) / z); hpx.append(0.094 * (480 / 2) / np.tan(np.deg2rad(22.5)) / z)  # 6.4 x 9.4 cm footprint (probed)
                d = np.linalg.norm(P[:, :2] - P[i, :2], axis=1); d[i] = 1e9; nn_px.append(d.min()); da = np.linalg.norm(A - A[i], axis=1); da[i] = 1e9; nn_cm.append(da.min() * 100)
                cell.append(any((int(P[j, 0] // 60), int(P[j, 1] // 60)) == (int(u // 60), int(v // 60)) for j in range(len(C)) if j != i)); patch.append(any((int(P[j, 0] // PATCH_PX), int(P[j, 1] // PATCH_PX)) == (int(u // PATCH_PX), int(v // PATCH_PX)) for j in range(len(C)) if j != i))
        res[t] = {"scenes": len(R), "cands/scene mean": float(np.mean(n)), "cands/scene max": int(max(n)), "object px (w x h) mean": [float(np.mean(wpx)), float(np.mean(hpx))], "NN spacing px mean/p10/min": [float(np.mean(nn_px)), float(np.percentile(nn_px, 10)), float(np.min(nn_px))],
                  "NN spacing cm mean/p10/min": [float(np.mean(nn_cm)), float(np.percentile(nn_cm, 10)), float(np.min(nn_cm))], "frac sharing an 8x8-VLM-token cell": float(np.mean(cell)), "frac sharing a 32x32 patch cell": float(np.mean(patch)), "required K (>= max cands)": int(max(n))}
    return res


@torch.no_grad()
def patch_feats(pol, rows):
    vm = pol.backbone.model.vlm_with_expert.get_vlm_model().vision_model; F_ = {}
    for r in rows:
        if r["task"] not in TASKS or not r["scene_ok"]: continue
        x = torch.from_numpy(cv2.resize(r["imgs480"][0], (512, 512), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).float().div(255)[None].cuda() * 2 - 1
        h = vm(pixel_values=x.to(vm.dtype)).last_hidden_state[0].float(); F_[r["scene_id"]] = h.reshape(32, 32, -1).permute(2, 0, 1).half().cpu()
    return F_


class PropHead(nn.Module):
    def __init__(self, ind=768):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(ind, 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, 3, 1))
    def forward(self, f): return self.net(f)


def targets(r):
    C, _ = centers(r); hm = np.zeros((32, 32), np.float32); off = np.zeros((2, 32, 32), np.float32); msk = np.zeros((32, 32), np.float32); yy, xx = np.mgrid[0:32, 0:32]
    for u, v in C:
        cu, cv_ = u / PATCH_PX, v / PATCH_PX; hm = np.maximum(hm, np.exp(-((xx + 0.5 - cu) ** 2 + (yy + 0.5 - cv_) ** 2) / 2.0)); iu, iv = int(cu), int(cv_)
        if 0 <= iu < 32 and 0 <= iv < 32: hm[iv, iu] = 1.0; off[:, iv, iu] = (cu - iu - 0.5, cv_ - iv - 0.5); msk[iv, iu] = 1
    return hm, off, msk


def focal(logit, hm):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4); pos = hm.eq(1).float(); neg = 1 - pos
    return -((1 - p) ** 2 * torch.log(p) * pos).sum() / pos.sum().clamp_min(1) - ((1 - hm) ** 4 * p ** 2 * torch.log(1 - p) * neg).sum() / pos.sum().clamp_min(1)


def decode(out, thr=THR, k=KMAX):
    heat = torch.sigmoid(out[0]); mx = F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]; pk = (heat == mx) & (heat >= thr); ys, xs = torch.nonzero(pk, as_tuple=True); sc = heat[ys, xs]; o = torch.argsort(-sc)[:k]; P = []
    for j in o.tolist():
        y, x = ys[j].item(), xs[j].item(); P.append(((x + 0.5 + out[1, y, x].item()) * PATCH_PX, (y + 0.5 + out[2, y, x].item()) * PATCH_PX, sc[j].item()))
    return P


def prop_eval(model, rows, feats):
    model.eval(); rec = {1: [], 3: [], 5: [], 8: []}; err, dup, tot, miss, nprop = [], 0, 0, 0, []; per = {}
    with torch.no_grad():
        for r in rows:
            if r["task"] not in TASKS or not r["scene_ok"] or r["scene_id"] not in feats: continue
            G, ids = centers(r); P = decode(model(feats[r["scene_id"]].float().cuda()[None])[0]); nprop.append(len(P)); per[r["scene_id"]] = P
            for K in rec: rec[K] += [bool(any(np.hypot(p[0] - g[0], p[1] - g[1]) <= MATCH_PX for p in P[:K])) for g in G]
            used = {}
            for pi, p in enumerate(P):
                d = [np.hypot(p[0] - g[0], p[1] - g[1]) for g in G]; j = int(np.argmin(d))
                if d[j] <= MATCH_PX: used[j] = used.get(j, 0) + 1; err.append(d[j])
            dup += sum(v - 1 for v in used.values()); tot += len(P); miss += sum(1 for j in range(len(G)) if j not in used)
    n_gt = len(rec[1]); return {"recall@1": float(np.mean(rec[1])), "recall@3": float(np.mean(rec[3])), "recall@5": float(np.mean(rec[5])), "recall@8": float(np.mean(rec[8])), "centre_err_px_mean": float(np.mean(err)), "centre_err_cm_approx": float(np.mean(err) * 0.28), "duplicate_rate": dup / max(tot, 1), "missed_object_rate": miss / n_gt, "mean_proposals": float(np.mean(nprop)), "n_gt": n_gt}, per


def main():
    rows = pickle.load(open(X / "referent_table_x.pkl", "rb")); split = json.load(open(X / "split_x.json")); orows = pickle.load(open(V21 / "referent_table.pkl", "rb")); res = {"audit_expansion": audit(rows), "audit_old_pilot": audit([r for r in orows if r["task"] in TASKS])}
    print(json.dumps(res, indent=1), flush=True)
    pol, pre = load_policy(); feats = patch_feats(pol, rows); tr = [r for r in rows if split[r["scene_id"]] == "train" and r["task"] in TASKS and r["scene_ok"]]; va = [r for r in rows if split[r["scene_id"]] == "val" and r["task"] in TASKS and r["scene_ok"]]
    T = {r["scene_id"]: targets(r) for r in rows if r["task"] in TASKS and r["scene_ok"]}; torch.manual_seed(H.SEED); rng = np.random.default_rng(H.SEED); m = PropHead().cuda(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(1, STEPS + 1):
        m.train(); b = [tr[i] for i in rng.choice(len(tr), BATCH, replace=False)]; f = torch.stack([feats[r["scene_id"]] for r in b]).float().cuda(); hm = torch.from_numpy(np.stack([T[r["scene_id"]][0] for r in b])).cuda(); off = torch.from_numpy(np.stack([T[r["scene_id"]][1] for r in b])).cuda(); mk = torch.from_numpy(np.stack([T[r["scene_id"]][2] for r in b])).cuda()
        out = m(f); loss = focal(out[:, 0], hm) + F.l1_loss(out[:, 1:] * mk[:, None], off * mk[:, None], reduction="sum") / mk.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0: print("prop step", step, float(loss), flush=True)
    res["proposal_val"], per_val = prop_eval(m, va, feats); res["proposal_train"], _ = prop_eval(m, tr[:100], feats); torch.save(m.state_dict(), X / "proposal_head.pt")
    for t in TASKS: res[f"proposal_val_{t}"], _ = prop_eval(m, [r for r in va if r["task"] == t], feats)
    print("proposal val", res["proposal_val"]); print("proposal train(100 scenes)", res["proposal_train"])
    for t in TASKS: print(t, res[f"proposal_val_{t}"])
    # stage 3: composition with the HR scorer trained on GT crops
    sc = H.Scorer("B").cuda(); sc.load_state_dict(torch.load(X / "scorer_expanded.pt")); sc.eval(); ve = pol.backbone.model.vlm_with_expert; import v21x_hr as XH; Tt = XH.text_tokens(pol, pre, va); comp = {t: {"gt_prop_acc": [], "learned_prop_acc": [], "target_covered": [], "n_prop": []} for t in TASKS}
    with torch.no_grad():
        for r in va:
            G, ids = centers(r); P = per_val[r["scene_id"]]
            for ti_pos, ti in enumerate(ids):
                txt = Tt[(r["scene_id"], ti)][None].cuda(); tm = torch.ones(1, txt.shape[1], dtype=torch.bool, device="cuda")
                def crops_from(centres):
                    out = []
                    for (u, v) in centres:
                        u, v = int(round(u)), int(round(v)); h = H.CROP_PX // 2; pad = np.pad(r["imgs480"][0], ((h, h), (h, h), (0, 0)), mode="edge"); crop = pad[min(max(v, 0), 479): min(max(v, 0), 479) + H.CROP_PX, min(max(u, 0), 479): min(max(u, 0), 479) + H.CROP_PX]
                        x = torch.from_numpy(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).float().div(255)[None].cuda() * 2 - 1; out.append(ve.embed_image(x)[0].float())
                    return torch.stack(out)[None]
                lg = sc(txt, tm, crops_from([(g[0], g[1]) for g in G]))[0]; comp[r["task"]]["gt_prop_acc"].append(float(lg.argmax().item() == ti_pos))
                covered = any(np.hypot(p[0] - G[ti_pos][0], p[1] - G[ti_pos][1]) <= MATCH_PX for p in P); comp[r["task"]]["target_covered"].append(float(covered)); comp[r["task"]]["n_prop"].append(len(P))
                if len(P) == 0: comp[r["task"]]["learned_prop_acc"].append(0.0); continue
                lg2 = sc(txt, tm, crops_from([(p[0], p[1]) for p in P]))[0]; pk = P[int(lg2.argmax())]; comp[r["task"]]["learned_prop_acc"].append(float(np.hypot(pk[0] - G[ti_pos][0], pk[1] - G[ti_pos][1]) <= MATCH_PX))
    res["composition"] = {t: {k: float(np.mean(v)) for k, v in d.items()} | {"n": len(d["gt_prop_acc"]), "chance(1/n_gt)": float(np.mean([1 / len(centers(r)[0]) for r in va if r["task"] == t]))} for t, d in comp.items()}; print("composition", json.dumps(res["composition"], indent=1))
    json.dump(res, open(X / "proposal_results.json", "w"), indent=1, default=float)

if __name__ == "__main__": main()
