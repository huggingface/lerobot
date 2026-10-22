#!/usr/bin/env python
"""[v2.1HR] ONE focused proposal-RECALL iteration (class-agnostic, language-independent proposals; the HR referent scorer stays FROZEN = scorer_expanded.pt).
Data (all seeds >= 100050, disjoint from 3000-3019):
  proposal TRAIN : expansion train scenes (200/task) + block A (seeds 100300-100399, 100/task)
  proposal DEV   : block B (seeds 100400-100439, 40/task) -- ALL variant/threshold choices are made here
  proposal TEST  : block C (seeds 100440-100499, 60/task) -- scored ONCE after the choice is frozen
  (the previously reported expansion-val scenes are NOT used for any selection; reported as secondary only)
Variants (fixed before running): V0 = 32x32 grid (15 px cells, sigma 1 cell); V1 = 64x64 grid (7.5 px cells, 2x learned up-sampling, sigma 1.5 cells).
Recipe: seed 1000, 4000 steps, batch 16, AdamW 1e-3 wd 1e-4, final step. Thresholds swept on DEV: {0.05,0.10,0.15,0.25}; K cap = 10 (>= 8).
SELECTION RULE (fixed): highest mahjong coverage on DEV subject to duplicate_rate <= 0.10, proposals/GT <= 1.5 and poker coverage >= 0.98.
Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/v21q_prop_iter.py"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import v21hr_candidate as H  # noqa: E402
import v21p_proposals as P0  # noqa: E402
import v21x_hr as XH  # noqa: E402
from v21_common import OUT as V21, load_policy  # noqa: E402
from v21_referent_table import MIN_SEP_M, cam_params, parse_identity, project  # noqa: E402

X = V21 / "x"; SRC = Path("outputs/stageplan/pilot_prop"); TASKS = H.TASKS; MATCH_PX = 10.0; KCAP = 10; THRS = (0.05, 0.10, 0.15, 0.25); STEPS, BATCH = 4000, 16


def build_rows(block):
    rows, cams = [], cam_params("select_poker")
    for task in TASKS:
        for s in pickle.load(open(SRC / block / f"scenes_{task}.pkl", "rb")):
            assert s["seed"] >= 100300
            cands = []
            for c in s["candidates"]:
                proj = [project(cm, c["anchor"] + s["base"]) for cm in cams]; cands.append({**c, "identity": parse_identity(task, c["name"]), "proj_uv_depth": proj, "in_frame": [bool(p is not None and 8 <= p[0] < 472 and 8 <= p[1] < 472) for p in proj], "is_target": c["name"] == s["target"]})
            A = np.stack([c["anchor"] for c in cands]); D = np.linalg.norm(A[:, None] - A[None], axis=-1) + np.eye(len(cands)) * 9
            for i, c in enumerate(cands): rs = ([] if any(c["in_frame"][:2]) else ["oof"]) + (["amb"] if D[i].min() < MIN_SEP_M else []); c["valid"], c["invalid_reasons"] = not rs, rs
            ck = s["checks"]; ok = bool(ck["target_in_candidates"] and ck["native_matches_live"] and ck["unique_instructions"])
            rows.append({"scene_id": f"{task}_{s['seed']}", "task": task, "seed": s["seed"], "scene_ok": ok, "n_cand": len(cands), "base_instruction": s["base_instruction"], "target": s["target"], "candidates": cands, "state": s["state"], "base": s["base"], "imgs480": s["imgs480"], "imgs224": s["imgs224"]})
    return rows


class Head(nn.Module):
    def __init__(self, grid):
        super().__init__(); self.g = grid
        L = [nn.Conv2d(768, 128, 3, padding=1), nn.GELU()] + ([nn.ConvTranspose2d(128, 128, 2, stride=2), nn.GELU()] if grid == 64 else []) + [nn.Conv2d(128, 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, 3, 1)]; self.net = nn.Sequential(*L)
    def forward(self, f): return self.net(f)


def targets(r, g):
    px = 480 / g; sig = 1.0 if g == 32 else 1.5; C, _ = P0.centers(r); hm = np.zeros((g, g), np.float32); off = np.zeros((2, g, g), np.float32); msk = np.zeros((g, g), np.float32); yy, xx = np.mgrid[0:g, 0:g]
    for u, v in C:
        cu, cv_ = u / px, v / px; hm = np.maximum(hm, np.exp(-((xx + 0.5 - cu) ** 2 + (yy + 0.5 - cv_) ** 2) / (2 * sig ** 2))); iu, iv = int(cu), int(cv_)
        if 0 <= iu < g and 0 <= iv < g: hm[iv, iu] = 1.0; off[:, iv, iu] = (cu - iu - 0.5, cv_ - iv - 0.5); msk[iv, iu] = 1
    return hm, off, msk


def decode(out, g, thr, k=KCAP):
    px = 480 / g; heat = torch.sigmoid(out[0]); mx = F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]; pk = (heat == mx) & (heat >= thr); ys, xs = torch.nonzero(pk, as_tuple=True); sc = heat[ys, xs]; o = torch.argsort(-sc)[:k]
    return [((xs[j].item() + 0.5 + out[1, ys[j], xs[j]].item()) * px, (ys[j].item() + 0.5 + out[2, ys[j], xs[j]].item()) * px, sc[j].item()) for j in o.tolist()]


def evaluate(outs, rows, g, thr):
    """outs[scene_id] = raw head output [3,g,g] (cpu). Returns coverage/proposal stats per task."""
    res = {}
    for t in TASKS:
        cov, err, dup, tot, np_, ngt, fp = [], [], 0, 0, [], [], 0; per = {}
        for r in [r for r in rows if r["task"] == t and r["scene_ok"]]:
            G, _ = P0.centers(r); Pp = decode(outs[r["scene_id"]], g, thr); per[r["scene_id"]] = Pp; used = {}; fpn = 0
            for p in Pp:
                d = [np.hypot(p[0] - q[0], p[1] - q[1]) for q in G]; j = int(np.argmin(d))
                if d[j] <= MATCH_PX: used[j] = used.get(j, 0) + 1; err.append(d[j])
                else: fpn += 1
            cov += [float(j in used) for j in range(len(G))]; dup += sum(v - 1 for v in used.values()); tot += len(Pp); np_.append(len(Pp)); ngt.append(len(G)); fp += fpn
        res[t] = {"coverage": float(np.mean(cov)), "proposals/scene": float(np.mean(np_)), "gt/scene": float(np.mean(ngt)), "proposals/GT": float(np.sum(np_) / np.sum(ngt)), "duplicate_rate": dup / max(tot, 1), "false_pos_rate": fp / max(tot, 1), "centre_err_px": float(np.mean(err)), "centre_err_cm_approx": float(np.mean(err) * 0.28), "n_gt": len(cov), "thr": thr}; res[t]["_per"] = per
    return res


def infer(model, feats, rows):
    model.eval(); o = {}
    with torch.no_grad():
        for r in rows:
            if r["scene_id"] in feats: o[r["scene_id"]] = model(feats[r["scene_id"]].float().cuda()[None])[0].cpu()
    return o


def compose(pol, pre, sc, rows, per, tag):
    ve = pol.backbone.model.vlm_with_expert; Tt = XH.text_tokens(pol, pre, rows); out = {t: {"gt_prop_acc": [], "learned_prop_acc": [], "target_covered": []} for t in TASKS}
    def crops(r, cs):
        L = []
        for (u, v) in cs:
            u, v = int(round(min(max(u, 0), 479))), int(round(min(max(v, 0), 479))); h = H.CROP_PX // 2; pad = np.pad(r["imgs480"][0], ((h, h), (h, h), (0, 0)), mode="edge"); crop = pad[v: v + H.CROP_PX, u: u + H.CROP_PX]
            L.append(ve.embed_image(torch.from_numpy(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).float().div(255)[None].cuda() * 2 - 1)[0].float())
        return torch.stack(L)[None]
    with torch.no_grad():
        for r in rows:
            G, ids = P0.centers(r); Pp = per[r["scene_id"]]
            for pos, ti in enumerate(ids):
                txt = Tt[(r["scene_id"], ti)][None].cuda(); tm = torch.ones(1, txt.shape[1], dtype=torch.bool, device="cuda"); out[r["task"]]["gt_prop_acc"].append(float(sc(txt, tm, crops(r, [(g[0], g[1]) for g in G]))[0].argmax().item() == pos))
                out[r["task"]]["target_covered"].append(float(any(np.hypot(p[0] - G[pos][0], p[1] - G[pos][1]) <= MATCH_PX for p in Pp)))
                if not Pp: out[r["task"]]["learned_prop_acc"].append(0.0); continue
                pk = Pp[int(sc(txt, tm, crops(r, [(p[0], p[1]) for p in Pp]))[0].argmax())]; out[r["task"]]["learned_prop_acc"].append(float(np.hypot(pk[0] - G[pos][0], pk[1] - G[pos][1]) <= MATCH_PX))
    return {t: {k: float(np.mean(v)) for k, v in d.items()} | {"n": len(d["gt_prop_acc"]), "chance": float(np.mean([1 / len(P0.centers(r)[0]) for r in rows if r["task"] == t]))} for t, d in out.items()}


def main():
    pol, pre = load_policy(); xr = pickle.load(open(X / "referent_table_x.pkl", "rb")); split = json.load(open(X / "split_x.json")); A, B, C = build_rows("A"), build_rows("B"), build_rows("C")
    cov = {n: {t: {"scenes": sum(r["task"] == t for r in R), "ok": sum(r["task"] == t and r["scene_ok"] for r in R), "seeds": [min(r["seed"] for r in R if r["task"] == t), max(r["seed"] for r in R if r["task"] == t)]} for t in TASKS} for n, R in (("train_extra_A", A), ("dev_B", B), ("test_C", C))}; print("coverage", json.dumps(cov), flush=True)
    tr = [r for r in xr if split[r["scene_id"]] == "train" and r["scene_ok"]] + [r for r in A if r["scene_ok"]]; xval = [r for r in xr if split[r["scene_id"]] == "val" and r["scene_ok"]]
    feats = {}
    for R in (tr, B, C, xval): feats.update(P0.patch_feats(pol, [r for r in R if r["scene_id"] not in feats]))
    sc = H.Scorer("B").cuda(); sc.load_state_dict(torch.load(X / "scorer_expanded.pt")); sc.eval(); res = {"coverage": cov, "train_scenes": {t: sum(r["task"] == t for r in tr) for t in TASKS}}; models = {}
    for g in (32, 64):
        T = {r["scene_id"]: targets(r, g) for r in tr}; torch.manual_seed(H.SEED); rng = np.random.default_rng(H.SEED); m = Head(g).cuda(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for step in range(1, STEPS + 1):
            m.train(); b = [tr[i] for i in rng.choice(len(tr), BATCH, replace=False)]; f = torch.stack([feats[r["scene_id"]] for r in b]).float().cuda(); hm, off, mk = [torch.from_numpy(np.stack([T[r["scene_id"]][k] for r in b])).cuda() for k in range(3)]
            out = m(f); loss = P0.focal(out[:, 0], hm) + F.l1_loss(out[:, 1:] * mk[:, None], off * mk[:, None], reduction="sum") / mk.sum().clamp_min(1); opt.zero_grad(); loss.backward(); opt.step()
            if step % 1000 == 0: print(f"grid{g} step {step} loss {float(loss):.3f}", flush=True)
        models[g] = m; torch.save(m.state_dict(), X / f"proposal_head_v21q_g{g}.pt"); od = infer(m, feats, B); res[f"dev_g{g}"] = {str(th): {t: {k: v for k, v in d.items() if k != "_per"} for t, d in evaluate(od, B, g, th).items()} for th in THRS}
        for th in THRS: print("DEV", g, th, {t: {k: round(v, 3) for k, v in d.items() if k in ("coverage", "proposals/GT", "duplicate_rate", "false_pos_rate")} for t, d in res[f"dev_g{g}"][str(th)].items()}, flush=True)
    # selection (rule fixed above)
    best = None
    for g in (32, 64):
        for th in THRS:
            d = res[f"dev_g{g}"][str(th)]; ok = all(d[t]["duplicate_rate"] <= 0.10 and d[t]["proposals/GT"] <= 1.5 for t in TASKS) and d["select_poker"]["coverage"] >= 0.98
            if ok and (best is None or d["select_mahjong"]["coverage"] > best[2]): best = (g, th, d["select_mahjong"]["coverage"])
    res["selected_on_dev"] = best; print("SELECTED", best, flush=True)
    if best is None: json.dump(res, open(X / "proposal_iter_results.json", "w"), indent=1, default=float); print("no config met the rule; STOP"); return
    g, th, _ = best; m = models[g]
    # dev composition + TEST (once)
    for name, R in (("dev", B), ("test", C), ("expansion_val_secondary", xval)):
        o = infer(m, feats, R); ev = evaluate(o, R, g, th); per = {t: ev[t]["_per"] for t in TASKS}; per_all = {k: v for t in TASKS for k, v in per[t].items()}
        res[name] = {"proposal": {t: {k: v for k, v in d.items() if k != "_per"} for t, d in ev.items()}, "composition_frozen_scorer": compose(pol, pre, sc, R, per_all, name)}; print(name, json.dumps(res[name], indent=0)[:1500], flush=True)
    json.dump(res, open(X / "proposal_iter_results.json", "w"), indent=1, default=float)

if __name__ == "__main__": main()
