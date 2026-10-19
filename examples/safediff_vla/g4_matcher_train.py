#!/usr/bin/env python
"""G4 FineMatcher smoke test (poker, ONE seed, mixed-domain 3:1; no LoRA, no resolution change, no decoder retraining).
Frozen: G3-M1 coarse path (cached w, v), VLM vision tower + connector (cached crop embeddings), VLM token embedding table (cached text embeddings).
Trainable: FineMatcher (~0.5M) + the anchor head (initialised from G3-M1, lr 10x lower).
FineMatcher: instruction token embeddings -> M=2 learned-query attention pooling heads -> q_m; crop embedding -> per-head projection g_m; score_j = sum_m <q_m, g_m(h_j)>/sqrt(dk)*scale.
(an instruction<->crop MATCHING score; there is no 52-way class output.)  Scores are softmaxed per camera over its 2 candidate crops; final spatial weights
w'_i = w_i * (sum_j p_j * 1[i in 3x3 neighbourhood of peak j] + EPS), renormalised; w' goes through the SAME pooling + anchor head as the coarse path.
Loss (sim)  = anchor + 0.5 L_ref(w') + 1.0 L_fine ; dataset = anchor only.  L_fine = per-camera multi-positive CE over that camera's 2 crops (positive = crop whose
window contains the GT card centre); skipped if no candidate contains the target (candidate-generation failure, counted); GT card is NEVER inserted into the candidate list."""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np, torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from g1_counterfactual_metrics import metrics  # noqa: E402
from g1_train import Bottleneck, huber, UNIT  # noqa: E402
from g2_train import FRAMES, guard_excess  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_g4_offline"); EPS = 0.05; COARSE = "outputs/train/safediff_vla_g3/M1_s0.pt"; dev = "cuda"
HELDF = lambda n: n.split("_of_")[0] == "9"


class FineMatcher(nn.Module):
    def __init__(self, d=960, dk=128, M=2):
        super().__init__(); self.dk, self.M = dk, M; self.ln_t, self.ln_h = nn.LayerNorm(d), nn.LayerNorm(d)
        self.tq = nn.Parameter(torch.randn(M, dk) * 0.1); self.Wk = nn.Linear(d, dk, bias=False); self.Wv = nn.Linear(d, dk); self.Wh = nn.ModuleList([nn.Linear(d, dk) for _ in range(M)])
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, tok, mask, crop):  # tok [B,T,960], mask [B,T], crop [B,J,960] -> scores [B,J]
        x = self.ln_t(tok); a = torch.einsum("md,btd->bmt", self.tq, self.Wk(x)) / math.sqrt(self.dk); a = a.masked_fill(~mask.unsqueeze(1), -1e9).softmax(-1)
        q = torch.einsum("bmt,btd->bmd", a, self.Wv(x)); h = self.ln_h(crop); g = torch.stack([w(h) for w in self.Wh], 1)  # [B,M,J,dk]
        return torch.einsum("bmd,bmjd->bj", q, g) / math.sqrt(self.dk) * self.scale


class G4Model(nn.Module):
    def __init__(self):
        super().__init__(); base = Bottleneck(); base.load_state_dict(torch.load(COARSE)); self.head = base.head; self.matcher = FineMatcher()
        i = torch.arange(192); self.register_buffer("cam", i // 64); self.register_buffer("row", ((i % 64) // 8).float() / 7); self.register_buffer("col", (i % 8).float() / 7)
        nb = torch.zeros(128, 192, dtype=torch.bool)
        for cam in (0, 1):
            for cell in range(64):
                r, c = divmod(cell, 8)
                for rr in range(max(0, r - 1), min(8, r + 2)):
                    for cc in range(max(0, c - 1), min(8, c + 2)): nb[cam * 64 + cell, cam * 64 + rr * 8 + cc] = True
        self.register_buffer("nbr", nb)

    def forward(self, w, v, cell, crop, tok, mask, use_fine=True):
        B = w.shape[0]; s = self.matcher(tok, mask, crop); p = torch.cat([s[:, :2].softmax(-1), s[:, 2:].softmax(-1)], 1)  # per-camera softmax
        if use_fine:
            camof = torch.tensor([0, 0, 1, 1], device=w.device); factor = torch.full((B, 192), EPS, device=w.device)
            for j in range(4): factor = factor + p[:, j:j + 1] * self.nbr[camof[j] * 64 + cell[:, j]].float()
            w2 = w * factor; w2 = w2 / w2.sum(-1, keepdim=True)
        else: w2 = w
        f = (w2.unsqueeze(-1) * v).sum(1); feats = []
        for c in range(3):
            m = (self.cam == c).float(); feats += [(w2 * m).sum(-1), (w2 * m * self.row).sum(-1), (w2 * m * self.col).sum(-1)]
        return self.head(torch.cat([f, torch.stack(feats, -1)], -1)), w2, p, s


class Data:
    def __init__(self):
        c = torch.load(OUT / "g4_cache.pt", weights_only=False); self.c = c; self.crop_emb = c["crops"]["emb"].to(dev).float(); self.txt = c["text"]["tok_emb"].to(dev).float(); self.txm = c["text"]["tok_mask"].to(dev)
        self.mean, self.std = c["consts"]["pos_mean"].to(dev), c["consts"]["pos_std"].to(dev); s = c["sim"]; self.sim = s
        S = len(s["sids"]); self.names = s["names"]; self.split = np.array(s["split"]); valid = s["valid"].numpy(); self.pairs = np.array([(i, k) for i in range(S) for k in range(4) if valid[i, k]])
        self.pair_of = {(i, k): n for n, (i, k) in enumerate(self.pairs)}; sel = (self.pairs[:, 0], self.pairs[:, 1])
        g = lambda x: x[sel]
        self.w = g(s["w"]).to(dev).float(); self.v = g(s["v"]).to(dev); self.cell = torch.from_numpy(g(s["cell"])).to(dev); self.crop = torch.from_numpy(g(s["crop"])).to(dev); self.instr = torch.from_numpy(g(s["instr"])).to(dev)
        self.anc = g(s["anchors"]).to(dev); self.cf = g(s["cellfrac"]).to(dev); self.heldq = g(s["held"]).to(dev); self.pur = {k: torch.from_numpy(g(v_)).to(dev) for k, v_ in s["purity"].items()}
        self.mass0 = torch.from_numpy(g(s["mass"])).to(dev); self.scene = torch.from_numpy(self.pairs[:, 0]).to(dev)
        self.d = {}
        for name in ("dtrain", "dvald", "dheld"):
            d = c[name]; n, V = d["cell"].shape[:2]; self.d[name] = {"w": d["w"].to(dev).float(), "v": d["v"].to(dev), "cell": torch.from_numpy(d["cell"]).to(dev), "crop": torch.from_numpy(d["crop"]).to(dev), "instr": torch.from_numpy(d["instr"]).to(dev),
                                                                  "anc": d["anchors"].to(dev), "meta": d["meta"], "n": n, "V": V}
        self.dtrain_bucket = {b: np.array([i for i, m in enumerate(c["dtrain"]["meta"]) if m["bucket"] == b]) for b in ("EARLY", "MID", "NEAR")}

    def sim_idx(self, split, held=False):
        m = np.isin(self.split[self.pairs[:, 0]], [split]); h = self.heldq.cpu().numpy(); return np.where(m & (h if held else ~h))[0]

    def fwd_sim(self, model, idx, use_fine=True):
        t = torch.as_tensor(idx, device=dev); ins = self.instr[t]; return model(self.w[t], self.v[t].float(), self.cell[t], self.crop_emb[self.crop[t]], self.txt[ins], self.txm[ins], use_fine)

    def fwd_data(self, model, name, idx, text=0, use_fine=True):
        d = self.d[name]; t = torch.as_tensor(idx, device=dev); ins = d["instr"][t, text]; return model(d["w"][t, text], d["v"][t, text].float(), d["cell"][t, text], self.crop_emb[d["crop"][t, text]], self.txt[ins], self.txm[ins], use_fine)


def to_m(x, data): return x * data.std + data.mean


@torch.no_grad()
def eval_sim(model, data, split, use_fine=True):
    model.eval(); held_split = split == "test_heldout_identity"; idx = data.sim_idx("test_heldout_identity" if held_split else split, held=False) if not held_split else np.where(data.split[data.pairs[:, 0]] == "test_heldout_identity")[0]
    xyz, w2, p, s = [], [], [], []
    for i in range(0, len(idx), 128):
        a, b, c, d = data.fwd_sim(model, idx[i:i + 128], use_fine); xyz.append(to_m(a, data).cpu()); w2.append(b); p.append(c); s.append(d)
    xyz = torch.cat(xyz).numpy(); w2 = torch.cat(w2); p = torch.cat(p).cpu().numpy(); s = torch.cat(s).cpu().numpy(); scenes = {}
    for n, i in enumerate(idx):
        sc, k = data.pairs[i]; nm = data.names[sc][k]; e = scenes.setdefault(sc, {"anchors": {}, "preds": {}})
        e["anchors"][nm] = data.anc[i].cpu().numpy(); e["preds"][nm] = xyz[n]
    # anchors of ALL valid candidates (incl. non-queried) so that selection is judged among every card in the scene
    for sc, e in scenes.items():
        for kk in range(4):
            if data.sim["valid"][sc, kk] and data.names[sc][kk] not in e["anchors"]: e["anchors"][data.names[sc][kk]] = data.sim["anchors"][sc, kk].numpy(); e["preds"][data.names[sc][kk]] = e["anchors"][data.names[sc][kk]] + 0.0  # placeholder, excluded by query_ok
    queried = {data.names[data.pairs[i][0]][data.pairs[i][1]] for i in idx}
    scn = [{"anchors": e["anchors"], "preds": e["preds"]} for e in scenes.values() if sum(nm in queried for nm in e["anchors"]) >= 2]
    qok = lambda c: True
    m = metrics_with_queries(scenes, data, idx, split)
    # ---- fine-stage diagnostics on (query, camera) pairs where EXACTLY ONE of the camera's 2 crops contains the target ----
    pur = {k: v[torch.as_tensor(idx, device=dev)].cpu().numpy() for k, v in data.pur.items()}; contains = pur["contains_target"]; diag = {"n_queries": len(idx)}; mass_final = np.stack([(w2.cpu().numpy()[:, None, :] * data.model_nbr[None]).sum(-1)], 0)[0] if hasattr(data, "model_nbr") else None
    rows = []
    for cam in (0, 1):
        sl = slice(cam * 2, cam * 2 + 2); c = contains[:, sl]; both = c.sum(1) == 2; one = c.sum(1) == 1; none = c.sum(1) == 0
        diag[f"cam{cam}_target_in_top2"] = float((c.sum(1) >= 1).mean()); diag[f"cam{cam}_ambiguous_both_contain_target"] = float(both.mean())
        pos = c.argmax(1); coarse_m = data.mass0.cpu().numpy()[idx][:, sl]; coarse_acc = (coarse_m.argmax(1) == pos); fine_acc = (s[:, sl].argmax(1) == pos); final_acc = None
        nb = model.nbr.cpu().numpy(); fm = np.stack([(w2.cpu().numpy() * nb[cam * 64 + data.cell.cpu().numpy()[idx][:, cam * 2 + j]]).sum(-1) for j in range(2)], 1); final_acc = (fm.argmax(1) == pos)
        diag[f"cam{cam}_n_one_positive"] = int(one.sum()); diag[f"cam{cam}_coarse_acc"] = float(coarse_acc[one].mean()) if one.any() else None; diag[f"cam{cam}_fine_only_acc"] = float(fine_acc[one].mean()) if one.any() else None
        diag[f"cam{cam}_final_mass_acc"] = float(final_acc[one].mean()) if one.any() else None
        cen = pur["target_nearest_to_center"][:, sl][np.arange(len(idx)), pos]; nc = pur["n_cards_inside"][:, sl][np.arange(len(idx)), pos]; neg_nc = pur["n_cards_inside"][:, sl][np.arange(len(idx)), 1 - pos]
        for i in np.where(one)[0]: rows.append((cam, int(cen[i]), int(nc[i]), int(neg_nc[i]), bool(coarse_acc[i]), bool(fine_acc[i]), bool(final_acc[i])))
    R = np.array(rows, dtype=int) if rows else np.zeros((0, 7), int)
    def acc(mask): return {"n": int(mask.sum()), "coarse": float(R[mask, 4].mean()) if mask.any() else None, "fine_only": float(R[mask, 5].mean()) if mask.any() else None, "final": float(R[mask, 6].mean()) if mask.any() else None}
    diag["strata"] = {"positive_crop_centered": acc(R[:, 1] == 1), "positive_crop_off_center": acc(R[:, 1] == 0), "positive_crop_single_card": acc(R[:, 2] <= 1), "positive_crop_multi_card": acc(R[:, 2] >= 2),
                      "negative_crop_background(no card)": acc(R[:, 3] == 0), "negative_crop_contains_other_card": acc(R[:, 3] >= 1), "all_one_positive": acc(np.ones(len(R), bool))}
    m["fine"] = diag; m["_arrays"] = {"idx": idx, "p": p, "s": s, "w2": w2.cpu().numpy(), "xyz": xyz}; return m


def metrics_with_queries(scenes, data, idx, split):
    held_split = split == "test_heldout_identity"; queried = {}
    for i in idx: queried.setdefault(int(data.pairs[i][0]), set()).add(data.names[data.pairs[i][0]][data.pairs[i][1]])
    sc_list = []
    for sc, e in scenes.items():
        q = queried[sc]
        if len(q) >= 2: sc_list.append({"anchors": {k: v for k, v in e["anchors"].items() if k in q}, "preds": {k: v for k, v in e["preds"].items() if k in q}})
    if held_split: return metrics(sc_list, pair_ok=lambda a, b: HELDF(a) or HELDF(b), query_ok=HELDF)
    return metrics(sc_list)


@torch.no_grad()
def eval_data(model, data, name, use_fine=True, swaps=False):
    model.eval(); d = data.d[name]; n = d["n"]; P = []; Pall = []
    for t in range(d["V"] if swaps else 1):
        pr = []
        for i in range(0, n, 128): a, *_ = data.fwd_data(model, name, np.arange(i, min(n, i + 128)), t, use_fine); pr.append(to_m(a, data).cpu())
        Pall.append(torch.cat(pr).numpy())
    A = d["anc"].cpu().numpy(); fr = np.array([m["frame"] for m in d["meta"]]) if name != "dtrain" else None; e = np.linalg.norm((Pall[0] - A)[:, :2], axis=1) * 100
    out = {k: {"median_xy_cm": float(np.median(e[fr == k])), "mean_xy_cm": float(e[fr == k].mean())} for k in FRAMES} if fr is not None else {}
    if fr is not None: out["pooled"] = {"median_xy_cm": float(np.median(e)), "mean_xy_cm": float(e.mean())}
    if swaps and len(Pall) > 1:
        sh = np.mean([np.linalg.norm((Pall[t] - Pall[0])[:, :2], axis=1) for t in (1, 2)], 0) * 100; out["swap_shift_xy_cm"] = {k: float(sh[fr == k].mean()) for k in FRAMES}
    return out


def select(h):
    ok = [x for x in h if x["excess"] <= 0]
    if not ok: return min(h, key=lambda x: (x["excess"], -x["R"]))
    b = max(x["R"] for x in ok); c = [x for x in ok if x["R"] >= b - 0.02]; ba = max(x["acc"] for x in c); c = [x for x in c if x["acc"] >= ba - 0.02]; return min(c, key=lambda x: x["loc"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--eval-every", type=int, default=100); ap.add_argument("--lam-fine", type=float, default=1.0); a = ap.parse_args()
    data = Data(); torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed); drng = np.random.default_rng(a.seed + 1000); model = G4Model().to(dev)
    # data.model_nbr unused; reference (V2 on VAL-D) from the G2/G3 machinery
    from g1_train import V2Modules
    from g2_train import DataCache, eval_frames
    refm = V2Modules().to(dev); dval0 = DataCache(Path("outputs/eval/safediff_vla_g3_offline/g3_vald_cache.pt"), dev); ref_vald = eval_frames(refm, dval0, data.mean, data.std, False)
    opt = torch.optim.AdamW([{"params": model.matcher.parameters(), "lr": 3e-4}, {"params": model.head.parameters(), "lr": 3e-5}], weight_decay=1e-4); base = [g["lr"] for g in opt.param_groups]
    tr_scenes = np.unique(data.pairs[data.split[data.pairs[:, 0]] == "train"][:, 0]); order = rng.permutation(tr_scenes); pos = 0; B = 16; hist = []; best = None; log = []
    scene_of = data.pairs[:, 0]; heldq = data.heldq
    for step in range(1, a.steps + 1):
        if pos + B > len(order): order = rng.permutation(tr_scenes); pos = 0
        sc = order[pos:pos + B]; pos += B; model.train()
        for g, b in zip(opt.param_groups, base): g["lr"] = b * min(1.0, step / 100)
        idx = np.where(np.isin(scene_of, sc) & ~heldq.cpu().numpy())[0]; n_sim = len(idx); n_dat = int(round(n_sim / 3)); per = [n_dat // 3 + (1 if i < n_dat % 3 else 0) for i in range(3)]
        di = np.concatenate([drng.choice(data.dtrain_bucket[b], size=per[i], replace=True) for i, b in enumerate(("EARLY", "MID", "NEAR"))])
        xs, w2s, ps, ss = data.fwd_sim(model, idx); xd, *_ = data.fwd_data(model, "dtrain", di); t = torch.as_tensor(idx, device=dev)
        la_s = huber((to_m(xs, data) - data.anc[t]) / UNIT).mean(-1); y = data.cf[t]; has = y.sum(-1) > 0; lref = torch.zeros_like(la_s)
        if has.any(): yy = y[has] / y[has].sum(-1, keepdim=True); lref[has] = -(yy * torch.log(w2s[has] + 1e-8)).sum(-1)
        la_d = huber((to_m(xd, data) - data.d["dtrain"]["anc"][torch.as_tensor(di, device=dev)]) / UNIT).mean(-1)
        main_loss = ((la_s + 0.5 * lref).sum() + la_d.sum()) / (n_sim + n_dat)
        # L_fine: per camera, multi-positive CE over the 2 crops, only when >=1 and <2 crops contain the target
        cont = data.pur["contains_target"][t]; lf = []
        for cam in (0, 1):
            c = cont[:, cam * 2:cam * 2 + 2]; el = (c.sum(1) == 1); 
            if el.any(): logp = F.log_softmax(ss[:, cam * 2:cam * 2 + 2], -1); lf.append(-(logp * c).sum(1)[el])
        l_fine = torch.cat(lf).mean() if lf else torch.zeros((), device=dev); loss = main_loss + a.lam_fine * l_fine
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 25 == 0: log.append({"step": step, "loss": float(loss), "L_fine": float(l_fine), "anchor_sim": float(la_s.mean()), "ref": float(lref[has].mean()) if has.any() else 0.0})
        if step % a.eval_every == 0:
            m = eval_sim(model, data, "val"); r = eval_data(model, data, "dvald"); excess = guard_excess({k: r[k] for k in FRAMES}, ref_vald)
            h = {"step": step, "R": m["R_follow_xy_pooled"], "acc": m["selection_acc"], "loc": m["loc_err_xy_cm_median"], "excess": excess, "fine_acc_front": m["fine"]["cam0_fine_only_acc"], "fine_acc_right": m["fine"]["cam1_fine_only_acc"], "retention_valD": {k: r[k]["median_xy_cm"] for k in FRAMES}}
            hist.append(h); print(f"step {step:4d} L_fine {float(l_fine):.3f} | val R={h['R']:+.3f} acc={h['acc']:.2f} loc={h['loc']:.1f} | fine acc front/right {h['fine_acc_front']:.2f}/{h['fine_acc_right']:.2f} | valD excess {excess:+.2f}", flush=True)
            if select(hist)["step"] == step: best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sel = select(hist); model.load_state_dict(best); res = {"seed": a.seed, "selected_step": sel["step"], "history": hist, "train_log": log, "guard_ok_at_selected": sel["excess"] <= 0}
    OUT.mkdir(exist_ok=True); torch.save(model.state_dict(), OUT / f"g4_matcher_s{a.seed}.pt"); arrays = {}
    for sp in ("val", "test", "test_heldout_identity"):
        m = eval_sim(model, data, sp); arrays[sp] = m.pop("_arrays"); res[sp] = m
        m0 = eval_sim(model, data, sp, use_fine=False); m0.pop("_arrays"); res[sp + "_coarse_only_same_head"] = {k: m0[k] for k in ("R_follow_xy_pooled", "selection_acc", "loc_err_xy_cm_median")}
    res["retention_heldout"] = eval_data(model, data, "dheld", swaps=True); res["retention_valD"] = eval_data(model, data, "dvald"); torch.save(arrays, OUT / f"g4_matcher_s{a.seed}_arrays.pt")
    (OUT / f"g4_smoke_s{a.seed}.json").write_text(json.dumps(res, indent=1, default=float))
    t = res["test"]; print("TEST R=%.3f acc=%.3f loc=%.1f | held-out id R=%.3f acc=%.3f | fine:" % (t["R_follow_xy_pooled"], t["selection_acc"], t["loc_err_xy_cm_median"], res["test_heldout_identity"]["R_follow_xy_pooled"], res["test_heldout_identity"]["selection_acc"]), json.dumps(t["fine"], default=float)[:600])


if __name__ == "__main__":
    main()
