#!/usr/bin/env python
"""G4b FineMatcher smoke (poker, ONE seed). Differences from the G4 smoke (the only two variables): (1) TRAIN candidates come from OOF proposal maps ("OOF proposal maps with the fixed final-M1 value pathway": w = fold model output,
v / anchor head = final G3-M1; v never enters candidate selection or the FineMatcher), (2) every candidate is a FRONT-view 144px crop (right camera only proposes, via 3x3 centroid -> table plane -> front).
UNCHANGED: FineMatcher architecture / params, optimizer, lrs, weight decay, steps, batch structure (16 scenes, sim:dataset 3:1), loss weights (anchor + 0.5 L_ref + 1.0 L_fine), checkpoint selection, no dropout / extra negatives / LoRA.
Adaptation forced by the single front crop set: the per-camera softmax becomes one softmax over the (<=4, deduplicated) valid candidates; candidate -> spatial-weight mapping uses each candidate's precomputed 3x3 cell neighbourhood in BOTH cameras
(front pixel of the crop, right pixel of the same table point); L_fine = -log(sum of p over positive crops) (== old CE when one positive), only for queries with >=1 positive and >=1 negative candidate."""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np, torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from g1_train import Bottleneck, huber, UNIT, V2Modules  # noqa: E402
from g2_train import FRAMES, guard_excess, DataCache, eval_frames  # noqa: E402
from g4_matcher_train import FineMatcher, metrics_with_queries, select, to_m, EPS, COARSE  # noqa: E402

OLD = Path("outputs/eval/safediff_vla_g4_offline"); OUT = Path("outputs/eval/safediff_vla_g4b_offline"); dev = "cuda"


class G4bModel(nn.Module):
    def __init__(self):
        super().__init__(); base = Bottleneck(); base.load_state_dict(torch.load(COARSE)); self.head = base.head; self.matcher = FineMatcher()
        i = torch.arange(192); self.register_buffer("cam", i // 64); self.register_buffer("row", ((i % 64) // 8).float() / 7); self.register_buffer("col", (i % 8).float() / 7)

    def forward(self, w, v, nbr, cvalid, crop, tok, mask, use_fine=True):
        s = self.matcher(tok, mask, crop).masked_fill(~cvalid, -1e9); p = s.softmax(-1)
        if use_fine: w2 = w * (EPS + (p.unsqueeze(-1) * nbr.float()).sum(1)); w2 = w2 / w2.sum(-1, keepdim=True)
        else: w2 = w
        f = (w2.unsqueeze(-1) * v).sum(1); feats = []
        for c in range(3):
            m = (self.cam == c).float(); feats += [(w2 * m).sum(-1), (w2 * m * self.row).sum(-1), (w2 * m * self.col).sum(-1)]
        return self.head(torch.cat([f, torch.stack(feats, -1)], -1)), w2, p, s


class Data:
    def __init__(self):
        o = torch.load(OLD / "g4_cache.pt", weights_only=False); n = torch.load(OUT / "g4b_cache.pt", weights_only=False); self.crop_emb = n["crops"]["emb"].to(dev).float(); self.txt = o["text"]["tok_emb"].to(dev).float(); self.txm = o["text"]["tok_mask"].to(dev)
        self.mean, self.std = o["consts"]["pos_mean"].to(dev), o["consts"]["pos_std"].to(dev); s, sn = o["sim"], n["sim"]; self.sim = s; self.names = s["names"]; self.split = np.array(s["split"]); S = len(s["sids"]); valid = s["valid"].numpy()
        self.pairs = np.array([(i, k) for i in range(S) for k in range(4) if valid[i, k]]); self.pair_of = {(i, k): m for m, (i, k) in enumerate(self.pairs)}; sel = (self.pairs[:, 0], self.pairs[:, 1]); g = lambda x: x[sel]
        T = lambda x: torch.from_numpy(np.ascontiguousarray(g(x))).to(dev)
        self.w = T(sn["w"]).float(); self.v = g(s["v"]).to(dev); self.nbr = T(sn["nbr"]); self.cvalid = T(sn["valid"]); self.crop = T(sn["crop"]); self.instr = torch.from_numpy(g(s["instr"])).to(dev); self.src = T(sn["src"])
        self.anc = g(s["anchors"]).to(dev); self.cf = g(s["cellfrac"]).to(dev); self.heldq = g(s["held"]).to(dev); self.pur = {k: T(v_) for k, v_ in sn["purity"].items()}; self.scene = torch.from_numpy(self.pairs[:, 0]).to(dev)
        self.d = {}
        for name in ("dtrain", "dvald", "dheld"):
            d, dn = o[name], n[name]; self.d[name] = {"w": dn["w"].to(dev).float(), "v": d["v"].to(dev), "nbr": torch.from_numpy(dn["nbr"]).to(dev), "cvalid": torch.from_numpy(dn["valid"]).to(dev), "crop": torch.from_numpy(dn["crop"]).to(dev),
                                                      "instr": torch.from_numpy(d["instr"]).to(dev), "anc": d["anchors"].to(dev), "meta": d["meta"], "n": d["cell"].shape[0], "V": d["cell"].shape[1]}
        self.dtrain_bucket = {b: np.array([i for i, m in enumerate(o["dtrain"]["meta"]) if m["bucket"] == b]) for b in ("EARLY", "MID", "NEAR")}

    def sim_idx(self, split, held=False):
        m = self.split[self.pairs[:, 0]] == split; h = self.heldq.cpu().numpy(); return np.where(m & (h if held else ~h))[0]

    def fwd_sim(self, model, idx, use_fine=True):
        t = torch.as_tensor(idx, device=dev); ins = self.instr[t]; return model(self.w[t], self.v[t].float(), self.nbr[t], self.cvalid[t], self.crop_emb[self.crop[t]], self.txt[ins], self.txm[ins], use_fine)

    def fwd_data(self, model, name, idx, text=0, use_fine=True):
        d = self.d[name]; t = torch.as_tensor(idx, device=dev); ins = d["instr"][t, text]; return model(d["w"][t, text], d["v"][t, text].float(), d["nbr"][t, text], d["cvalid"][t, text], self.crop_emb[d["crop"][t, text]], self.txt[ins], self.txm[ins], use_fine)


def fine_diag(s, cvalid, pur, w, w2, nbr):
    """candidate-level matching diagnostics. s: fine scores [N,J]; pur: dict of [N,J]; coarse score of a candidate = coarse mass in its neighbourhood; final = mass of w2 in it."""
    contains = pur["contains_target"].astype(bool) & cvalid; ncard = pur["n_cards"]; negcard = ~contains & cvalid & (ncard > 0); negbg = ~contains & cvalid & (ncard == 0); neg = ~contains & cvalid
    cm = (w[:, None, :] * nbr).sum(-1); fm = (w2[:, None, :] * nbr).sum(-1); out = {"n_queries": int(len(s))}
    def best(x, m): return np.where(m, x, -1e9).max(1)
    for tag, sc in (("fine", s), ("coarse", cm), ("final_map", fm)):
        for name, negm in (("card_vs_card", negcard), ("card_vs_background", negbg), ("positive_vs_all_negatives", neg)):
            ok = contains.any(1) & negm.any(1)
            out.setdefault(name, {})[tag] = float((best(sc, contains)[ok] > best(sc, negm)[ok]).mean()) if ok.any() else None; out[name]["n"] = int(ok.sum())
    ok = contains.any(1); out["target_in_any_candidate"] = float(ok.mean()); ok2 = ok & neg.any(1)
    for tag, sc in (("fine", s), ("coarse", cm), ("final_map", fm)): out.setdefault("top1_candidate_contains_target", {})[tag] = float(contains[np.arange(len(s)), np.where(cvalid, sc, -1e9).argmax(1)][ok2].mean())
    return out


@torch.no_grad()
def eval_sim(model, data, split, use_fine=True):
    model.eval(); held_split = split == "test_heldout_identity"; idx = np.where(data.split[data.pairs[:, 0]] == "test_heldout_identity")[0] if held_split else data.sim_idx(split)
    xyz, w2, p, s = [], [], [], []
    for i in range(0, len(idx), 128): a, b, c, d = data.fwd_sim(model, idx[i:i + 128], use_fine); xyz.append(to_m(a, data).cpu()); w2.append(b); p.append(c); s.append(d)
    xyz = torch.cat(xyz).numpy(); w2 = torch.cat(w2); p = torch.cat(p).cpu().numpy(); s = torch.cat(s).cpu().numpy(); scenes = {}
    for n, i in enumerate(idx):
        sc, k = data.pairs[i]; nm = data.names[sc][k]; e = scenes.setdefault(sc, {"anchors": {}, "preds": {}}); e["anchors"][nm] = data.anc[i].cpu().numpy(); e["preds"][nm] = xyz[n]
    m = metrics_with_queries(scenes, data, idx, split); t = torch.as_tensor(idx, device=dev)
    if not held_split or True:
        pur = {k: v[t].cpu().numpy() for k, v in data.pur.items()}; m["fine"] = fine_diag(s, data.cvalid[t].cpu().numpy(), pur, data.w[t].cpu().numpy(), w2.cpu().numpy(), data.nbr[t].cpu().numpy().astype(np.float32))
    m["_arrays"] = {"idx": idx, "p": p, "s": s, "w2": w2.cpu().numpy(), "xyz": xyz}; return m


@torch.no_grad()
def eval_data(model, data, name, use_fine=True, swaps=False):
    model.eval(); d = data.d[name]; n = d["n"]; Pall = []
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


@torch.no_grad()
def train_fine_diag(model, data):
    model.eval(); idx = data.sim_idx("train"); s, w2 = [], []
    for i in range(0, len(idx), 128): _, b, _, d = data.fwd_sim(model, idx[i:i + 128]); s.append(d); w2.append(b)
    t = torch.as_tensor(idx, device=dev); pur = {k: v[t].cpu().numpy() for k, v in data.pur.items()}
    return fine_diag(torch.cat(s).cpu().numpy(), data.cvalid[t].cpu().numpy(), pur, data.w[t].cpu().numpy(), torch.cat(w2).cpu().numpy(), data.nbr[t].cpu().numpy().astype(np.float32))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--eval-every", type=int, default=100); ap.add_argument("--lam-fine", type=float, default=1.0); a = ap.parse_args()
    data = Data(); torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed); drng = np.random.default_rng(a.seed + 1000); model = G4bModel().to(dev)
    refm = V2Modules().to(dev); dval0 = DataCache(Path("outputs/eval/safediff_vla_g3_offline/g3_vald_cache.pt"), dev); ref_vald = eval_frames(refm, dval0, data.mean, data.std, False)
    opt = torch.optim.AdamW([{"params": model.matcher.parameters(), "lr": 3e-4}, {"params": model.head.parameters(), "lr": 3e-5}], weight_decay=1e-4); base = [g["lr"] for g in opt.param_groups]
    tr_scenes = np.unique(data.pairs[data.split[data.pairs[:, 0]] == "train"][:, 0]); order = rng.permutation(tr_scenes); pos = 0; B = 16; hist = []; best = None; log = []; scene_of = data.pairs[:, 0]; heldq = data.heldq
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
        cont = data.pur["contains_target"][t] > 0.5; cv = data.cvalid[t]; c = cont & cv; neg = ~cont & cv; el = c.any(1) & neg.any(1)
        l_fine = -((ps[el] * c[el].float()).sum(1).clamp_min(1e-8).log()).mean() if el.any() else torch.zeros((), device=dev); loss = main_loss + a.lam_fine * l_fine
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 25 == 0: log.append({"step": step, "loss": float(loss.detach()), "L_fine": float(l_fine.detach()), "anchor_sim": float(la_s.detach().mean()), "ref": float(lref[has].mean()) if has.any() else 0.0})
        if step % a.eval_every == 0:
            m = eval_sim(model, data, "val"); r = eval_data(model, data, "dvald"); excess = guard_excess({k: r[k] for k in FRAMES}, ref_vald); fd = m["fine"]
            h = {"step": step, "R": m["R_follow_xy_pooled"], "acc": m["selection_acc"], "loc": m["loc_err_xy_cm_median"], "excess": excess, "card_vs_card_fine": fd["card_vs_card"]["fine"], "retention_valD": {k: r[k]["median_xy_cm"] for k in FRAMES}}
            hist.append(h); print(f"step {step:4d} L_fine {float(l_fine):.3f} | val R={h['R']:+.3f} acc={h['acc']:.2f} loc={h['loc']:.1f} | card-vs-card fine {h['card_vs_card_fine']:.2f} (coarse {fd['card_vs_card']['coarse']:.2f}) | valD excess {excess:+.2f}", flush=True)
            if select(hist)["step"] == step: best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sel = select(hist); model.load_state_dict(best); res = {"seed": a.seed, "selected_step": sel["step"], "history": hist, "train_log": log, "guard_ok_at_selected": sel["excess"] <= 0}
    torch.save(model.state_dict(), OUT / f"g4b_matcher_s{a.seed}.pt"); arrays = {}; res["train_fine"] = train_fine_diag(model, data)
    for sp in ("val", "test", "test_heldout_identity"):
        m = eval_sim(model, data, sp); arrays[sp] = m.pop("_arrays"); res[sp] = m
        m0 = eval_sim(model, data, sp, use_fine=False); m0.pop("_arrays"); res[sp + "_coarse_only_same_head"] = {k: m0[k] for k in ("R_follow_xy_pooled", "selection_acc", "loc_err_xy_cm_median")}
    res["retention_heldout"] = eval_data(model, data, "dheld", swaps=True); res["retention_valD"] = eval_data(model, data, "dvald"); res["retention_heldout_V2"] = eval_frames(refm, DataCache(Path("outputs/eval/safediff_vla_g3_offline/g3_held_cache.pt"), dev), data.mean, data.std, False)
    torch.save(arrays, OUT / f"g4b_matcher_s{a.seed}_arrays.pt"); (OUT / f"g4b_smoke_s{a.seed}.json").write_text(json.dumps(res, indent=1, default=float))
    t = res["test"]; print("TEST R=%.3f acc=%.3f loc=%.1f | held-out id R=%.3f acc=%.3f" % (t["R_follow_xy_pooled"], t["selection_acc"], t["loc_err_xy_cm_median"], res["test_heldout_identity"]["R_follow_xy_pooled"], res["test_heldout_identity"]["selection_acc"]))
    for k in ("train_fine", "val", "test", "test_heldout_identity"): f = res[k]["fine"] if k != "train_fine" else res[k]; print(k, "card_vs_card", json.dumps(f["card_vs_card"]), "| top1", json.dumps(f["top1_candidate_contains_target"]))


if __name__ == "__main__":
    main()
