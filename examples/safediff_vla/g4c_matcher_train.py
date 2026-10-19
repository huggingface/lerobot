#!/usr/bin/env python
"""G4c INTRA-CROP REFERENT LOCALIZATION smoke (poker, ONE seed, mixed-domain 3:1). Same OOF proposals / 144px front crops / dedup / value pathway / anchor head / optimizer / lrs / steps / selection as G4b. Changes (G4c-A..F):
 A  crop = its 64 frozen tokens H_j [64,960] (no mean pooling; frozen VLM, no LoRA).
 B  per head m: instruction vector q_m (same text pooling as G4b); fine map  a_{j,t,m}=<q_m, Wg_m LN(H_jt)>/sqrt(dk)*s_a,  alpha_{j,m}=softmax_t;  f_{jm}=sum_t alpha_{jtm} Wc_m LN(H_jt);  candidate score s_j=sum_m <q_m,f_jm>/sqrt(dk)*s_s.  alpha_j = head mean.
 C  L_fine_ref = CE(y_j, alpha_j) on positive crops, y_jt = GT target-card area fraction in token t (sim segmentation bbox, TRAINING only). L_candidate = -log sum_{positive crops} p_j (as G4b). Separate terms.
 D  global map: the 64 crop-token weights alpha_j are scattered (bilinear) back onto the 8x8 coarse cells of the front camera and, via front ray -> table plane, the right camera: M_j=alpha_j.proj_j [192];
    w' = norm(w * (EPS + sum_j p_j M_j)). No GT is used at inference; GT never moves/adds/removes candidates.
 E  anchor head: unchanged G4b/M1 head, same low lr.   F  loss = anchor + 0.5 L_ref(w') + 1.0 L_candidate + 1.0 L_fine_ref (dataset: anchor only)."""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np, torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import g4b_matcher_train as gb  # noqa: E402
from g1_train import Bottleneck, huber, UNIT, V2Modules  # noqa: E402
from g2_train import FRAMES, guard_excess, DataCache, eval_frames  # noqa: E402
from g4_matcher_train import select, to_m, EPS, COARSE  # noqa: E402

OUTC = Path("outputs/eval/safediff_vla_g4c_offline"); dev = "cuda"


class SpatialMatcher(nn.Module):
    def __init__(self, d=960, dk=128, M=2):
        super().__init__(); self.dk, self.M = dk, M; self.ln_t, self.ln_h = nn.LayerNorm(d), nn.LayerNorm(d)
        self.tq = nn.Parameter(torch.randn(M, dk) * 0.1); self.Wk = nn.Linear(d, dk, bias=False); self.Wv = nn.Linear(d, dk)
        self.Wg = nn.ModuleList([nn.Linear(d, dk) for _ in range(M)]); self.Wc = nn.ModuleList([nn.Linear(d, dk) for _ in range(M)]); self.scale_a = nn.Parameter(torch.tensor(1.0)); self.scale_s = nn.Parameter(torch.tensor(1.0))

    def forward(self, tok, mask, H):  # tok [B,T,960], mask [B,T], H [B,J,64,960] -> s [B,J], alpha [B,J,64]
        x = self.ln_t(tok); a = torch.einsum("md,btd->bmt", self.tq, self.Wk(x)) / math.sqrt(self.dk); a = a.masked_fill(~mask.unsqueeze(1), -1e9).softmax(-1); q = torch.einsum("bmt,btd->bmd", a, self.Wv(x))
        h = self.ln_h(H); g = torch.stack([w(h) for w in self.Wg], 1); c = torch.stack([w(h) for w in self.Wc], 1)  # [B,M,J,64,dk]
        alpha_m = (torch.einsum("bmd,bmjtd->bmjt", q, g) / math.sqrt(self.dk) * self.scale_a).softmax(-1); f = torch.einsum("bmjt,bmjtd->bmjd", alpha_m, c)
        return torch.einsum("bmd,bmjd->bj", q, f) / math.sqrt(self.dk) * self.scale_s, alpha_m.mean(1)


class G4cModel(nn.Module):
    def __init__(self, fusion="mult", beta=0.2):  # fusion: "mult" (G4c: w*(EPS+M_fine)) | "add" (G4d: norm((1-beta)*M_fine + beta*w), M_fine = norm(sum_j p_j alpha_j.proj_j))
        super().__init__(); self.fusion, self.beta = fusion, beta; base = Bottleneck(); base.load_state_dict(torch.load(COARSE)); self.head = base.head; self.matcher = SpatialMatcher()
        i = torch.arange(192); self.register_buffer("cam", i // 64); self.register_buffer("row", ((i % 64) // 8).float() / 7); self.register_buffer("col", (i % 8).float() / 7)

    def forward(self, w, v, proj, cvalid, H, tok, mask, use_fine=True, p_force=None):  # p_force [B,J]: inference-time candidate-probability override (G6 referent lock); None = unchanged G4c/G4d behaviour
        s, alpha = self.matcher(tok, mask, H); s = s.masked_fill(~cvalid, -1e9); p = s.softmax(-1)
        if p_force is not None: p = p_force
        if use_fine:
            Mj = torch.einsum("bjt,bjtc->bjc", alpha, proj.float()); Mf = (p.unsqueeze(-1) * Mj).sum(1)
            if self.fusion == "mult": w2 = w * (EPS + Mf); w2 = w2 / w2.sum(-1, keepdim=True)
            else: Mf = Mf / Mf.sum(-1, keepdim=True).clamp_min(1e-8); w2 = (1 - self.beta) * Mf + self.beta * w; w2 = w2 / w2.sum(-1, keepdim=True)
        else: w2 = w
        f = (w2.unsqueeze(-1) * v).sum(1); feats = []
        for c in range(3):
            m = (self.cam == c).float(); feats += [(w2 * m).sum(-1), (w2 * m * self.row).sum(-1), (w2 * m * self.col).sum(-1)]
        return self.head(torch.cat([f, torch.stack(feats, -1)], -1)), w2, p, s, alpha


class Data(gb.Data):
    def __init__(self):
        super().__init__(); self.tokens = torch.load(OUTC / "g4c_crop_tokens.pt").to(dev); pm = torch.load(OUTC / "g4c_proj_tokmask.pt"); pi = torch.as_tensor(self.pairs[:, 0]); pk = torch.as_tensor(self.pairs[:, 1])
        self.proj = pm["sim"]["proj"][pi, pk].to(dev); self.tokmask = pm["sim"]["tokmask"][pi, pk].to(dev)  # [P,J,64,192], [P,J,4,64]
        self.tgt_tok = self.tokmask[torch.arange(len(pk), device=dev), :, torch.as_tensor(self.pairs[:, 1], device=dev)]  # [P,J,64] target-card area fraction
        for name in ("dtrain", "dvald", "dheld"): self.d[name]["proj"] = pm[name]["proj"].to(dev)

    def fwd_sim(self, model, idx, use_fine=True):
        t = torch.as_tensor(idx, device=dev); ins = self.instr[t]; return model(self.w[t], self.v[t].float(), self.proj[t], self.cvalid[t], self.tokens[self.crop[t]].float(), self.txt[ins], self.txm[ins], use_fine)

    def fwd_data(self, model, name, idx, text=0, use_fine=True):
        d = self.d[name]; t = torch.as_tensor(idx, device=dev); ins = d["instr"][t, text]; return model(d["w"][t, text], d["v"][t, text].float(), d["proj"][t, text], d["cvalid"][t, text], self.tokens[d["crop"][t, text]].float(), self.txt[ins], self.txm[ins], use_fine)


def intra_diag(alpha, cvalid, contains, tokmask, k, s):
    """alpha [N,J,64]; tokmask [N,J,4cards,64] (area fraction of each card in each crop token); k [N] target card. Card token set = tokens with >=25% card coverage (fallback: best token).
    Reported on (a) every POSITIVE crop (crop contains the target), (b) the crop the fine matcher ranks first, when it contains the target. top_token_in_target, target_mass (sum alpha on target tokens), chance_mass (target token fraction),
    correct_card (card with the largest alpha mass among the cards present in the crop; multi-card crops only), and `uniform` = same metrics for alpha=1/64."""
    N, J = alpha.shape[:2]; tm = tokmask.astype(np.float32); sets = tm >= 0.25
    for n in range(N):
        for j in range(J):
            for c in range(tm.shape[2]):
                if tm[n, j, c].max() > 0 and not sets[n, j, c].any(): sets[n, j, c, tm[n, j, c].argmax()] = True
    tgt = sets[np.arange(N), :, k]; out = {}
    top = np.where(cvalid, s, -1e9).argmax(1)
    for lab, sel in (("positive_crops", contains & tgt.any(-1) & cvalid), ("fine_selected_crop_if_contains_target", (np.arange(J)[None] == top[:, None]) & contains & tgt.any(-1))):
        nn_, jj = np.where(sel); 
        if len(nn_) == 0: out[lab] = {"n": 0}; continue
        a = alpha[nn_, jj]; T = tgt[nn_, jj]; masses = np.einsum("nt,nct->nc", a, sets[nn_, jj].astype(np.float32)); present = sets[nn_, jj].any(-1); multi = present.sum(1) >= 2
        masses_u = np.einsum("t,nct->nc", np.full(64, 1 / 64, np.float32), sets[nn_, jj].astype(np.float32)); kk = k[nn_]
        cc = np.where(present, masses, -1).argmax(1) == kk; cu = np.where(present, masses_u, -1).argmax(1) == kk
        out[lab] = {"n": int(len(nn_)), "top_token_in_target": float(T[np.arange(len(nn_)), a.argmax(1)].mean()), "target_mass": float((a * T).sum(1).mean()), "chance_mass(target token fraction)": float(T.mean()),
                    "n_multi_card": int(multi.sum()), "correct_card_multi": float(cc[multi].mean()) if multi.any() else None, "correct_card_multi_uniform_alpha": float(cu[multi].mean()) if multi.any() else None}
    return out


def intra_for(data, arr_idx, alpha, s):
    t = torch.as_tensor(arr_idx, device=dev); cont = data.pur["contains_target"][t].cpu().numpy() > .5
    return intra_diag(alpha, data.cvalid[t].cpu().numpy(), cont, data.tokmask[t].float().cpu().numpy(), data.pairs[arr_idx, 1], s)


@torch.no_grad()
def eval_sim(model, data, split, use_fine=True):
    model.eval(); held_split = split == "test_heldout_identity"; idx = np.where(data.split[data.pairs[:, 0]] == "test_heldout_identity")[0] if held_split else data.sim_idx(split)
    xyz, w2, p, s, al = [], [], [], [], []
    for i in range(0, len(idx), 64): a, b, c, d, e = data.fwd_sim(model, idx[i:i + 64], use_fine); xyz.append(to_m(a, data).cpu()); w2.append(b); p.append(c); s.append(d); al.append(e)
    xyz = torch.cat(xyz).numpy(); w2 = torch.cat(w2); p = torch.cat(p).cpu().numpy(); s = torch.cat(s).cpu().numpy(); al = torch.cat(al).cpu().numpy(); scenes = {}
    for n, i in enumerate(idx):
        sc, k = data.pairs[i]; nm = data.names[sc][k]; e = scenes.setdefault(sc, {"anchors": {}, "preds": {}}); e["anchors"][nm] = data.anc[i].cpu().numpy(); e["preds"][nm] = xyz[n]
    m = gb.metrics_with_queries(scenes, data, idx, split); t = torch.as_tensor(idx, device=dev); pur = {k: v[t].cpu().numpy() for k, v in data.pur.items()}
    m["fine"] = gb.fine_diag(s, data.cvalid[t].cpu().numpy(), pur, data.w[t].cpu().numpy(), w2.cpu().numpy(), data.nbr[t].cpu().numpy().astype(np.float32)); m["intra"] = intra_for(data, idx, al, s)
    m["_arrays"] = {"idx": idx, "p": p, "s": s, "alpha": al, "w2": w2.cpu().numpy(), "xyz": xyz}; return m


@torch.no_grad()
def train_diag(model, data):
    model.eval(); idx = data.sim_idx("train"); s, w2, al = [], [], []
    for i in range(0, len(idx), 64): _, b, _, d, e = data.fwd_sim(model, idx[i:i + 64]); s.append(d); w2.append(b); al.append(e)
    s = torch.cat(s).cpu().numpy(); w2 = torch.cat(w2).cpu().numpy(); al = torch.cat(al).cpu().numpy(); t = torch.as_tensor(idx, device=dev); pur = {k: v[t].cpu().numpy() for k, v in data.pur.items()}
    return gb.fine_diag(s, data.cvalid[t].cpu().numpy(), pur, data.w[t].cpu().numpy(), w2, data.nbr[t].cpu().numpy().astype(np.float32)), intra_for(data, idx, al, s)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--eval-every", type=int, default=100); ap.add_argument("--lam-fine", type=float, default=1.0); ap.add_argument("--lam-intra", type=float, default=1.0); ap.add_argument("--fusion", default="mult"); ap.add_argument("--beta", type=float, default=0.2); ap.add_argument("--tag", default="g4c"); ap.add_argument("--out", default=str(OUTC)); a = ap.parse_args(); out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    data = Data(); torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed); drng = np.random.default_rng(a.seed + 1000); model = G4cModel(a.fusion, a.beta).to(dev)
    refm = V2Modules().to(dev); dval0 = DataCache(Path("outputs/eval/safediff_vla_g3_offline/g3_vald_cache.pt"), dev); ref_vald = eval_frames(refm, dval0, data.mean, data.std, False)
    opt = torch.optim.AdamW([{"params": model.matcher.parameters(), "lr": 3e-4}, {"params": model.head.parameters(), "lr": 3e-5}], weight_decay=1e-4); base = [g["lr"] for g in opt.param_groups]
    tr_scenes = np.unique(data.pairs[data.split[data.pairs[:, 0]] == "train"][:, 0]); order = rng.permutation(tr_scenes); pos = 0; B = 16; hist = []; best = None; log = []; scene_of = data.pairs[:, 0]; heldq = data.heldq.cpu().numpy()
    for step in range(1, a.steps + 1):
        if pos + B > len(order): order = rng.permutation(tr_scenes); pos = 0
        sc = order[pos:pos + B]; pos += B; model.train()
        for g, b in zip(opt.param_groups, base): g["lr"] = b * min(1.0, step / 100)
        idx = np.where(np.isin(scene_of, sc) & ~heldq)[0]; n_sim = len(idx); n_dat = int(round(n_sim / 3)); per = [n_dat // 3 + (1 if i < n_dat % 3 else 0) for i in range(3)]
        di = np.concatenate([drng.choice(data.dtrain_bucket[b], size=per[i], replace=True) for i, b in enumerate(("EARLY", "MID", "NEAR"))])
        xs, w2s, ps, ss, al = data.fwd_sim(model, idx); xd, *_ = data.fwd_data(model, "dtrain", di); t = torch.as_tensor(idx, device=dev)
        la_s = huber((to_m(xs, data) - data.anc[t]) / UNIT).mean(-1); y = data.cf[t]; has = y.sum(-1) > 0; lref = torch.zeros_like(la_s)
        if has.any(): yy = y[has] / y[has].sum(-1, keepdim=True); lref[has] = -(yy * torch.log(w2s[has] + 1e-8)).sum(-1)
        la_d = huber((to_m(xd, data) - data.d["dtrain"]["anc"][torch.as_tensor(di, device=dev)]) / UNIT).mean(-1)
        main_loss = ((la_s + 0.5 * lref).sum() + la_d.sum()) / (n_sim + n_dat)
        cont = data.pur["contains_target"][t] > 0.5; cv = data.cvalid[t]; c = cont & cv; neg = ~cont & cv; el = c.any(1) & neg.any(1)
        l_cand = -((ps[el] * c[el].float()).sum(1).clamp_min(1e-8).log()).mean() if el.any() else torch.zeros((), device=dev)
        ytok = data.tgt_tok[t]; ysum = ytok.sum(-1); pc = c & (ysum > 1e-6)
        l_intra = -((ytok[pc] / ysum[pc].unsqueeze(-1)) * torch.log(al[pc] + 1e-8)).sum(-1).mean() if pc.any() else torch.zeros((), device=dev)
        loss = main_loss + a.lam_fine * l_cand + a.lam_intra * l_intra
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 25 == 0: log.append({"step": step, "loss": float(loss.detach()), "L_candidate": float(l_cand.detach()), "L_fine_ref": float(l_intra.detach()), "anchor_sim": float(la_s.detach().mean()), "ref": float(lref[has].mean()) if has.any() else 0.0})
        if step % a.eval_every == 0:
            m = eval_sim(model, data, "val"); r = eval_data = gb.eval_data(model, data, "dvald"); excess = guard_excess({k: r[k] for k in FRAMES}, ref_vald); fd = m["fine"]; ia = m["intra"]["positive_crops"]
            h = {"step": step, "R": m["R_follow_xy_pooled"], "acc": m["selection_acc"], "loc": m["loc_err_xy_cm_median"], "excess": excess, "card_vs_card_fine": fd["card_vs_card"]["fine"], "intra_top_token": ia["top_token_in_target"], "intra_mass": ia["target_mass"], "retention_valD": {k: r[k]["median_xy_cm"] for k in FRAMES}}
            hist.append(h); print(f"step {step:4d} Lc {float(l_cand):.3f} Lintra {float(l_intra):.3f} | val R={h['R']:+.3f} acc={h['acc']:.2f} loc={h['loc']:.1f} | cvc {h['card_vs_card_fine']:.2f} | intra top-tok {h['intra_top_token']:.2f} mass {h['intra_mass']:.2f} | valD excess {excess:+.2f}", flush=True)
            if select(hist)["step"] == step: best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sel = select(hist); model.load_state_dict(best); res = {"seed": a.seed, "selected_step": sel["step"], "history": hist, "train_log": log, "guard_ok_at_selected": sel["excess"] <= 0}
    torch.save(model.state_dict(), out_dir / f"{a.tag}_matcher_s{a.seed}.pt"); arrays = {}; res["train_fine"], res["train_intra"] = train_diag(model, data)
    for sp in ("val", "test", "test_heldout_identity"):
        m = eval_sim(model, data, sp); arrays[sp] = m.pop("_arrays"); res[sp] = m
        m0 = eval_sim(model, data, sp, use_fine=False); m0.pop("_arrays"); res[sp + "_coarse_only_same_head"] = {k: m0[k] for k in ("R_follow_xy_pooled", "selection_acc", "loc_err_xy_cm_median")}
    res["retention_heldout"] = gb.eval_data(model, data, "dheld", swaps=True); res["retention_valD"] = gb.eval_data(model, data, "dvald"); torch.save(arrays, out_dir / f"{a.tag}_matcher_s{a.seed}_arrays.pt"); (out_dir / f"{a.tag}_smoke_s{a.seed}.json").write_text(json.dumps(res, indent=1, default=float))
    t = res["test"]; print("TEST R=%.3f acc=%.3f loc=%.1f | held-out id R=%.3f acc=%.3f" % (t["R_follow_xy_pooled"], t["selection_acc"], t["loc_err_xy_cm_median"], res["test_heldout_identity"]["R_follow_xy_pooled"], res["test_heldout_identity"]["selection_acc"]))
    print("train intra", json.dumps(res["train_intra"])[:500])
    for k in ("val", "test", "test_heldout_identity"): print(k, "cvc", res[k]["fine"]["card_vs_card"]["fine"], "intra", json.dumps(res[k]["intra"])[:600])


if __name__ == "__main__":
    main()
