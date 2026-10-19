#!/usr/bin/env python
"""G2 MIXED-DOMAIN referent grounding (fruit only). Base = G1 A3 recipe (explicit referent bottleneck, anchor + L_ref, NO L_cf), same
optimizer / lr / schedule / trainable modules / V2-20k initialisation / sim batch structure (16 scenes) as G1; the ONLY changed variable is
the dataset-domain share of each batch.
  sim samples    : anchor + 0.5 * L_ref        (counterfactual simulator data; referent-region token supervision)
  dataset samples: anchor only                 (original demonstrations, v2's own label; NO fake L_ref)
Mixing M1/M2/M3 = sim : dataset = 3:1 / 1:1 / 1:3 in SAMPLES per optimizer step (dataset samples = n_sim_samples * ratio), dataset frames drawn
equally from the EARLY / MID / NEAR k-buckets. Loss = sample-weighted mean over the combined batch.
Checkpoint selection (VAL only, fixed in advance): candidates that satisfy the retention guard on VAL-D in ALL of f0/k45/k30/k10
(median XY err <= V2 + max(2 cm, 25%)) are ranked by the G1 sim-val lexicographic rule (R_follow, selection acc, median err); if no candidate
satisfies the guard, the checkpoint with the smallest worst-bucket excess is taken. Held-out episodes and sim test are touched once."""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np, torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from g1_train import BETA, UNIT, Bottleneck, TaskData, V2Modules, eval_heldout_fruit, evaluate, huber, predict, select  # noqa: E402

OUTE = Path("outputs/eval/safediff_vla_g2_offline"); OUTT = Path("outputs/train/safediff_vla_g2")
RATIO = {"M1": 1 / 3, "M2": 1.0, "M3": 3.0}; BUCKETS = ("EARLY", "MID", "NEAR"); FRAMES = ("f0", "k45", "k30", "k10")
C = torch.load  # alias


@torch.no_grad()
def internals(model, tok, mod):
    valid = mod != 3; q = model.extractor(tok, valid, mod)
    if isinstance(model, Bottleneck):
        w, x = model.weights(tok, mod); f = (w.unsqueeze(-1) * model.Wv(x)).sum(1)
    else:
        img = (mod == 0) & valid; g = model.cross(q, tok, valid, mod); _, w = model.cross.attn(q.unsqueeze(1), tok, tok, key_padding_mask=~img, need_weights=True); w = w[:, 0, :192]; f = g
    return q, w, f


class DataCache:
    def __init__(self, path, dev):
        c = torch.load(path, weights_only=False); self.tok = c["tokens"].to(dev); self.mod = c["modality"].to(dev).long(); self.anc = c["anchors"].to(dev); self.meta = c["meta"]; self.dev = dev


def pad_to(tok, mod, n):
    if tok.shape[1] == n: return tok, mod
    p = n - tok.shape[1]; return F.pad(tok, (0, 0, 0, p)), F.pad(mod, (0, p), value=3)


@torch.no_grad()
def eval_frames(model, cache, mean, std, with_swaps):
    """per frame-type median/mean XY error (true text) [+ instruction-swap sensitivity]. cache.tok: [S,V,N,960]."""
    model.eval(); S, V = cache.tok.shape[:2]; P, Q, W, Fe = [], [], [], []
    for i in range(0, S, 16):
        t = cache.tok[i:i + 16, :V].float(); m = cache.mod[i:i + 16, :V]; s = t.shape[0]; t = t.reshape(s * V, *t.shape[2:]); m = m.reshape(s * V, -1)
        x, _ = model(t, m); P.append((x * std + mean).reshape(s, V, 3).cpu())
        if with_swaps: q, w, f = internals(model, t, m); Q.append(q.reshape(s, V, -1).cpu()); W.append(w.reshape(s, V, -1).cpu()); Fe.append(f.reshape(s, V, -1).cpu())
    P = torch.cat(P).numpy(); A = cache.anc.cpu().numpy(); fr = np.array([m["frame"] for m in cache.meta]); e = np.linalg.norm((P[:, 0] - A)[:, :2], axis=1) * 100
    out = {k: {"median_xy_cm": float(np.median(e[fr == k])), "mean_xy_cm": float(e[fr == k].mean()), "n": int((fr == k).sum())} for k in FRAMES if (fr == k).any()}
    out["pooled"] = {"median_xy_cm": float(np.median(e)), "mean_xy_cm": float(e.mean()), "n": int(len(e))}
    if with_swaps and V > 1:
        Q, W, Fe = torch.cat(Q), torch.cat(W), torch.cat(Fe); sens = {}
        for k in FRAMES:
            s = fr == k
            if not s.any(): continue
            sh = np.linalg.norm((P[s, 1:] - P[s, :1])[..., :2], axis=2).mean(1) * 100
            dq = ((Q[s, 1:] - Q[s, :1]).norm(dim=-1) / Q[s, :1].norm(dim=-1)).mean().item(); df_ = ((Fe[s, 1:] - Fe[s, :1]).norm(dim=-1) / Fe[s, :1].norm(dim=-1)).mean().item()
            dw = (1 - F.cosine_similarity(W[s, 1:], W[s, :1].expand_as(W[s, 1:]), dim=-1)).mean().item(); tv = (0.5 * (W[s, 1:] - W[s, :1]).abs().sum(-1)).mean().item()
            sens[k] = {"pred_shift_xy_cm_mean": float(sh.mean()), "pred_shift_xy_cm_median": float(np.median(sh)), "frac_shift_gt_30cm": float((sh > 30).mean()), "query_rel_change": dq,
                       "feature_rel_change": df_, "referent_map_cos_distance": dw, "referent_map_total_variation": tv,
                       "median_err_to_original_anchor_under_swap_cm": float(np.median(np.linalg.norm((P[s, 1:] - A[s, None])[..., :2], axis=2).mean(1)) * 100)}
        out["swap_sensitivity"] = sens
    return out


def guard_excess(err, ref):  # cm over the V2-relative allowance, worst bucket
    return max(err[k]["median_xy_cm"] - (ref[k]["median_xy_cm"] + max(2.0, 0.25 * ref[k]["median_xy_cm"])) for k in FRAMES if k in ref)


def sps_ids(sim, sp): return sim.idx(sp)


def run(mix, seed, steps, eval_every, sim, dtrain, dval, dheld, refs, consts):
    dev = sim.dev; torch.manual_seed(seed); np.random.seed(seed); model = Bottleneck().to(dev)
    opt = torch.optim.AdamW(model.groups((3e-4, 3e-5)), weight_decay=1e-4); base = [g["lr"] for g in opt.param_groups]
    tr = sim.idx("train"); rng = np.random.default_rng(seed); drng = np.random.default_rng(seed + 1000); order = rng.permutation(tr); pos = 0; B = 16
    bidx = {b: np.array([i for i, m in enumerate(dtrain.meta) if m["bucket"] == b]) for b in BUCKETS}
    mean, std = sim.mean, sim.std; hist = []; best = None; nsim_tot = ndat_tot = 0; bcount = {b: 0 for b in BUCKETS}
    for step in range(1, steps + 1):
        if pos + B > len(order): order = rng.permutation(tr); pos = 0
        ids = order[pos:pos + B]; pos += B; model.train()
        for g, b in zip(opt.param_groups, base): g["lr"] = b * min(1.0, step / 100)
        tok_s, mod_s, anc_s, valid_s, cf_s, held_s = sim.batch(ids); S, K = anc_s.shape[:2]; qm = (valid_s & ~held_s).reshape(-1)
        n_sim = int(qm.sum()); n_dat = int(round(n_sim * RATIO[mix])); per = [n_dat // 3 + (1 if i < n_dat % 3 else 0) for i in range(3)]
        sel = np.concatenate([drng.choice(bidx[b], size=per[i], replace=len(bidx[b]) < per[i]) for i, b in enumerate(BUCKETS)]) if n_dat else np.array([], int)
        for i, b in enumerate(BUCKETS): bcount[b] += per[i]
        nsim_tot += n_sim; ndat_tot += n_dat
        n = max(tok_s.shape[1], dtrain.tok.shape[2]); ts, ms = pad_to(tok_s, mod_s, n)
        if n_dat:
            td, md = pad_to(dtrain.tok[sel, 0].float(), dtrain.mod[sel, 0], n); tok = torch.cat([ts, td]); mod = torch.cat([ms, md])
        else: tok, mod = ts, ms
        xyz_n, w = model(tok, mod); pred = xyz_n * std + mean; ns = S * K
        la_s = huber((pred[:ns] - anc_s.reshape(ns, 3)) / UNIT).mean(-1); y = cf_s.reshape(ns, 192); has = (y.sum(-1) > 0) & qm
        lref = torch.zeros(ns, device=dev)
        if has.any(): yy = y[has] / y[has].sum(-1, keepdim=True); lref[has] = -(yy * torch.log(w[:ns][has] + 1e-8)).sum(-1)
        sim_term = ((la_s + 0.5 * lref)[qm]).sum()
        dat_term = huber((pred[ns:] - dtrain.anc[sel]) / UNIT).mean(-1).sum() if n_dat else torch.zeros((), device=dev)
        loss = (sim_term + dat_term) / (n_sim + n_dat)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % eval_every == 0:
            m = evaluate(model, sim, "val"); r = eval_frames(model, dval, mean, std, False)
            h = {"step": step, "R": m["R_follow_xy_pooled"], "acc": m["selection_acc"], "loc": m["loc_err_xy_cm_median"], "excess": guard_excess(r, refs["vald"]), "retention_valD": {k: r[k]["median_xy_cm"] for k in FRAMES if k in r}, "loss": loss.item()}
            hist.append(h)
            ok = [x for x in hist if x["excess"] <= 0]; pick = select(ok) if ok else min(hist, key=lambda x: (x["excess"], -x["R"]))
            if pick["step"] == step: best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ok = [x for x in hist if x["excess"] <= 0]; sel_h = select(ok) if ok else min(hist, key=lambda x: (x["excess"], -x["R"])); model.load_state_dict(best)
    heldout = evaluate(model, sim, "test_heldout_identity") if sim.task == "select_poker" else eval_heldout_fruit(model, sim)
    out = {"mix": mix, "ratio_dataset_over_sim": RATIO[mix], "seed": seed, "selected_step": sel_h["step"], "guard_satisfied_at_selected_on_valD": bool(ok), "n_candidates_satisfying_guard": len(ok),
           "val": evaluate(model, sim, "val"), "test": evaluate(model, sim, "test"), "heldout_identity": heldout,
           "retention_heldout": eval_frames(model, dheld, mean, std, True), "retention_valD": eval_frames(model, dval, mean, std, False), "history": hist,
           "batch_stats": {"mean_sim_samples_per_step": nsim_tot / steps, "mean_dataset_samples_per_step": ndat_tot / steps, "dataset_samples_per_bucket_total": bcount,
                           "dataset_samples_per_bucket_frac": {b: bcount[b] / max(1, sum(bcount.values())) for b in BUCKETS}}}
    OUTT.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), OUTT / f"{mix}_s{seed}.pt")
    sps = [sp for sp in ("val", "test", "test_heldout_identity") if len(sim.idx(sp))]
    torch.save({"preds": {sp: predict(model, sim, sim.idx(sp)) for sp in sps}, "ids": {sp: [sim.sids[i] for i in sps_ids(sim, sp)] for sp in sps}}, OUTT / f"{mix}_s{seed}_preds.pt")
    return out


def main():
    global OUTE, OUTT
    ap = argparse.ArgumentParser(); ap.add_argument("--mixes", nargs="+", default=["M1", "M2", "M3"]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--task", default="select_fruit"); ap.add_argument("--prefix", default="g2"); ap.add_argument("--out-eval", default=str(OUTE)); ap.add_argument("--out-train", default=str(OUTT)); a = ap.parse_args()
    OUTE, OUTT = Path(a.out_eval), Path(a.out_train); OUTE.mkdir(parents=True, exist_ok=True)
    sim = TaskData(a.task, "cuda"); dev = "cuda"; P = a.prefix
    dtrain = DataCache(OUTE / f"{P}_train_dataset_cache.pt", dev); dval = DataCache(OUTE / f"{P}_vald_cache.pt", dev); dheld = DataCache(OUTE / f"{P}_held_cache.pt", dev)
    refm = V2Modules().to(dev); refs = {"vald": eval_frames(refm, dval, sim.mean, sim.std, False)}
    print("V2 VAL-D reference median XY (cm):", {k: round(v["median_xy_cm"], 2) for k, v in refs["vald"].items() if "median_xy_cm" in v}, flush=True)
    for mix in a.mixes:
        for seed in a.seeds:
            o = run(mix, seed, a.steps, a.eval_every, sim, dtrain, dval, dheld, refs, None); (OUTE / f"{P}_{mix}_s{seed}.json").write_text(json.dumps(o, indent=1))
            t = o["test"]; r = o["retention_heldout"]
            print(f"{mix} s{seed} step={o['selected_step']} guardOK(valD)={o['guard_satisfied_at_selected_on_valD']}({o['n_candidates_satisfying_guard']}/{len(o['history'])}) | sim test R={t['R_follow_xy_pooled']:+.3f} acc={t['selection_acc']:.2f} loc={t['loc_err_xy_cm_median']:.1f} | held-out median XY cm: " + " ".join(f"{k}={r[k]['median_xy_cm']:.1f}" for k in (*FRAMES, "pooled")), flush=True)


if __name__ == "__main__":
    main()
