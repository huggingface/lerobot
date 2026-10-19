#!/usr/bin/env python
"""G1 offline prototype training on CACHED frozen-v2 VLM latents (no VLM forward, no env, no rollout, no decoder).
Arms (task-specific, poker / fruit):
  A0  v2 grounding modules (TargetQueryExtractor + VisualGroundingCrossAttention + TargetPointHead), warm-started, trained on
      the counterfactual data with anchor + cf loss                      [does the DATA alone fix it?]
  A1  explicit referent bottleneck + anchor + cf                         [is the bottleneck needed?]
  A2  A1 + L_ref (referent-region token supervision, sim segmentation, training only)
  A3  A2 without L_cf (anchor + L_ref only)                              [is the displacement loss needed?]
  V2  untouched v2 modules (reference only, not trained)
Checkpoint selection per run, on VAL only, lexicographic (never anchor error alone): max R_follow_xy; among those within 0.02,
max referent-selection accuracy; among those within 0.02, min median XY error. Test/held-out are evaluated once, at the selected step.
Units: losses in units of 10 cm, Huber(beta=0.2)."""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np, torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from g1_counterfactual_metrics import metrics  # noqa: E402

from lerobot.policies.safediff_vla.grounded_grasp_v2 import TargetQueryExtractor, VisualGroundingCrossAttention  # noqa: E402
from lerobot.policies.safediff_vla.target_point_head import TargetPointHead  # noqa: E402

D = Path("outputs/data/safediff_vla_g1_counterfactual"); OUTT = Path("outputs/train/safediff_vla_g1"); OUTE = Path("outputs/eval/safediff_vla_g1_offline")
V2_CKPT = "outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/020000/pretrained_model/model.safetensors"
HELD = {"select_poker": lambda n: n.split("_of_")[0] == "9", "select_fruit": lambda n: n in {"kiwi", "lemon"}}
UNIT, BETA = 0.1, 0.2
ARMS = {"A0": dict(model="v2", lam_cf=1.0, lam_ref=0.0), "A1": dict(model="bn", lam_cf=1.0, lam_ref=0.0),
        "A2": dict(model="bn", lam_cf=1.0, lam_ref=0.5), "A3": dict(model="bn", lam_cf=0.0, lam_ref=0.5)}


def v2_state(prefix):
    from safetensors.torch import load_file
    sd = load_file(V2_CKPT); return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class V2Modules(nn.Module):
    def __init__(self, warm=True):
        super().__init__()
        self.extractor = TargetQueryExtractor(960, 4); self.cross = VisualGroundingCrossAttention(960, 4); self.head = TargetPointHead(960, 256)
        if warm:
            self.extractor.load_state_dict(v2_state("target_query_extractor.")); self.cross.load_state_dict(v2_state("visual_grounding_cross_attn.")); self.head.load_state_dict(v2_state("target_point_head."))

    def forward(self, tok, mod):
        valid = mod != 3; q = self.extractor(tok, valid, mod); g = self.cross(q, tok, valid, mod)
        return self.head(g), None

    @torch.no_grad()
    def score_map(self, tok, mod):  # head-averaged cross-attention weight over the 192 image tokens (diagnostic only)
        valid = mod != 3; q = self.extractor(tok, valid, mod); img = (mod == 0) & valid
        _, w = self.cross.attn(q.unsqueeze(1), tok, tok, key_padding_mask=~img, need_weights=True)
        return w[:, 0, :192]

    def groups(self, lr): return [{"params": self.parameters(), "lr": lr[0]}]


class Bottleneck(nn.Module):
    def __init__(self, dk=128, dv=256):
        super().__init__()
        self.extractor = TargetQueryExtractor(960, 4); self.extractor.load_state_dict(v2_state("target_query_extractor."))
        self.ln_t, self.ln_q = nn.LayerNorm(960), nn.LayerNorm(960)
        self.Wk, self.Wq, self.Wv = nn.Linear(960, dk, bias=False), nn.Linear(960, dk, bias=False), nn.Linear(960, dv)
        self.bcam = nn.Parameter(torch.zeros(3)); self.dk = dk
        self.head = nn.Sequential(nn.Linear(dv + 9, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 3))
        i = torch.arange(192); self.register_buffer("cam", i // 64); self.register_buffer("row", ((i % 64) // 8).float() / 7); self.register_buffer("col", (i % 8).float() / 7)

    def weights(self, tok, mod):
        valid = mod != 3; q = self.extractor(tok, valid, mod); x = self.ln_t(tok[:, :192])
        s = (self.Wk(x) * self.Wq(self.ln_q(q)).unsqueeze(1)).sum(-1) / math.sqrt(self.dk) + self.bcam[self.cam]
        return F.softmax(s, -1), x

    def forward(self, tok, mod):
        w, x = self.weights(tok, mod); f = (w.unsqueeze(-1) * self.Wv(x)).sum(1); feats = []
        for c in range(3):
            m = (self.cam == c).float(); feats += [(w * m).sum(-1), (w * m * self.row).sum(-1), (w * m * self.col).sum(-1)]
        return self.head(torch.cat([f, torch.stack(feats, -1)], -1)), w

    @torch.no_grad()
    def score_map(self, tok, mod): return self.weights(tok, mod)[0]

    def groups(self, lr):
        ext = list(self.extractor.parameters()); ids = {id(p) for p in ext}
        return [{"params": ext, "lr": lr[1]}, {"params": [p for p in self.parameters() if id(p) not in ids], "lr": lr[0]}]


class TaskData:
    def __init__(self, task, dev):
        c = torch.load(D / task / "latent_cache.pt"); self.task = task; self.dev = dev
        self.tok = c["tokens"]; self.mod = c["modality"]; self.anc = c["anchors"]; self.valid = c["valid"]; self.cf = c["cellfrac"]
        self.names = c["names"]; self.sids = c["scene_ids"]; self.split = np.array(c["split"]); self.mean = torch.tensor(c["consts"]["pos_mean"]); self.std = torch.tensor(c["consts"]["pos_std"])
        self.held = torch.tensor([[bool(k < len(n) and HELD[task](n[k])) for k in range(4)] for n in self.names])
        assert (self.mod[:, :, :192][self.valid] == 0).all(), "first 192 tokens must be IMAGE tokens"
        self.tok = self.tok.to(dev); self.mod = self.mod.to(dev).long(); self.anc = self.anc.to(dev); self.valid = self.valid.to(dev); self.cf = self.cf.to(dev); self.held = self.held.to(dev)
        self.mean, self.std = self.mean.to(dev), self.std.to(dev)

    def idx(self, split): return np.where(self.split == split)[0]

    def qmask(self, split):  # which referent queries are used
        return self.valid & (self.held if split in ("test_heldout_identity",) else ~self.held)

    def batch(self, ids):
        t = torch.as_tensor(ids, device=self.dev); S, K, N, Dm = self.tok[t].shape
        return self.tok[t].float().reshape(S * K, N, Dm), self.mod[t].reshape(S * K, N), self.anc[t], self.valid[t], self.cf[t], self.held[t]


def huber(x): return F.smooth_l1_loss(x, torch.zeros_like(x), beta=BETA, reduction="none")


def compute_loss(model, data, ids, qmask_fn, lam_cf, lam_ref):
    tok, mod, anc, valid, cf, held = data.batch(ids); S, K = anc.shape[:2]
    xyz_n, w = model(tok, mod); pred = (xyz_n * data.std + data.mean).reshape(S, K, 3); qm = valid & ~held
    la = huber((pred - anc) / UNIT)[qm].mean(); lcf = torch.zeros((), device=anc.device); lref = torch.zeros((), device=anc.device)
    if lam_cf > 0:
        dp = (pred.unsqueeze(2) - pred.unsqueeze(1))[..., :2]; dg = (anc.unsqueeze(2) - anc.unsqueeze(1))[..., :2]
        pm = qm.unsqueeze(2) & qm.unsqueeze(1) & torch.triu(torch.ones(K, K, dtype=torch.bool, device=anc.device), 1)
        if pm.any(): lcf = huber((dp - dg) / UNIT)[pm].mean()
    if lam_ref > 0 and w is not None:
        y = cf.reshape(S * K, 192); has = (y.sum(-1) > 0) & qm.reshape(-1)
        if has.any(): y = y[has] / y[has].sum(-1, keepdim=True); lref = -(y * torch.log(w[has] + 1e-8)).sum(-1).mean()
    return la + lam_cf * lcf + lam_ref * lref, {"anchor": la.item(), "cf": lcf.item(), "ref": lref.item()}


@torch.no_grad()
def predict(model, data, ids, bs=16):
    model.eval(); P, W = [], []
    for i in range(0, len(ids), bs):
        tok, mod, anc, *_ = data.batch(ids[i:i + bs]); S, K = anc.shape[:2]
        xyz_n, w = model(tok, mod); P.append((xyz_n * data.std + data.mean).reshape(S, K, 3).cpu())
        if w is None: w = model.score_map(tok, mod)
        W.append(w.reshape(S, K, 192).cpu())
    return torch.cat(P), torch.cat(W)


def evaluate(model, data, split, preds=None):
    ids = data.idx(split); P, W = preds if preds is not None else predict(model, data, ids)
    held_split = split == "test_heldout_identity"; hf = HELD[data.task]; scenes = []; loc_hit = []; mass = []
    qm = data.qmask("test" if split == "val" else split).cpu(); anc = data.anc.cpu()
    for j, s in enumerate(ids):
        names = data.names[s]; ks = [k for k in range(len(names)) if bool(data.valid[s, k].item())]
        if not held_split: ks_q = [k for k in ks if not hf(names[k])]
        else: ks_q = ks
        scenes.append({"anchors": {names[k]: anc[s, k].numpy() for k in ks_q}, "preds": {names[k]: P[j, k].numpy() for k in ks_q}})
        for k in ks_q:
            y = data.cf[s, k].cpu(); 
            if y.sum() > 0: loc_hit.append(float(y[W[j, k].argmax()] > 0)); mass.append(float((W[j, k] * (y > 0)).sum()))
    scenes = [sc for sc in scenes if len(sc["anchors"]) >= 2]
    if split == "test_heldout_identity" or (data.task == "select_fruit" and split == "test_heldout_identity"):
        pass
    m = metrics(scenes, pair_ok=(lambda a, b: hf(a) or hf(b)) if held_split else None, query_ok=(lambda c: hf(c)) if held_split else None)
    m["referent_token_top1_in_mask"] = float(np.mean(loc_hit)) if loc_hit else None; m["referent_token_mass_in_mask"] = float(np.mean(mass)) if mass else None
    return m


def eval_heldout_fruit(model, data):
    """fruit held-out identity: test-split scenes, pairs/queries involving kiwi/lemon (never queried in training)."""
    ids = data.idx("test"); P, W = predict(model, data, ids); hf = HELD[data.task]; scenes = []
    anc = data.anc.cpu()
    for j, s in enumerate(ids):
        names = data.names[s]; ks = [k for k in range(len(names)) if bool(data.valid[s, k].item())]
        scenes.append({"anchors": {names[k]: anc[s, k].numpy() for k in ks}, "preds": {names[k]: P[j, k].numpy() for k in ks}})
    scenes = [sc for sc in scenes if any(hf(n) for n in sc["anchors"]) and len(sc["anchors"]) >= 2]
    return metrics(scenes, pair_ok=lambda a, b: hf(a) or hf(b), query_ok=hf)


def select(hist):
    best = max(h["R"] for h in hist); c = [h for h in hist if h["R"] >= best - 0.02]
    ba = max(h["acc"] for h in c); c = [h for h in c if h["acc"] >= ba - 0.02]
    return min(c, key=lambda h: h["loc"])


def run(task, arm, seed, steps, eval_every, data, lr_new=3e-4, lr_ext=3e-5, lr_v2=1e-4):
    cfg = ARMS[arm]; torch.manual_seed(seed); np.random.seed(seed); dev = data.dev
    model = (V2Modules() if cfg["model"] == "v2" else Bottleneck()).to(dev)
    opt = torch.optim.AdamW(model.groups((lr_v2, lr_v2) if cfg["model"] == "v2" else (lr_new, lr_ext)), weight_decay=1e-4)
    base = [g["lr"] for g in opt.param_groups]; tr = data.idx("train"); rng = np.random.default_rng(seed); hist = []; best_state = None; B = 16
    order = rng.permutation(tr); pos = 0
    for step in range(1, steps + 1):
        if pos + B > len(order): order = rng.permutation(tr); pos = 0
        ids = order[pos:pos + B]; pos += B; model.train()
        for g, b in zip(opt.param_groups, base): g["lr"] = b * min(1.0, step / 100)
        loss, parts = compute_loss(model, data, ids, None, cfg["lam_cf"], cfg["lam_ref"])
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % eval_every == 0:
            m = evaluate(model, data, "val"); h = {"step": step, "R": m["R_follow_xy_pooled"], "acc": m["selection_acc"], "loc": m["loc_err_xy_cm_median"], "loss": loss.item(), **parts}
            hist.append(h)
            if best_state is None or select(hist)["step"] == step: best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sel = select(hist); model.load_state_dict(best_state)
    out = {"task": task, "arm": arm, "seed": seed, "selected_step": sel["step"], "val": evaluate(model, data, "val"), "test": evaluate(model, data, "test"), "history": hist}
    out["heldout_identity"] = evaluate(model, data, "test_heldout_identity") if task == "select_poker" else eval_heldout_fruit(model, data)
    (OUTT / task).mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), OUTT / task / f"{arm}_s{seed}.pt")
    preds = {sp: predict(model, data, data.idx(sp)) for sp in ("val", "test", "test_heldout_identity") if len(data.idx(sp))}
    torch.save({"preds": preds, "ids": {sp: [data.sids[i] for i in data.idx(sp)] for sp in preds}}, OUTT / task / f"{arm}_s{seed}_preds.pt")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--arms", nargs="+", default=["A0", "A1", "A2", "A3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--reference", action="store_true"); a = ap.parse_args()
    data = TaskData(a.task, "cuda"); OUTE.mkdir(parents=True, exist_ok=True); res = []
    if a.reference:  # untouched v2 modules on the same splits
        m = V2Modules().cuda(); r = {"task": a.task, "arm": "V2", "seed": None, "val": evaluate(m, data, "val"), "test": evaluate(m, data, "test"),
             "heldout_identity": evaluate(m, data, "test_heldout_identity") if a.task == "select_poker" else eval_heldout_fruit(m, data)}
        (OUTE / f"{a.task}_V2_reference.json").write_text(json.dumps(r, indent=1)); print("V2 ref", {s: (round(r[s]["R_follow_xy_pooled"], 3), round(r[s]["selection_acc"], 2), round(r[s]["loc_err_xy_cm_median"], 1)) for s in ("val", "test", "heldout_identity")}, flush=True)
    for arm in a.arms:
        for seed in a.seeds:
            out = run(a.task, arm, seed, a.steps, a.eval_every, data); res.append(out)
            (OUTE / f"{a.task}_{arm}_s{seed}.json").write_text(json.dumps(out, indent=1))
            v, t = out["val"], out["test"]
            print(f"{a.task} {arm} s{seed} step={out['selected_step']} | val R={v['R_follow_xy_pooled']:+.3f} acc={v['selection_acc']:.2f} loc={v['loc_err_xy_cm_median']:.1f} | test R={t['R_follow_xy_pooled']:+.3f} {t['R_follow_ci95_scene_bootstrap']} acc={t['selection_acc']:.2f} loc={t['loc_err_xy_cm_median']:.1f}", flush=True)


if __name__ == "__main__":
    main()
