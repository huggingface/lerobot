#!/usr/bin/env python
"""[StagePlan-v2.1HR] Referent-ONLY candidate scorer, Poker+Mahjong, exact failed-v2.1 40/10 scene split. Frozen backbone; no executor/closed-loop.
PRE-DECLARED: seed 1000, 3000 steps, batch 32 (instruction samples), AdamW lr 3e-4 wd 1e-2, dropout 0.1, FINAL-step model; one run per feature variant, no sweeps.
 candidate proposals : simulator instance metadata (TRAIN/source-domain only): projected object centre in the front camera (verified overlays)
 variant A coarse    : post-VLM full-frame 8x8 token of the front camera at the candidate's cell (what v2.1 could see)
 variant B high-res  : CROP_PX window of the ORIGINAL 480px front image around the candidate -> 512 (backbone input size) -> frozen vision tower + connector -> 64 local tokens
 variant C pixels    : crop downsampled to 16x16x3 (no network features)
 language            : frozen post-VLM text-token states from the cached prefix (tokenizer output), learned attention pooling -- no per-task parsing
 scorer              : s_i = <Wq q, Wv v_i>/sqrt(d); L_ref = CE over the scene's valid candidates.
Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/v21hr_candidate.py"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from v21_common import OUT, load_policy  # noqa: E402
from v21_referent_table import cam_params, project  # noqa: E402

TASKS = ("select_poker", "select_mahjong"); CROP_PX = 72; D = 128; HR = OUT / "hr"; HR.mkdir(exist_ok=True)
STEPS, BATCH, SEED = 3000, 32, 1000


@torch.no_grad()
def extract(pol, rows):
    cam0 = cam_params("select_poker")[0]; ent = {}
    ve = pol.backbone.model.vlm_with_expert
    for r in rows:
        if r["task"] not in TASKS or not r["scene_ok"]: continue
        for ci, c in enumerate(r["candidates"]):
            u, v, _ = project(cam0, c["obj_center"] + r["base"]); u, v = int(round(u)), int(round(v)); h = CROP_PX // 2
            im = r["imgs480"][0]; pad = np.pad(im, ((h, h), (h, h), (0, 0)), mode="edge"); crop = pad[v: v + CROP_PX, u: u + CROP_PX]  # window centred at (u,v) in padded coords
            inframe = h <= u < 480 - h and h <= v < 480 - h
            x = torch.from_numpy(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1).float().div(255)[None].cuda() * 2 - 1
            toks = ve.embed_image(x)[0].float().cpu()  # [64,960] local tokens of the crop
            pix = torch.from_numpy(cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA)).float().div(255).reshape(-1)
            ent[(r["scene_id"], ci)] = {"crop": crop.copy(), "toks": toks.half(), "pix": pix, "uv": (u, v), "inframe": bool(inframe)}
    return ent


def build_samples(rows, split, cache, ent):
    cb = {(e["scene_id"], e["ci"]): e for e in cache}; S = []
    for r in rows:
        if r["task"] not in TASKS or not r["scene_ok"]: continue
        ids = [i for i, c in enumerate(r["candidates"]) if c["valid"]]
        if len(ids) < 2: continue
        for ti in ids:
            e = cb[(r["scene_id"], ti)]; txt = (e["mod"] == 1) & e["pad"]
            coarse = []
            for i in ids:
                u, v = ent[(r["scene_id"], i)]["uv"]; cc, rr = min(max(int(u / 60), 0), 7), min(max(int(v / 60), 0), 7); coarse.append(e["tok"][rr * 8 + cc].float())  # front cam = tokens 0..63
            S.append({"scene": r["scene_id"], "task": r["task"], "split": split[r["scene_id"]], "ids": ids, "t": ids.index(ti), "txt": e["tok"][txt].float(), "A": torch.stack(coarse),
                      "B": torch.stack([ent[(r["scene_id"], i)]["toks"].float() for i in ids]), "C": torch.stack([ent[(r["scene_id"], i)]["pix"] for i in ids])})
    return S


class Scorer(nn.Module):
    def __init__(self, variant):
        super().__init__(); self.v = variant; ind = {"A": 960, "B": 960, "C": 768}[variant]
        self.txt = nn.Linear(960, D); self.tq = nn.Parameter(torch.randn(1, 1, D) * .02); self.tattn = nn.MultiheadAttention(D, 4, batch_first=True, dropout=0.1)
        self.vis = nn.Sequential(nn.LayerNorm(ind), nn.Linear(ind, D)); self.drop = nn.Dropout(0.1)
        if variant == "B": self.pos = nn.Parameter(torch.randn(64, D) * .02); self.vq = nn.Parameter(torch.randn(1, 1, D) * .02); self.vattn = nn.MultiheadAttention(D, 4, batch_first=True, dropout=0.1)
        self.wq, self.wv = nn.Linear(D, D), nn.Linear(D, D)
    def forward(self, txt, tmask, feats):  # txt [B,T,960]; feats: A [B,K,960] / B [B,K,64,960] / C [B,K,768]
        t = self.txt(txt); q, _ = self.tattn(self.tq.expand(len(txt), -1, -1), t, t, key_padding_mask=~tmask, need_weights=False)
        if self.v == "B":
            Bn, K = feats.shape[:2]; v = self.vis(feats.reshape(Bn * K, 64, 960)) + self.pos; v, _ = self.vattn(self.vq.expand(Bn * K, -1, -1), v, v, need_weights=False); v = v.reshape(Bn, K, D)
        else: v = self.vis(feats)
        return torch.einsum("bd,bkd->bk", self.wq(self.drop(q.squeeze(1))), self.wv(self.drop(v))) / D ** 0.5


def collate(S, v):
    T = max(s["txt"].shape[0] for s in S); K = max(len(s["ids"]) for s in S); B = len(S)
    txt = torch.zeros(B, T, 960); tm = torch.zeros(B, T, dtype=torch.bool); cm = torch.zeros(B, K, dtype=torch.bool); tgt = torch.tensor([s["t"] for s in S])
    f = torch.zeros(B, K, *S[0][v].shape[1:])
    for b, s in enumerate(S): txt[b, :len(s["txt"])] = s["txt"]; tm[b, :len(s["txt"])] = True; cm[b, :len(s["ids"])] = True; f[b, :len(s["ids"])] = s[v]
    return txt.cuda(), tm.cuda(), f.cuda(), cm.cuda(), tgt.cuda()


def evaluate(model, S, v):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(S), 64):
            ch = S[i:i + 64]; txt, tm, f, cm, tgt = collate(ch, v); lg = model(txt, tm, f).masked_fill(~cm, -1e9); p = torch.softmax(lg, -1)
            for b, s in enumerate(ch): out.append({"scene": s["scene"], "task": s["task"], "t": s["t"], "n": len(s["ids"]), "ids": s["ids"], "p": p[b, :len(s["ids"])].cpu().numpy()})
    return out


def metrics(out, which, rows_by):
    res = {}
    for task in TASKS:
        O = [o for o in out if o["task"] == task]
        if not O: continue
        acc = np.mean([o["p"].argmax() == o["t"] for o in O]); top2 = np.mean([o["t"] in np.argsort(-o["p"])[:2] for o in O]); ch = np.mean([1 / o["n"] for o in O]); marg = np.mean([o["p"][o["t"]] - np.max(np.delete(o["p"], o["t"])) for o in O])
        # same-scene swap: for each ordered pair of instructions (i,j) of a scene, both predicted correctly
        by = {}
        for o in O: by.setdefault(o["scene"], {})[o["t"]] = o
        sw, chg, npair = 0, 0, 0
        for sc, d in by.items():
            for a in d:
                for b in d:
                    if a == b: continue
                    npair += 1; pa, pb = d[a]["p"].argmax(), d[b]["p"].argmax(); sw += (pa == a) and (pb == b); chg += pa != pb
        K = max(o["n"] for o in O); cmx = np.zeros((K, K), int)
        for o in O: cmx[o["t"], o["p"].argmax()] += 1
        res[task] = {"n": len(O), "acc": float(acc), "chance": float(ch), "top2": float(top2), "margin": float(marg), "swap_both_correct": float(sw / max(npair, 1)), "pred_changes_on_swap": float(chg / max(npair, 1)), "confusion_true_row_pred_col(slot=left->right order of listing)": cmx.tolist()}
    return res


def main():
    rows = pickle.load(open(OUT / "referent_table.pkl", "rb")); split = json.load(open(OUT / "pilot_scene_split.json")); cache = torch.load(OUT / "pilot_token_cache.pt", weights_only=False)
    pol, _ = load_policy(); ent = extract(pol, rows); pickle.dump({k: {"crop": v["crop"], "uv": v["uv"], "inframe": v["inframe"]} for k, v in ent.items()}, open(HR / "crops.pkl", "wb"))
    cov = {t: {"candidates": sum(1 for r in rows if r["task"] == t for c in r["candidates"]), "valid": sum(c["valid"] for r in rows if r["task"] == t for c in r["candidates"]), "crop_inframe": sum(ent[(r["scene_id"], i)]["inframe"] for r in rows if r["task"] == t for i, c in enumerate(r["candidates"]))} for t in TASKS}; print("coverage", cov)
    S = build_samples(rows, split, cache, ent); tr = [s for s in S if s["split"] == "train"]; va = [s for s in S if s["split"] == "val"]; print("train samples", len(tr), "val samples", len(va), "token res: crop 64 local tokens (8x8 over 72px) vs coarse 1 token per 60px cell")
    res = {"coverage": cov, "crop_px": CROP_PX}; rb = {r["scene_id"]: r for r in rows}
    for v in ("A", "B", "C"):
        torch.manual_seed(SEED); rng = np.random.default_rng(SEED); m = Scorer(v).cuda(); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-2); log = []
        for step in range(1, STEPS + 1):
            m.train(); idx = rng.choice(len(tr), BATCH, replace=False); txt, tm, f, cm, tgt = collate([tr[i] for i in idx], v); loss = F.cross_entropy(m(txt, tm, f).masked_fill(~cm, -1e9), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 500 == 0:
                a_tr = np.mean([o["p"].argmax() == o["t"] for o in evaluate(m, tr, v)]); a_va = np.mean([o["p"].argmax() == o["t"] for o in evaluate(m, va, v)]); log.append((step, float(loss), float(a_tr), float(a_va))); print(v, log[-1], flush=True)
        otr, ova = evaluate(m, tr, v), evaluate(m, va, v); res[v] = {"train": metrics(otr, "train", rb), "val": metrics(ova, "val", rb), "log": log}
        if v == "B": pickle.dump(ova, open(HR / "val_scores_B.pkl", "wb"))
    json.dump(res, open(OUT / "hr_results.json", "w"), indent=1, default=float)
    for v in "ABC":
        for t in TASKS: m_tr, m_va = res[v]["train"][t], res[v]["val"][t]; print(f"{v} {t:15s} train acc {m_tr['acc']:.3f} | val acc {m_va['acc']:.3f} chance {m_va['chance']:.3f} top2 {m_va['top2']:.3f} margin {m_va['margin']:.3f} swap_ok {m_va['swap_both_correct']:.3f} pred_changes {m_va['pred_changes_on_swap']:.3f} (n={m_va['n']})")

if __name__ == "__main__": main()
