#!/usr/bin/env python
"""G3 fine-grained identity diagnostics (poker). DIAGNOSTIC ONLY - nothing here feeds back into training or model selection.

(A) SELECTOR error anatomy for the trained G3 models (per seed): for each query card, the card the model 'picked' (a) by predicted anchor (nearest candidate
    anchor) and (b) by referent score-map mass (sum of map weights x GT cell coverage of each candidate). Outcome = exact / same rank wrong suit / same suit wrong rank /
    both wrong. rank accuracy = P(picked rank == target rank), suit accuracy likewise; compared with the expectation of a uniformly random pick among the scene's
    candidates (computed per query from the scene's actual rank/suit composition). Also on HARD-NEGATIVE queries (>=1 distractor sharing rank or suit).
(B) REPRESENTATION probes on FROZEN latents, no new model: card-region features = image-token latents pooled with the card's GT segmentation coverage.
    Latents come from the NEUTRAL instruction ("Please pick the poker", names no card) so identity can only be visual. Scene-level split (train scenes / val / test);
    rank (13), suit (4), identity (52) classification with L2 logistic regression (C picked on the val scenes). Controls: (i) raw-pixel crop probe at the card
    (is identity even resolvable at the camera resolution?), (ii) self-query latents (text names the card -> upper bound incl. text binding). Cosine similarity of
    region features across scenes: same card / same rank / same suit / different."""
from __future__ import annotations

import json, sys
from pathlib import Path

import cv2, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from g1_train import D, TaskData  # noqa: E402

OUTE = Path("outputs/eval/safediff_vla_g3_offline"); OUTT = Path("outputs/train/safediff_vla_g3")
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king", "ace"]; SUITS = ["spades", "clubs", "diamonds", "hearts"]
def rs(n): r, s = n.split("_of_"); return r, s


def selector_analysis(data, P, W, split):
    ids = data.idx(split); cf = data.cf.cpu().numpy(); anc = data.anc.cpu().numpy(); out = {"anchor": {}, "map": {}}
    for mode in out:
        for subset in ("all", "hard_negative", "seen_rank", "heldout_rank9"):
            out[mode][subset] = {"n": 0, "exact": 0, "same_rank_wrong_suit": 0, "same_suit_wrong_rank": 0, "both_wrong": 0, "exp_exact": 0.0, "exp_rank_match": 0.0, "exp_suit_match": 0.0}
    for j, s in enumerate(ids):
        names = data.names[s]; ks = [k for k in range(len(names)) if bool(data.valid[s, k])]; K = len(ks)
        for k in ks:
            held = bool(data.held[s, k]); 
            if split != "test_heldout_identity" and held: continue
            tr, ts = rs(names[k]); others = [c for c in ks if c != k]
            n_sr = sum(rs(names[c])[0] == tr for c in others); n_ss = sum(rs(names[c])[1] == ts for c in others)
            d = {c: np.linalg.norm((P[j, k].numpy() - anc[s, c])[:2]) for c in ks}; picks = {"anchor": min(d, key=d.get), "map": max(ks, key=lambda c: float((W[j, k].numpy() * cf[s, c]).sum()))}
            for mode, c in picks.items():
                pr, ps = rs(names[c]); cat = "exact" if c == k else ("same_rank_wrong_suit" if pr == tr else ("same_suit_wrong_rank" if ps == ts else "both_wrong"))
                for subset in ("all", *(["hard_negative"] if (n_sr or n_ss) else []), *(["heldout_rank9"] if held else ["seen_rank"])):
                    o = out[mode][subset]; o["n"] += 1; o[cat] += 1; o["exp_exact"] += 1 / K; o["exp_rank_match"] += (1 + n_sr) / K; o["exp_suit_match"] += (1 + n_ss) / K
    for mode in out:
        for subset, o in out[mode].items():
            n = o["n"]
            if n: o.update({"exact_acc": o["exact"] / n, "rank_acc": (o["exact"] + o["same_rank_wrong_suit"]) / n, "suit_acc": (o["exact"] + o["same_suit_wrong_rank"]) / n,
                            "chance_exact": o["exp_exact"] / n, "chance_rank_acc": o["exp_rank_match"] / n, "chance_suit_acc": o["exp_suit_match"] / n,
                            "frac_wrong_that_are_same_rank": o["same_rank_wrong_suit"] / max(1, n - o["exact"]), "frac_wrong_that_are_same_suit": o["same_suit_wrong_rank"] / max(1, n - o["exact"])})
    return out


def probe(Xtr, ytr, Xva, yva, Xte, yte, Cs=(0.001, 0.01, 0.1, 1.0)):
    sc = StandardScaler().fit(Xtr); best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr); a = (clf.predict(sc.transform(Xva)) == yva).mean()
        if best is None or a > best[0]: best = (a, C, clf)
    return {"C": best[1], "val_acc": float(best[0]), "test_acc": float((best[2].predict(sc.transform(Xte)) == yte).mean()), "n_train": len(ytr), "n_test": len(yte)}


def main():
    data = TaskData("select_poker", "cpu"); S = len(data.sids); res = {}
    # ---------- (A) selector anatomy ----------
    res["selector"] = {}
    for p in sorted(OUTT.glob("M1_s?_preds.pt")):
        seed = int(p.stem.split("_s")[1][0]); Z = torch.load(p, weights_only=False); res["selector"][f"seed{seed}"] = {}
        for sp in ("test", "test_heldout_identity", "val"):
            if sp in Z["preds"]: res["selector"][f"seed{seed}"][sp] = selector_analysis(data, *Z["preds"][sp], sp)
    # ---------- (B) representation ----------
    N = torch.load(OUTE / "g3_neutral_image_latents.pt", weights_only=False); assert N["scene_ids"] == data.sids
    X = N["tokens"].float().numpy(); cf = data.cf.cpu().numpy(); tokq = data.tok  # [S,4,N,960] fp16 (cpu)
    rows = []  # (scene, k, rank, suit, split)
    for s in range(S):
        for k in range(len(data.names[s])): rows.append((s, k, RANKS.index(rs(data.names[s][k])[0]), SUITS.index(rs(data.names[s][k])[1]), data.split[s]))
    sc_idx = np.array([r[0] for r in rows]); kk = np.array([r[1] for r in rows]); rk = np.array([r[2] for r in rows]); su = np.array([r[3] for r in rows]); spl = np.array([r[4] for r in rows])
    ident = rk * 4 + su
    def region_feat(src):
        out = np.zeros((len(rows), 960), np.float32)
        for i, (s, k) in enumerate(zip(sc_idx, kk)):
            w = cf[s, k]; w = w / max(w.sum(), 1e-9); f = src(s, k); out[i] = (w[:, None] * f).sum(0)
        return out
    F_neutral = region_feat(lambda s, k: X[s]); F_self = region_feat(lambda s, k: tokq[s, k, :192].float().numpy())
    # raw pixel crops
    import json as _j
    crops = np.zeros((len(rows), 2 * 20 * 20 * 3), np.float32); cache = {}
    for i, (s, k) in enumerate(zip(sc_idx, kk)):
        sid = data.sids[s]
        if sid not in cache: cache[sid] = (np.load(D / "select_poker" / f"{sid}.npz")["rgb"], _j.load(open(D / "select_poker" / f"{sid}.json")))
        rgb, rec = cache[sid]; parts = []
        for cam in (0, 1):
            u, v = rec["candidates"][k]["proj_px_per_camera"][cam]; u, v = int(np.clip(u, 20, 460)), int(np.clip(v, 20, 460))
            parts.append(cv2.resize(rgb[cam][v - 20:v + 20, u - 20:u + 20], (20, 20), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0)
        crops[i] = np.concatenate(parts)
    tr = spl == "train"; va = spl == "val"; te = spl == "test"; held = spl == "test_heldout_identity"
    res["probe"] = {}
    for name, F in (("frozen_latent_neutral_text", F_neutral), ("frozen_latent_self_query_text", F_self), ("raw_pixel_crop_front+right", crops)):
        res["probe"][name] = {}
        for lab, y, chance in (("rank", rk, 1 / 13), ("suit", su, 0.25), ("identity", ident, 1 / 48)):
            r = probe(F[tr], y[tr], F[va], y[va], F[te], y[te]); r["chance"] = chance; res["probe"][name][lab] = r
        # suit probe also on held-out-rank scenes (suit is a seen label there)
        sc = StandardScaler().fit(F[tr]); clf = LogisticRegression(C=res["probe"][name]["suit"]["C"], max_iter=3000).fit(sc.transform(F[tr]), su[tr])
        res["probe"][name]["suit"]["acc_on_heldout_rank9_scenes"] = float((clf.predict(sc.transform(F[held])) == su[held]).mean())
    # cosine similarity across scenes (neutral latents)
    G = F_neutral / np.linalg.norm(F_neutral, axis=1, keepdims=True); Cm = G @ G.T; diff_scene = sc_idx[:, None] != sc_idx[None, :]
    same_id = ident[:, None] == ident[None, :]; same_rank = (rk[:, None] == rk[None, :]) & ~same_id; same_suit = (su[:, None] == su[None, :]) & ~same_id; neither = (rk[:, None] != rk[None, :]) & (su[:, None] != su[None, :])
    Fc = F_neutral.copy()
    for s in range(S): m = sc_idx == s; Fc[m] -= Fc[m].mean(0)
    Gc = Fc / np.maximum(np.linalg.norm(Fc, axis=1, keepdims=True), 1e-9); Cc = Gc @ Gc.T
    res["similarity_cross_scene_neutral_latents"] = {"raw": {n: float(Cm[m & diff_scene].mean()) for n, m in (("same_card", same_id), ("same_rank_diff_suit", same_rank), ("same_suit_diff_rank", same_suit), ("different_card", neither))},
                                                      "scene_mean_removed": {n: float(Cc[m & diff_scene].mean()) for n, m in (("same_card", same_id), ("same_rank_diff_suit", same_rank), ("same_suit_diff_rank", same_suit), ("different_card", neither))},
                                                      "n_pairs": {n: int((m & diff_scene).sum() // 2) for n, m in (("same_card", same_id), ("same_rank_diff_suit", same_rank), ("same_suit_diff_rank", same_suit), ("different_card", neither))}}
    res["region_stats"] = {"median_cells_covered_per_card(of 192)": float(np.median((cf[sc_idx, kk] > 0).sum(1))), "median_card_pixels_front_cam(480x480)": 850}
    (OUTE / "g3_identity_diagnostic.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "selector"}, indent=1))


if __name__ == "__main__":
    main()
