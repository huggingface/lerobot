#!/usr/bin/env python
"""G4b STEP 0b identity sanity: frozen crop encoder (vision tower + connector, mean of 64 tokens), FRONT crops only. No training of any network (L2 logistic probes only, same probe as G3/G4-A).
(a) GT-cell-snapped centred crop (inference-feasible centre: the 60px coarse cell containing the card), 96 vs 144.
(b) the ACTUAL proposed candidate crop that contains the target (old96 cell-centre list vs new144 centroid list, M1 maps), probe trained on the (a) train-scene features of the same size.
Reference: full-frame pooled identity ~0.08 (G3), 96px GT-centred C1_front 0.65, 96px cell-snapped 0.452."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from g1_train import D, TaskData  # noqa: E402
from g3_identity_diagnostic import RANKS, SUITS, probe, rs  # noqa: E402
import g4b_proposals as P  # noqa: E402
from lerobot.policies.common.vla_utils import resize_with_pad  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402

PD = D / "select_poker"; OUT = Path("outputs/eval/safediff_vla_g4b_offline")


def main():
    data = TaskData("select_poker", "cpu"); B = {}
    for p in sorted(PD.glob("bboxes_*.json")): B.update(json.load(open(p)))
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); vlm = policy.backbone.model.vlm_with_expert; scale = math.sqrt(960)

    @torch.no_grad()
    def embed(imgs):
        x = torch.cat([resize_with_pad(torch.from_numpy(i).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0, 512, 512, pad_value=0) * 2.0 - 1.0 for i in imgs]); return (vlm.embed_image(x).float() * scale).mean(1).cpu().numpy()

    Z = torch.load("outputs/train/safediff_vla_g3/M1_s0_preds.pt", weights_only=False); rows = []; feats = {96: [], 144: []}; cand = {"old96": [], "new144": []}
    qpred = {}
    for sp in ("test", "test_heldout_identity"):
        for j, s in enumerate(data.idx(sp)): qpred[(s, sp)] = (j, Z["preds"][sp][1])
    for s, sid in enumerate(data.sids):
        rgb = np.load(PD / f"{sid}.npz")["rgb"][0]; names = data.names[s]
        for k in range(len(names)):
            if not data.valid[s, k]: continue
            tb = B[sid]["boxes_xyxy_per_camera"][names[k]][0]; cx, cy = (int(((tb[0] + tb[2]) / 2) // 60) + .5) * 60, (int(((tb[1] + tb[3]) / 2) // 60) + .5) * 60; r, su = rs(names[k]); rows.append((s, k, RANKS.index(r), SUITS.index(su), data.split[s], bool(data.held[s, k])))
            for side in (96, 144):
                x0, y0, _ = P.front_box(cx, cy, side); feats[side].append(embed([rgb[y0:y0 + side, x0:x0 + side]])[0])
            # candidate crops (test / held-out splits; query = this card, coarse map from final M1)
            sp = data.split[s]
            if sp in ("test", "test_heldout_identity") and (s, sp) in qpred:
                j, W = qpred[(s, sp)]; w = W[j, k].numpy().astype(np.float64)
                for tag, kw in (("old96", dict(side=96, mode="old96")), ("new144", dict(side=144, mode="new"))):
                    cl = P.make(w, **kw); hit = [c for c in cl if P.inside(c["box"], tb)]
                    if hit: x0, y0, sd = hit[0]["box"]; cand[tag].append((len(rows) - 1, embed([rgb[y0:y0 + sd, x0:x0 + sd]])[0]))
        if (s + 1) % 100 == 0: print(s + 1, "/", len(data.sids), flush=True)
    R = np.array([(r[2], r[3]) for r in rows]); spl = np.array([r[4] for r in rows]); held = np.array([r[5] for r in rows]); rk, su = R[:, 0], R[:, 1]; ident = rk * 4 + su; tr, va, te = spl == "train", spl == "val", spl == "test"; res = {}
    for side in (96, 144):
        X = np.stack(feats[side]).astype(np.float32); res[f"snapped{side}"] = {}
        for lab, y in (("rank", rk), ("suit", su), ("identity", ident)): r = probe(X[tr], y[tr], X[va], y[va], X[te], y[te]); res[f"snapped{side}"][lab] = {"test_acc": r["test_acc"], "C": r["C"]}
        # (b) candidate crops that contain the target
        tag = "old96" if side == 96 else "new144"; idx = np.array([i for i, _ in cand[tag]]); Xc = np.stack([f for _, f in cand[tag]]).astype(np.float32); sc = StandardScaler().fit(X[tr]); out = {}
        for lab, y in (("rank", rk), ("suit", su), ("identity", ident)):
            clf = LogisticRegression(C=res[f"snapped{side}"][lab]["C"], max_iter=3000).fit(sc.transform(X[tr]), y[tr]); pr = clf.predict(sc.transform(Xc)); m = ~held[idx]
            out[lab] = {"seen_test_acc": float((pr[m] == y[idx][m]).mean()), "n_seen": int(m.sum())}
            if lab == "suit": out[lab]["heldout_rank9_acc"] = float((pr[~m] == y[idx][~m]).mean()); out[lab]["n_heldout"] = int((~m).sum())
        res[f"candidate_crop_containing_target_{tag}"] = out
    for k, v in res.items(): print(k, json.dumps({a: (round(b.get("test_acc", b.get("seen_test_acc")), 3)) for a, b in v.items()}), "| heldout rank9 suit:", v.get("suit", {}).get("heldout_rank9_acc"))
    (OUT / "g4b_step0b_identity_sanity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
