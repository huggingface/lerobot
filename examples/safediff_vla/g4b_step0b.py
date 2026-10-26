#!/usr/bin/env python
"""G4b STEP 0b (no training): (1) candidate-quality table old(96, cell-centre) vs new(144, centroid) on seen-test / held-out rank9 with M1 maps; (2) 144-vs-96 identity-representation sanity with the frozen crop encoder."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from g1_train import D, TaskData  # noqa: E402
import g4b_proposals as P  # noqa: E402

PD = D / "select_poker"; OUT = Path("outputs/eval/safediff_vla_g4b_offline")


def table(data, B, Z):
    res = {}
    for label, sp, hs in (("seen_test", "test", False), ("heldout_rank9", "test_heldout_identity", True)):
        W = Z["preds"][sp][1]
        for tag, kw in (("old96_cell", dict(side=96, mode="old96")), ("new144_centroid", dict(side=144, mode="new")), ("hybrid_front96+proj144(step0 repro)", None)):
            acc = {"n": 0, "rec1": 0, "rec2": 0, "ncards": [], "nearest": [], "tgt": [], "dis": [], "ncand": [], "neg_bg": 0, "neg_card": 0, "q_with_pos_and_cardneg": 0, "q_with_pos": 0}
            for j, s in enumerate(data.idx(sp)):
                names = data.names[s]; sid = data.sids[s]; ks = [k for k in range(len(names)) if bool(data.valid[s, k])]; gtb = [B[sid]["boxes_xyxy_per_camera"][names[c]][0] if c in ks else None for c in range(len(names))]
                for k in ks:
                    if bool(data.held[s, k]) != hs: continue
                    w = W[j, k].numpy().astype(np.float64); acc["n"] += 1
                    if kw is None:  # Step-0 reproduction: front 96 cell + projected centroid 144 (K=1: first of each camera, K=2: top-2 each)
                        cand_all = P.make(w, 144, "new", dedup=False); fr = P.make(w, 96, "old96", dedup=False)
                        sel = lambda K: [c for c in fr if c["src"] == "front" and c["rank"] < K] + [c for c in cand_all if c["src"] == "right" and c["rank"] < K]
                        acc["rec1"] += any(P.inside(c["box"], gtb[k]) for c in sel(1)); acc["rec2"] += any(P.inside(c["box"], gtb[k]) for c in sel(2)); acc["ncand"].append(len(sel(2))); continue
                    cl = P.make(w, **kw, dedup=False); cd = P.make(w, **kw); cl1 = [c for c in cl if c["rank"] < 1]
                    acc["rec1"] += any(P.inside(c["box"], gtb[k]) for c in cl1); acc["rec2"] += any(P.inside(c["box"], gtb[k]) for c in cl); acc["ncand"].append(len(cd))
                    for c in cd:
                        p = P.purity(c["box"], gtb, k); acc["ncards"].append(p["n_cards"]); acc["tgt"].append(p["tgt_frac"]); acc["dis"].append(p["dis_frac"])
                        if p["contains_target"]: acc["nearest"].append(p["target_nearest"])
                    pos = [P.purity(c["box"], gtb, k)["contains_target"] for c in cd]
                    if any(pos):
                        acc["q_with_pos"] += 1; negs = [P.purity(c["box"], gtb, k) for c, pp in zip(cd, pos) if not pp]
                        for q in negs: acc["neg_card" if q["n_cards"] > 0 else "neg_bg"] += 1
                        acc["q_with_pos_and_cardneg"] += any(q["n_cards"] > 0 for q in negs)
            n = acc["n"]; nn = max(1, acc["neg_bg"] + acc["neg_card"])
            res.setdefault(label, {})[tag] = {"n": n, "recall_K1": acc["rec1"] / n, "recall_K2": acc["rec2"] / n, "mean_candidates_after_dedup": float(np.mean(acc["ncand"])), "cards_per_crop_mean": float(np.mean(acc["ncards"])) if acc["ncards"] else None,
                                              "target_nearest_to_center_given_contains": float(np.mean(acc["nearest"])) if acc["nearest"] else None, "target_area_frac_mean": float(np.mean(acc["tgt"])) if acc["tgt"] else None, "distractor_area_frac_mean": float(np.mean(acc["dis"])) if acc["dis"] else None,
                                              "negatives_background_frac": acc["neg_bg"] / nn, "negatives_other_card_frac": acc["neg_card"] / nn, "frac_queries_with_positive_and_other_card_negative": acc["q_with_pos_and_cardneg"] / n}
    return res


def main():
    data = TaskData("select_poker", "cpu"); B = {}
    for p in sorted(PD.glob("bboxes_*.json")): B.update(json.load(open(p)))
    Z = torch.load("outputs/train/safediff_vla_g3/M1_s0_preds.pt", weights_only=False); res = {"table": table(data, B, Z)}
    for lab, d in res["table"].items():
        print(lab)
        for tag, v in d.items(): print(f"  {tag:38s} " + " ".join(f"{k}={x:.3f}" if isinstance(x, float) else f"{k}={x}" for k, x in v.items()))
    (OUT / "g4b_step0b_table.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
