#!/usr/bin/env python
"""G4b OOF distribution gate (no training): same proposal geometry (centroid+144, dedup) applied to coarse maps from (a) the in-sample final M1 on TRAIN scenes, (b) OOF fold models on TRAIN scenes, (c) final M1 on val, (d) test. Seen-identity queries only."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from g1_train import D  # noqa: E402
import g4b_proposals as P  # noqa: E402

PD = D / "select_poker"; OUT = Path("outputs/eval/safediff_vla_g4b_offline"); OLD = Path("outputs/eval/safediff_vla_g4_offline")


def main():
    old = torch.load(OLD / "g4_cache.pt", weights_only=False)["sim"]; new = torch.load(OUT / "g4b_cache.pt", weights_only=False)["sim"]; B = {}
    for p in sorted(PD.glob("bboxes_*.json")): B.update(json.load(open(p)))
    split = np.array(old["split"]); valid = old["valid"].numpy(); held = old["held"].numpy(); wold = old["w"].float().numpy(); wnew = new["w"].numpy(); res = {}
    for label, sp, w in (("train_in_sample(final M1)", "train", wold), ("train_OOF", "train", wnew), ("val", "val", wnew), ("test", "test", wnew), ("heldout_rank9", "test_heldout_identity", wnew)):
        a = {"n": 0, "f_top1": 0, "any_top1": 0, "any_top2": 0, "cand_recall": 0, "ncards": [], "nearest": [], "neg_bg": 0, "neg_card": 0, "q_pos_cardneg": 0, "q_pos": 0, "ncand": [], "pos_rate_per_cand": []}
        for s in np.where(split == sp)[0]:
            names = old["names"][s]; sid = old["sids"][s]; gtb = [B[sid]["boxes_xyxy_per_camera"][names[c]][0] if (c < len(names) and valid[s, c]) else None for c in range(len(names))]
            for k in range(4):
                if not valid[s, k] or bool(held[s, k]) != (sp == "test_heldout_identity"): continue
                a["n"] += 1; cl = P.make(w[s, k].astype(np.float64), dedup=False); cd = P.make(w[s, k].astype(np.float64)); hit = lambda c: P.inside(c["box"], gtb[k])
                a["f_top1"] += hit(cl[0]); a["any_top1"] += any(hit(c) for c in cl if c["rank"] < 1); a["any_top2"] += any(hit(c) for c in cl); a["ncand"].append(len(cd))
                ps = [P.purity(c["box"], gtb, k) for c in cd]; pos = [q["contains_target"] for q in ps]; a["cand_recall"] += any(pos)
                for q in ps: a["ncards"].append(q["n_cards"]); a["pos_rate_per_cand"].append(q["contains_target"])
                for q in ps:
                    if q["contains_target"]: a["nearest"].append(q["target_nearest"])
                if any(pos):
                    a["q_pos"] += 1; negs = [q for q in ps if not q["contains_target"]]
                    for q in negs: a["neg_card" if q["n_cards"] > 0 else "neg_bg"] += 1
                    a["q_pos_cardneg"] += any(q["n_cards"] > 0 for q in negs)
        n = a["n"]; nn = max(1, a["neg_bg"] + a["neg_card"])
        res[label] = {"n_queries": n, "front_top1_crop_contains_target": a["f_top1"] / n, "K1_union_contains_target": a["any_top1"] / n, "K2_union_contains_target": a["any_top2"] / n, "candidate_recall_after_dedup": a["cand_recall"] / n, "mean_candidates": float(np.mean(a["ncand"])),
                      "negatives_background_frac": a["neg_bg"] / nn, "negatives_other_card_frac": a["neg_card"] / nn, "frac_queries_with_positive_and_other_card_negative": a["q_pos_cardneg"] / n, "cards_per_crop_mean": float(np.mean(a["ncards"])), "target_nearest_to_center_given_contains": float(np.mean(a["nearest"])),
                      "positive_rate_per_candidate": float(np.mean(a["pos_rate_per_cand"]))}
    (OUT / "g4b_oof_gate.json").write_text(json.dumps(res, indent=1)); ks = list(next(iter(res.values())).keys())
    print(f"{'':52s}" + "".join(f"{k[:24]:>26s}" for k in res))
    for m in ks: print(f"{m:52s}" + "".join(f"{res[k][m]:26.3f}" for k in res))


if __name__ == "__main__":
    main()
