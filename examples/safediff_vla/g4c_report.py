#!/usr/bin/env python
"""G4c report: funnel (G4b vs G4c), intra-crop localization (train/val/test/rank9), card-vs-card, end-to-end, anchor A/B on the final global map, retention."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g4c_matcher_train as gc  # noqa: E402

OLD = Path("outputs/eval/safediff_vla_g4_offline"); OB = Path("outputs/eval/safediff_vla_g4b_offline"); OC = gc.OUTC; G3 = Path("outputs/eval/safediff_vla_g3_offline")


def per_query(d, a, with_alpha):
    idx = a["idx"]; t = torch.as_tensor(idx, device="cuda"); cont = d.pur["contains_target"][t].cpu().numpy() > .5; cv = d.cvalid[t].cpu().numpy(); N = len(idx); r = np.arange(N); top = np.where(cv, a["s"], -1e9).argmax(1)
    cf = d.sim["cellfrac"].numpy(); valid = d.sim["valid"].numpy(); k = d.pairs[idx, 1]; map_ok = []
    for n, i in enumerate(idx):
        sc = d.pairs[i, 0]; ks = [c for c in range(4) if valid[sc, c]]; map_ok.append(max(ks, key=lambda c: float((a["w2"][n] * cf[sc, c]).sum())) == k[n])
    err = np.linalg.norm((a["xyz"] - d.anc[t].cpu().numpy())[:, :2], axis=1) * 100; q = {"proposal": (cont & cv).any(1), "fine": cont[r, top], "map": np.array(map_ok), "anc5": err < 5, "err": err}
    if with_alpha:
        tm = d.tokmask[t].float().cpu().numpy(); sets = tm >= 0.25; ok = np.zeros(N, bool)
        for n in range(N):
            j = top[n]
            if not cont[n, j]: continue
            S = sets[n, j]
            for c in range(4):
                if tm[n, j, c].max() > 0 and not S[c].any(): S[c, tm[n, j, c].argmax()] = True
            m = (S.astype(np.float32) * a["alpha"][n, j][None]).sum(-1); present = S.any(-1); ok[n] = np.where(present, m, -1).argmax() == k[n]
        q["intra"] = ok
    return q


def funnel(q):
    n = len(q["proposal"]); out = {"n": n, "1_proposal_contains_target": q["proposal"].mean(), "2_fine_top1_crop_correct": q["fine"].mean()}; cum = q["proposal"] & q["fine"]; out["2_cum"] = cum.mean()
    if "intra" in q: out["3_intra_card_correct(uncond)"] = (q["intra"]).mean(); cum = cum & q["intra"]; out["3_cum"] = cum.mean()
    out["4_final_map_correct_card(uncond)"] = q["map"].mean(); cum = cum & q["map"]; out["4_cum"] = cum.mean(); out["5_anchor<5cm(uncond)"] = q["anc5"].mean(); out["5_cum"] = (cum & q["anc5"]).mean(); return out


def ab(q):
    f = lambda x: {"n": int(len(x)), "median_xy_cm": float(np.median(x)) if len(x) else None, "mean_xy_cm": float(x.mean()) if len(x) else None, "frac_gt_5cm": float((x > 5).mean()) if len(x) else None}
    return {"map_selects_correct": float(q["map"].mean()), "A_correct": f(q["err"][q["map"]]), "B_wrong": f(q["err"][~q["map"]])}


def main():
    r4 = json.load(open(OB / "g4b_smoke_s0.json")); r1 = json.load(open(OLD / "g4_smoke_s0.json")); r3 = json.load(open(G3 / "g3_M1_s0.json")); r5 = json.load(open(OC / "g4c_smoke_s0.json")); A4 = torch.load(OB / "g4b_matcher_s0_arrays.pt", weights_only=False); A5 = torch.load(OC / "g4c_matcher_s0_arrays.pt", weights_only=False); d = gc.Data(); R = {}
    print("selected step G4c", r5["selected_step"], "guard", r5["guard_ok_at_selected"])
    print("\n== FUNNEL (fractions of ALL queries; *_cum = cumulative through that stage) ==")
    for sp in ("val", "test", "test_heldout_identity"):
        q4 = per_query(d, A4[sp], False); q5 = per_query(d, A5[sp], True); f4, f5 = funnel(q4), funnel(q5); R[sp] = {"G4b": f4, "G4c": f5, "G4b_AB": ab(q4), "G4c_AB": ab(q5)}; print(sp)
        keys = list(dict.fromkeys(list(f4) + list(f5)))
        for k in keys: print(f"   {k:36s} G4b {('%.3f' % f4[k]) if k in f4 else '  -  '} | G4c {('%.3f' % f5[k]) if k in f5 else '  -  '}")
    print("\n== INTRA-CROP localization (G4c) ==")
    for lab, src in (("train(OOF cand.)", r5["train_intra"]), ("val", r5["val"]["intra"]), ("test", r5["test"]["intra"]), ("rank9", r5["test_heldout_identity"]["intra"])):
        for sub in ("positive_crops", "fine_selected_crop_if_contains_target"):
            x = src[sub]; print(f"   {lab:16s} {sub:40s} n={x['n']:4d} top-token-in-target {x['top_token_in_target']:.3f} | target alpha mass {x['target_mass']:.3f} (uniform {x['chance_mass(target token fraction)']:.3f}) | correct-card in multi-card crops {x['correct_card_multi']:.3f} (uniform-alpha {x['correct_card_multi_uniform_alpha']:.3f}, n={x['n_multi_card']})")
    print("\n== card-vs-card candidate accuracy (fine) ==")
    print("G4b: train %.3f val %.3f test %.3f rank9 %.3f" % (r4["train_fine"]["card_vs_card"]["fine"], r4["val"]["fine"]["card_vs_card"]["fine"], r4["test"]["fine"]["card_vs_card"]["fine"], r4["test_heldout_identity"]["fine"]["card_vs_card"]["fine"]))
    print("G4c: train %.3f val %.3f test %.3f rank9 %.3f" % (r5["train_fine"]["card_vs_card"]["fine"], r5["val"]["fine"]["card_vs_card"]["fine"], r5["test"]["fine"]["card_vs_card"]["fine"], r5["test_heldout_identity"]["fine"]["card_vs_card"]["fine"]))
    print("\n== END-TO-END ==")
    for sp in ("val", "test", "test_heldout_identity"):
        print(sp)
        for nm, rr in (("G3 M1", r3.get(sp)), ("G4 v1", r1[sp]), ("G4b", r4[sp]), ("G4c", r5[sp])):
            if rr: print(f"   {nm:6s} R={rr['R_follow_xy_pooled']:.3f} CI{rr['R_follow_ci95_scene_bootstrap']} sel={rr['selection_acc']:.3f} medXY={rr['loc_err_xy_cm_median']:.2f} meanXY={rr['loc_err_xy_cm_mean']:.2f}")
        print(f"   G4c coarse-only same head: R={r5[sp + '_coarse_only_same_head']['R_follow_xy_pooled']:.3f} sel={r5[sp + '_coarse_only_same_head']['selection_acc']:.3f}")
    print("\n== ANCHOR A/B (final global map) ==")
    for sp in ("test", "test_heldout_identity"):
        for nm in ("G4b", "G4c"): print(sp, nm, json.dumps(R[sp][nm + "_AB"]))
    print("\n== retention (held-out episodes, median XY cm) ==")
    ks = ("f0", "k45", "k30", "k10", "pooled"); print(f"{'':8s}" + "".join(f"{k:>8s}" for k in ks))
    for nm, rr in (("V2", r4["retention_heldout_V2"]), ("G3 M1", r3["retention_heldout"]), ("G4 v1", r1["retention_heldout"]), ("G4b", r4["retention_heldout"]), ("G4c", r5["retention_heldout"])): print(f"{nm:8s}" + "".join(f"{rr[k]['median_xy_cm']:8.2f}" for k in ks))
    json.dump(R, open(OC / "g4c_report.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
