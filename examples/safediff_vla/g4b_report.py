#!/usr/bin/env python
"""G4b report: card-vs-card fine accuracy (G4 v1 recomputed on its own front pair), end-to-end offline metrics (G3 / G4v1 / G4b), anchor-following diagnostic (A: final referent map selects the correct card; B: wrong), retention."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import g4_matcher_train as g1  # noqa: E402
import g4b_matcher_train as g2  # noqa: E402

OLD = Path("outputs/eval/safediff_vla_g4_offline"); OUT = Path("outputs/eval/safediff_vla_g4b_offline"); G3 = Path("outputs/eval/safediff_vla_g3_offline")


def v1_front_cvc(d1, arr, sp):
    """G4 v1: front pair (2 crops): queries with exactly one positive crop whose negative crop contains another card -> fine score(pos) > score(neg)."""
    idx = arr["idx"]; t = torch.as_tensor(idx, device="cuda"); c = d1.pur["contains_target"][t].cpu().numpy()[:, :2] > .5; n = d1.pur["n_cards_inside"][t].cpu().numpy()[:, :2]; s = arr["s"][:, :2]
    ok = (c.sum(1) == 1) & (((~c) & (n > 0)).sum(1) == 1); pos = c.argmax(1); return float((s[ok, :].argmax(1) == pos[ok]).mean()), int(ok.sum())


def anchor_follow(P, d, idx, w2, split):
    """P: predicted xyz per pair, w2: final map; correct = card with max sum(w2*cellcover) is the instructed one."""
    cfall = d.sim["cellfrac"].numpy(); valid = d.sim["valid"].numpy(); rows = []
    for n, i in enumerate(idx):
        sc, k = d.pairs[i]; ks = [c for c in range(4) if valid[sc, c]]; sel = max(ks, key=lambda c: float((w2[n] * cfall[sc, c]).sum())); err = float(np.linalg.norm((P[n] - d.anc[i].cpu().numpy())[:2]) * 100); rows.append((sel == k, err))
    r = np.array(rows, float); A, Bm = r[r[:, 0] == 1, 1], r[r[:, 0] == 0, 1]
    f = lambda x: {"n": int(len(x)), "median_xy_cm": float(np.median(x)), "mean_xy_cm": float(x.mean()), "frac_gt_5cm": float((x > 5).mean())}
    return {"map_selects_correct_frac": float(r[:, 0].mean()), "A_correct_map": f(A), "B_wrong_map": f(Bm)}


def main():
    r4 = json.load(open(OUT / "g4b_smoke_s0.json")); r1 = json.load(open(OLD / "g4_smoke_s0.json")); r3 = json.load(open(G3 / "g3_M1_s0.json")); A4 = torch.load(OUT / "g4b_matcher_s0_arrays.pt", weights_only=False); A1 = torch.load(OLD / "g4_matcher_s0_arrays.pt", weights_only=False)
    d1 = g1.Data(); d2 = g2.Data(); rep = {}
    print("== card-vs-card FINE accuracy (positive crop outranks the other-card negatives) ==")
    print("G4b (front-only 144px, OOF train):  train %.3f | val %.3f | test %.3f | held-out rank9 %.3f" % (r4["train_fine"]["card_vs_card"]["fine"], r4["val"]["fine"]["card_vs_card"]["fine"], r4["test"]["fine"]["card_vs_card"]["fine"], r4["test_heldout_identity"]["fine"]["card_vs_card"]["fine"]))
    print("   n:", r4["train_fine"]["card_vs_card"]["n"], r4["val"]["fine"]["card_vs_card"]["n"], r4["test"]["fine"]["card_vs_card"]["n"], r4["test_heldout_identity"]["fine"]["card_vs_card"]["n"])
    for sp, lab in (("val", "val"), ("test", "test"), ("test_heldout_identity", "heldout")): print(f"   {lab}: coarse-score baseline {r4[sp]['fine']['card_vs_card']['coarse']:.3f} | fine {r4[sp]['fine']['card_vs_card']['fine']:.3f} | final map {r4[sp]['fine']['card_vs_card']['final_map']:.3f} || card-vs-bg fine {r4[sp]['fine']['card_vs_background']['fine']} | top-1 cand contains target: coarse {r4[sp]['fine']['top1_candidate_contains_target']['coarse']:.3f} fine {r4[sp]['fine']['top1_candidate_contains_target']['fine']:.3f}")
    v1 = {sp: v1_front_cvc(d1, A1[sp], sp) for sp in ("val", "test", "test_heldout_identity")}; print("G4 v1 front (96px pair), recomputed same-definition:", {k: (round(a, 3), n) for k, (a, n) in v1.items()}, "(quoted by PI: train 1.000, val 0.797, test 0.725)")
    print("\n== end-to-end offline (seed 0) ==")
    for sp in ("val", "test", "test_heldout_identity"):
        print(sp)
        for nm, rr in (("G3 M1", r3[sp] if sp in r3 else None), ("G4 v1", r1[sp]), ("G4b", r4[sp])):
            if rr is None: continue
            print(f"   {nm:6s} R_follow={rr['R_follow_xy_pooled']:.3f} CI{rr['R_follow_ci95_scene_bootstrap']} selection={rr['selection_acc']:.3f} medXY={rr['loc_err_xy_cm_median']:.2f}cm meanXY={rr['loc_err_xy_cm_mean']:.2f}cm")
        print(f"   G4b coarse-only same head: R={r4[sp + '_coarse_only_same_head']['R_follow_xy_pooled']:.3f} sel={r4[sp + '_coarse_only_same_head']['selection_acc']:.3f}")
    print("\n== anchor-following (test): A = final map selects the correct card, B = wrong card ==")
    d2.heldq  # noqa
    af = {}
    af["G4b"] = anchor_follow(A4["test"]["xyz"], d2, A4["test"]["idx"], A4["test"]["w2"], "test"); af["G4 v1"] = anchor_follow(A1["test"]["xyz"], d1, A1["test"]["idx"], A1["test"]["w2"], "test")
    af["G4b_heldout"] = anchor_follow(A4["test_heldout_identity"]["xyz"], d2, A4["test_heldout_identity"]["idx"], A4["test_heldout_identity"]["w2"], "h")
    for k, v in af.items(): print(k, json.dumps(v))
    print("\n== retention: held-out dataset episodes, median XY err (cm) ==")
    keys = ("f0", "k45", "k30", "k10", "pooled"); V2 = r4["retention_heldout_V2"]
    print(f"{'':8s}" + "".join(f"{k:>8s}" for k in keys))
    for nm, rr in (("V2", V2), ("G3 M1", r3["retention_heldout"]), ("G4 v1", r1["retention_heldout"]), ("G4b", r4["retention_heldout"])): print(f"{nm:8s}" + "".join(f"{rr[k]['median_xy_cm']:8.2f}" for k in keys))
    print("G4b selected step", r4["selected_step"], "guard_ok", r4["guard_ok_at_selected"], "| swap-shift G4b", r4["retention_heldout"].get("swap_shift_xy_cm"), "v1", r1["retention_heldout"].get("swap_shift_xy_cm"))
    json.dump({"v1_front_card_vs_card": v1, "anchor_following": af}, open(OUT / "g4b_report_extra.json", "w"), indent=1)


if __name__ == "__main__":
    main()
