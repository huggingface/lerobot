#!/usr/bin/env python
"""G4d report: funnel G4b / G4c / G4d, intra-crop localization, end-to-end, anchor A/B on the final map, retention."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import g4c_matcher_train as gc  # noqa: E402
from g4c_report import per_query, funnel, ab  # noqa: E402

OLD = Path("outputs/eval/safediff_vla_g4_offline"); OB = Path("outputs/eval/safediff_vla_g4b_offline"); OC = gc.OUTC; OD = Path("outputs/eval/safediff_vla_g4d_offline"); G3 = Path("outputs/eval/safediff_vla_g3_offline")


def main():
    r = {n: json.load(open(p)) for n, p in (("G3", G3 / "g3_M1_s0.json"), ("v1", OLD / "g4_smoke_s0.json"), ("G4b", OB / "g4b_smoke_s0.json"), ("G4c", OC / "g4c_smoke_s0.json"), ("G4d", OD / "g4d_smoke_s0.json"))}
    A = {"G4b": torch.load(OB / "g4b_matcher_s0_arrays.pt", weights_only=False), "G4c": torch.load(OC / "g4c_matcher_s0_arrays.pt", weights_only=False), "G4d": torch.load(OD / "g4d_matcher_s0_arrays.pt", weights_only=False)}; d = gc.Data(); R = {}
    print("G4d selected step", r["G4d"]["selected_step"], "guard", r["G4d"]["guard_ok_at_selected"])
    print("\n== FUNNEL ==")
    for sp in ("val", "test", "test_heldout_identity"):
        q = {n: per_query(d, A[n][sp], n != "G4b") for n in A}; f = {n: funnel(q[n]) for n in q}; R[sp] = {"funnel": f, "AB": {n: ab(q[n]) for n in q}}; print(sp)
        keys = ["1_proposal_contains_target", "2_fine_top1_crop_correct", "3_intra_card_correct(uncond)", "4_final_map_correct_card(uncond)", "5_anchor<5cm(uncond)", "2_cum", "3_cum", "4_cum", "5_cum"]
        print(f"   {'':36s}{'G4b':>8s}{'G4c':>8s}{'G4d':>8s}")
        for k in keys: print(f"   {k:36s}" + "".join(f"{f[n][k]:8.3f}" if k in f[n] else f"{'-':>8s}" for n in ("G4b", "G4c", "G4d")))
        print("   stage3->stage4 drop (3 minus 4, uncond):  G4c %.3f | G4d %.3f" % (f["G4c"]["3_intra_card_correct(uncond)"] - f["G4c"]["4_final_map_correct_card(uncond)"], f["G4d"]["3_intra_card_correct(uncond)"] - f["G4d"]["4_final_map_correct_card(uncond)"]))
    print("\n== INTRA-CROP localization, positive crops (top-token / alpha mass / correct card in multi-card crops) ==")
    for lab, k in (("train", "train_intra"), ("val", "val"), ("test", "test"), ("rank9", "test_heldout_identity")):
        a = (r["G4c"][k] if lab == "train" else r["G4c"][k]["intra"])["positive_crops"]; b = (r["G4d"][k] if lab == "train" else r["G4d"][k]["intra"])["positive_crops"]
        print(f"   {lab:6s} G4c {a['top_token_in_target']:.3f} {a['target_mass']:.3f} {a['correct_card_multi']:.3f} | G4d {b['top_token_in_target']:.3f} {b['target_mass']:.3f} {b['correct_card_multi']:.3f} (n={b['n']})")
    print("\n== card-vs-card ==")
    for n in ("G4b", "G4c", "G4d"): print(n, "train %.3f val %.3f test %.3f rank9 %.3f" % (r[n]["train_fine"]["card_vs_card"]["fine"], r[n]["val"]["fine"]["card_vs_card"]["fine"], r[n]["test"]["fine"]["card_vs_card"]["fine"], r[n]["test_heldout_identity"]["fine"]["card_vs_card"]["fine"]))
    print("\n== END-TO-END ==")
    for sp in ("val", "test", "test_heldout_identity"):
        print(sp)
        for n in ("G3", "v1", "G4b", "G4c", "G4d"):
            x = r[n].get(sp)
            if x: print(f"   {n:4s} R={x['R_follow_xy_pooled']:.3f} CI{x['R_follow_ci95_scene_bootstrap']} sel={x['selection_acc']:.3f} medXY={x['loc_err_xy_cm_median']:.2f} meanXY={x['loc_err_xy_cm_mean']:.2f}")
    print("\n== ANCHOR A/B (final map correct = A) ==")
    for sp in ("test", "test_heldout_identity"):
        for n in ("G4b", "G4c", "G4d"): x = R[sp]["AB"][n]; print(f"{sp:22s} {n}: map-correct {x['map_selects_correct']:.3f} | A n={x['A_correct']['n']} med {x['A_correct']['median_xy_cm']:.2f} mean {x['A_correct']['mean_xy_cm']:.2f} >5cm {x['A_correct']['frac_gt_5cm']:.2f} | B n={x['B_wrong']['n']} med {x['B_wrong']['median_xy_cm']:.2f}")
    print("\n== retention (held-out episodes median XY cm) ==")
    ks = ("f0", "k45", "k30", "k10", "pooled"); print(f"{'':6s}" + "".join(f"{k:>8s}" for k in ks))
    for n, rr in (("V2", r["G4b"]["retention_heldout_V2"]), ("G3", r["G3"]["retention_heldout"]), ("v1", r["v1"]["retention_heldout"]), ("G4b", r["G4b"]["retention_heldout"]), ("G4c", r["G4c"]["retention_heldout"]), ("G4d", r["G4d"]["retention_heldout"])): print(f"{n:6s}" + "".join(f"{rr[k]['median_xy_cm']:8.2f}" for k in ks))
    print("swap-shift G4c", {k: round(v, 2) for k, v in r["G4c"]["retention_heldout"]["swap_shift_xy_cm"].items()}, "| G4d", {k: round(v, 2) for k, v in r["G4d"]["retention_heldout"]["swap_shift_xy_cm"].items()})
    json.dump(R, open(OD / "g4d_report.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
