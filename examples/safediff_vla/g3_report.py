#!/usr/bin/env python
"""G3 (poker) report: V2 / G1-A3 / G3-M1 on the SAME sim splits and dataset caches; fruit G2-M1 listed as reference only."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from g1_train import Bottleneck, TaskData, V2Modules  # noqa: E402
from g2_train import FRAMES, DataCache, eval_frames  # noqa: E402

G1E = Path("outputs/eval/safediff_vla_g1_offline"); G1T = Path("outputs/train/safediff_vla_g1/select_poker"); G3E = Path("outputs/eval/safediff_vla_g3_offline"); G3T = Path("outputs/train/safediff_vla_g3"); G2E = Path("outputs/eval/safediff_vla_g2_offline")


def agg(v): return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "per_seed": [round(float(x), 3) for x in v]}


def main():
    dev = "cuda"; sim = TaskData("select_poker", dev); mean, std = sim.mean, sim.std; dheld = DataCache(G3E / "g3_held_cache.pt", dev); dval = DataCache(G3E / "g3_vald_cache.pt", dev)
    runs = {"V2": [json.load(open(G1E / "select_poker_V2_reference.json"))], "G1_A3": [json.load(open(G1E / f"select_poker_A3_s{s}.json")) for s in range(3)],
            "G2_fruit_M1(ref only, fruit scenes)": [json.load(open(G2E / f"g2_M1_s{s}.json")) for s in range(3)], "G3_M1": [json.load(open(G3E / f"g3_M1_s{s}.json")) for s in range(3)]}
    res = {"sim": {}, "ret": {}, "sens": {}}
    for n, rs_ in runs.items():
        res["sim"][n] = {sp: {k: agg([r[sp][k] for r in rs_]) for k in ("R_follow_xy_pooled", "selection_acc", "loc_err_xy_cm_median", "referent_token_top1_in_mask", "referent_token_mass_in_mask", "pred_shift_xy_cm_mean") if rs_[0][sp].get(k) is not None} | {"R_ci_lower_per_seed": [round(r[sp]["R_follow_ci95_scene_bootstrap"][0], 3) for r in rs_]} for sp in ("val", "test", "heldout_identity")}
    models = {"V2": [V2Modules().to(dev)], "G1_A3": [], "G3_M1": []}
    for s in range(3):
        m = Bottleneck().to(dev); m.load_state_dict(torch.load(G1T / f"A3_s{s}.pt")); models["G1_A3"].append(m)
        m = Bottleneck().to(dev); m.load_state_dict(torch.load(G3T / f"M1_s{s}.pt")); models["G3_M1"].append(m)
    for n, ms in models.items():
        H = [eval_frames(m, dheld, mean, std, True) for m in ms]
        res["ret"][n] = {k: {"median_xy_cm": agg([h[k]["median_xy_cm"] for h in H]), "mean_xy_cm": agg([h[k]["mean_xy_cm"] for h in H])} for k in (*FRAMES, "pooled")}
        res["sens"][n] = {k: {kk: agg([h["swap_sensitivity"][k][kk] for h in H]) for kk in H[0]["swap_sensitivity"][k]} for k in FRAMES}
    ref = res["ret"]["V2"]
    for n in res["ret"]:
        for k in (*FRAMES, "pooled"):
            b = ref[k]["median_xy_cm"]["mean"]; lim = b + max(2.0, 0.25 * b); r = res["ret"][n][k]["median_xy_cm"]; r["limit_cm"] = lim; r["seeds_within"] = int(sum(v <= lim for v in r["per_seed"]))
    (G3E / "g3_summary.json").write_text(json.dumps(res, indent=1))
    print("== SIM (poker; same splits)  mean[min,max] over seeds")
    for n in res["sim"]:
        for sp in ("val", "test", "heldout_identity"):
            x = res["sim"][n][sp]; f = lambda k: f"{x[k]['mean']:.3f}[{x[k]['min']:.2f},{x[k]['max']:.2f}]" if k in x else "n/a"
            print(f"{n[:22]:22s} {sp:17s} R={f('R_follow_xy_pooled')} CIlow={x['R_ci_lower_per_seed']} acc={f('selection_acc')} medXY={f('loc_err_xy_cm_median')} top1tok={f('referent_token_top1_in_mask')} mass={f('referent_token_mass_in_mask')}")
    print("\n== DATASET RETENTION held-out poker (median XY cm; limit = V2 + max(2cm,25%))")
    for n in res["ret"]: print(f"{n:6s}", " | ".join(f"{k}: {res['ret'][n][k]['median_xy_cm']['mean']:.1f}[{res['ret'][n][k]['median_xy_cm']['min']:.1f},{res['ret'][n][k]['median_xy_cm']['max']:.1f}] (mean {res['ret'][n][k]['mean_xy_cm']['mean']:.1f}; ok {res['ret'][n][k]['median_xy_cm']['seeds_within']}/{len(res['ret'][n][k]['median_xy_cm']['per_seed'])}, lim {res['ret'][n][k]['median_xy_cm']['limit_cm']:.1f})" for k in (*FRAMES, "pooled")))
    print("\n== dataset swap sensitivity (shift cm | frac>30cm | query | feature | map cosdist)")
    for n in res["sens"]:
        for k in FRAMES:
            s = res["sens"][n][k]; print(f"{n:6s} {k:4s}: {s['pred_shift_xy_cm_mean']['mean']:5.1f} | {s['frac_shift_gt_30cm']['mean']:.2f} | {s['query_rel_change']['mean']:.3f} | {s['feature_rel_change']['mean']:.3f} | {s['referent_map_cos_distance']['mean']:.3f}")
    g3 = [json.load(open(G3E / f"g3_M1_s{s}.json")) for s in range(3)]
    print("\nG3 selected steps", [r["selected_step"] for r in g3], "guard satisfied", [r["guard_satisfied_at_selected_on_valD"] for r in g3], "| batch", g3[0]["batch_stats"])


if __name__ == "__main__":
    main()
