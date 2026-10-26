#!/usr/bin/env python
"""G2 report: V2 / G1-A3 (3 seeds) / G2 M1-M3 (3 seeds) on the SAME sim splits and the SAME dataset caches. Reads g2_M*_s*.json + checkpoints."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from g1_train import Bottleneck, TaskData, V2Modules  # noqa: E402
from g2_train import FRAMES, OUTE, DataCache, eval_frames  # noqa: E402

G1E = Path("outputs/eval/safediff_vla_g1_offline"); G1T = Path("outputs/train/safediff_vla_g1")


def agg(vals): return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "per_seed": [round(float(v), 3) for v in vals]}


def main():
    dev = "cuda"; sim = TaskData("select_fruit", dev); mean, std = sim.mean, sim.std; dheld = DataCache(OUTE / "g2_held_cache.pt", dev); dval = DataCache(OUTE / "g2_vald_cache.pt", dev)
    res = {"sim": {}, "ret_held": {}, "ret_vald": {}, "sens": {}}
    g2 = {m: [json.load(open(OUTE / f"g2_{m}_s{s}.json")) for s in range(3)] for m in ("M1", "M2", "M3")}
    g1 = [json.load(open(G1E / f"select_fruit_A3_s{s}.json")) for s in range(3)]; v2 = json.load(open(G1E / "select_fruit_V2_reference.json"))
    sets = {"V2": [v2], "G1_A3": g1, **g2}
    for name, runs in sets.items():
        res["sim"][name] = {}
        for sp in ("val", "test", "heldout_identity"):
            res["sim"][name][sp] = {k: agg([r[sp][k] for r in runs]) for k in ("R_follow_xy_pooled", "selection_acc", "loc_err_xy_cm_median", "loc_err_xy_cm_mean", "referent_token_top1_in_mask", "referent_token_mass_in_mask", "pred_shift_xy_cm_mean") if runs[0][sp].get(k) is not None}
            res["sim"][name][sp]["R_ci_lower_per_seed"] = [round(r[sp]["R_follow_ci95_scene_bootstrap"][0], 3) for r in runs]
    # retention + sensitivity on the dataset caches, uniformly for V2 / G1 A3 / G2 (checkpoints re-evaluated with identical code)
    models = {"V2": [V2Modules().to(dev)]}
    models["G1_A3"] = []
    for s in range(3):
        m = Bottleneck().to(dev); m.load_state_dict(torch.load(G1T / "select_fruit" / f"A3_s{s}.pt")); models["G1_A3"].append(m)
    for mix in g2:
        models[mix] = []
        for s in range(3):
            m = Bottleneck().to(dev); m.load_state_dict(torch.load(Path("outputs/train/safediff_vla_g2") / f"{mix}_s{s}.pt")); models[mix].append(m)
    for name, ms in models.items():
        H = [eval_frames(m, dheld, mean, std, True) for m in ms]; Vd = [eval_frames(m, dval, mean, std, False) for m in ms]
        res["ret_held"][name] = {k: {"median_xy_cm": agg([h[k]["median_xy_cm"] for h in H]), "mean_xy_cm": agg([h[k]["mean_xy_cm"] for h in H])} for k in (*FRAMES, "pooled")}
        res["ret_vald"][name] = {k: agg([v[k]["median_xy_cm"] for v in Vd]) for k in (*FRAMES, "pooled")}
        res["sens"][name] = {k: {kk: agg([h["swap_sensitivity"][k][kk] for h in H]) for kk in H[0]["swap_sensitivity"][k]} for k in FRAMES}
    ref = res["ret_held"]["V2"]
    for name in res["ret_held"]:
        for k in (*FRAMES, "pooled"):
            b = ref[k]["median_xy_cm"]["mean"]; allow = b + max(2.0, 0.25 * b); r = res["ret_held"][name][k]["median_xy_cm"]
            r["allowed_limit_cm"] = allow; r["seeds_within_limit"] = int(sum(v <= allow for v in r["per_seed"]))
    (OUTE / "g2_summary.json").write_text(json.dumps(res, indent=1))
    S = res["sim"]; print("== [2] SIM (fruit; same held-out scene splits as G1)  mean[min,max] over seeds")
    for name in S:
        for sp in ("val", "test", "heldout_identity"):
            x = S[name][sp]; f = lambda k: f"{x[k]['mean']:.3f}[{x[k]['min']:.2f},{x[k]['max']:.2f}]" if k in x else "n/a"
            print(f"{name:6s} {sp:17s} R={f('R_follow_xy_pooled')} CIlow={x['R_ci_lower_per_seed']} acc={f('selection_acc')} medXY={f('loc_err_xy_cm_median')} top1tok={f('referent_token_top1_in_mask')} mass={f('referent_token_mass_in_mask')}")
    print("\n== [3] DATASET RETENTION held-out (median XY cm, mean[min,max]; limit = V2 + max(2cm,25%))")
    for name in res["ret_held"]:
        print(f"{name:6s}", " | ".join(f"{k}: {res['ret_held'][name][k]['median_xy_cm']['mean']:.1f}[{res['ret_held'][name][k]['median_xy_cm']['min']:.1f},{res['ret_held'][name][k]['median_xy_cm']['max']:.1f}] (mean {res['ret_held'][name][k]['mean_xy_cm']['mean']:.1f}; ok {res['ret_held'][name][k]['median_xy_cm']['seeds_within_limit']}/{len(res['ret_held'][name][k]['median_xy_cm']['per_seed'])}, lim {res['ret_held'][name][k]['median_xy_cm']['allowed_limit_cm']:.1f})" for k in (*FRAMES, "pooled")))
    print("\n== [4] DATASET-FRAME instruction-swap sensitivity (mean over seeds)  shift cm | frac>30cm | query | feature | map cosdist | map TV")
    for name in res["sens"]:
        for k in FRAMES:
            s = res["sens"][name][k]; print(f"{name:6s} {k:4s}: {s['pred_shift_xy_cm_mean']['mean']:5.1f} | {s['frac_shift_gt_30cm']['mean']:.2f} | {s['query_rel_change']['mean']:.3f} | {s['feature_rel_change']['mean']:.3f} | {s['referent_map_cos_distance']['mean']:.3f} | {s['referent_map_total_variation']['mean']:.3f}")
    print("\n== batch composition (per run): ", {m: g2[m][0]["batch_stats"] for m in g2})
    print("selected steps:", {m: [r["selected_step"] for r in g2[m]] for m in g2}, "| guard satisfied on VAL-D:", {m: [r["guard_satisfied_at_selected_on_valD"] for r in g2[m]] for m in g2})


if __name__ == "__main__":
    main()
