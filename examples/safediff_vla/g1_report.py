#!/usr/bin/env python
"""G1 offline report: per-arm seed consistency, KEEP rule, arm comparisons, retention. Reads outputs/eval/safediff_vla_g1_offline/*.json.
KEEP (strong-candidate) rule, fixed in advance: on BOTH val AND test, R_follow_xy >= 0.6, scene-bootstrap 95% CI lower bound >= 0.4,
referent-selection accuracy >= 0.8, median XY localisation error <= 5 cm. An arm is a strong-KEEP candidate iff at least 2 of 3 seeds
individually satisfy all four on val AND test, and the seed-mean of each metric also satisfies them. Held-out identity: report only (target R_follow >= 0.3)."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from g1_retention import eval_model  # noqa: E402
from g1_train import ARMS, Bottleneck, OUTE, OUTT, V2Modules  # noqa: E402

TASKS = ("select_poker", "select_fruit")


def passes(m): return m["R_follow_xy_pooled"] >= 0.6 and m["R_follow_ci95_scene_bootstrap"][0] >= 0.4 and m["selection_acc"] >= 0.8 and m["loc_err_xy_cm_median"] <= 5.0


def main():
    S = {}
    for task in TASKS:
        S[task] = {}
        for arm in ARMS:
            runs = [json.load(open(p)) for p in sorted(OUTE.glob(f"{task}_{arm}_s*.json"))]
            if not runs: continue
            agg = {}
            for sp in ("val", "test", "heldout_identity"):
                for key, short in (("R_follow_xy_pooled", "R"), ("selection_acc", "acc"), ("loc_err_xy_cm_median", "loc"), ("pred_shift_xy_cm_mean", "shift")):
                    v = [r[sp][key] for r in runs]; agg[f"{sp}_{short}"] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "per_seed": [round(x, 3) for x in v]}
                agg[f"{sp}_R_ci_lower_per_seed"] = [r[sp]["R_follow_ci95_scene_bootstrap"][0] for r in runs]
            seed_pass = [passes(r["val"]) and passes(r["test"]) for r in runs]
            mean_ok = all(agg[f"{sp}_R"]["mean"] >= 0.6 and np.mean(agg[f"{sp}_R_ci_lower_per_seed"]) >= 0.4 and agg[f"{sp}_acc"]["mean"] >= 0.8 and agg[f"{sp}_loc"]["mean"] <= 5.0 for sp in ("val", "test"))
            agg["seeds_passing_all_four_on_val_and_test"] = f"{sum(seed_pass)}/{len(runs)}"; agg["seed_mean_satisfies_all"] = bool(mean_ok)
            agg["verdict"] = "strong-KEEP candidate" if (sum(seed_pass) >= 2 and mean_ok) else "REJECT"
            agg["heldout_identity_R_target_0.3_met_by_seed_mean"] = bool(agg["heldout_identity_R"]["mean"] >= 0.3)
            agg["token_localisation_top1_val_mean"] = float(np.mean([r["val"]["referent_token_top1_in_mask"] or 0 for r in runs]))
            S[task][arm] = agg
        ref = OUTE / f"{task}_V2_reference.json"
        if ref.exists(): r = json.load(open(ref)); S[task]["V2"] = {sp: {k: r[sp][k] for k in ("R_follow_xy_pooled", "selection_acc", "loc_err_xy_cm_median", "pred_shift_xy_cm_mean")} for sp in ("val", "test", "heldout_identity")}
    # arm comparisons (seed-mean, val/test R_follow and selection acc)
    cmp = {}
    for task in TASKS:
        cmp[task] = {}
        for a, b, q in (("A0", "A1", "is the bottleneck needed?"), ("A1", "A2", "is L_ref needed?"), ("A2", "A3", "is L_cf needed?")):
            if a in S[task] and b in S[task]:
                cmp[task][f"{a}->{b}"] = {"question": q, **{f"d_{sp}_{k}": round(S[task][b][f"{sp}_{k}"]["mean"] - S[task][a][f"{sp}_{k}"]["mean"], 3) for sp in ("val", "test") for k in ("R", "acc", "loc")}}
    # retention on dataset frames
    ret = {}
    for task in TASKS:
        ret[task] = {"V2": eval_model(V2Modules().cuda(), task)}; base = ret[task]["V2"]["median_xy_err_cm_pooled"]; allow = base + max(2.0, 0.25 * base)
        for arm in ARMS:
            for p in sorted((OUTT / task).glob(f"{arm}_s?.pt")):
                m = (V2Modules(warm=False) if ARMS[arm]["model"] == "v2" else Bottleneck()).cuda(); m.load_state_dict(torch.load(p))
                r = eval_model(m, task); r["within_forgetting_guard"] = bool(r["median_xy_err_cm_pooled"] <= allow); r["guard_limit_cm"] = allow; ret[task][p.stem] = r
    out = {"summary": S, "arm_comparisons": cmp, "retention": ret}; (OUTE / "g1_summary.json").write_text(json.dumps(out, indent=1))
    for task in TASKS:
        print(f"\n===== {task}")
        if "V2" in S[task]: print("V2 reference:", {sp: tuple(round(v, 3) for v in S[task]['V2'][sp].values()) for sp in S[task]["V2"]}, "(R, acc, loc cm, shift cm)")
        for arm, a in S[task].items():
            if arm == "V2": continue
            f = lambda k: f"{a[k]['mean']:+.3f}[{a[k]['min']:+.2f},{a[k]['max']:+.2f}]" if k.endswith("_R") else f"{a[k]['mean']:.2f}[{a[k]['min']:.2f},{a[k]['max']:.2f}]"
            print(f"{arm}: val R {f('val_R')} acc {f('val_acc')} loc {f('val_loc')} | test R {f('test_R')} acc {f('test_acc')} loc {f('test_loc')} | heldID R {f('heldout_identity_R')} acc {f('heldout_identity_acc')} | seeds pass {a['seeds_passing_all_four_on_val_and_test']} mean-ok {a['seed_mean_satisfies_all']} -> {a['verdict']} | top1-token {a['token_localisation_top1_val_mean']:.2f}")
        print("comparisons:", json.dumps(cmp[task]))
        print("retention (median xy err cm pooled | k30 swap shift | f0 swap shift | guard):", {k: (round(v["median_xy_err_cm_pooled"], 1), round(v["k30"]["mean_pred_shift_under_swap_cm"], 2), round(v["f0"]["mean_pred_shift_under_swap_cm"], 2), v.get("within_forgetting_guard")) for k, v in ret[task].items()})


if __name__ == "__main__":
    main()
