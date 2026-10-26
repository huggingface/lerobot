#!/usr/bin/env python
"""PHASE G1-A: prototype counterfactual dataset statistics + a-priori scene-level splits (no training)."""
from __future__ import annotations

import glob, json, sys
from pathlib import Path

import numpy as np, pandas as pd

D = Path("outputs/data/safediff_vla_g1_counterfactual")
O = Path("outputs/eval/safediff_vla_phase_d0_training_data_audit")
HELD = {"select_poker": {"ranks": {"9"}}, "select_fruit": {"queries": {"kiwi", "lemon"}}}


def split_of(task, i, names):
    """scene-level; held-out IDENTITY handled first (a scene is never split across train/test)."""
    if task == "select_poker" and any(n.split("_of_")[0] in HELD[task]["ranks"] for n in names):
        return "test_heldout_identity"
    b = i % 20
    return "train" if b < 14 else ("val" if b < 17 else "test")


def main():
    eps = pd.read_json(O / "episodes.json"); eps = eps[eps.has_anchor]
    res = {}
    for task in ("select_poker", "select_fruit"):
        sc = [json.load(open(f)) for f in sorted(glob.glob(str(D / task / "*.json"))) if "index" not in f and "splits" not in f]
        Ks = [len(s["candidates"]) for s in sc]; rows = []; sep = []; splits = {}
        for s in sc:
            i = s["seed"] - 10000; names = [c["name"] for c in s["candidates"]]
            sp = split_of(task, i, names); splits[s["scene_id"]] = sp
            A = np.array([c["anchor_robot"] for c in s["candidates"]])
            for a in range(len(A)):
                for b in range(a + 1, len(A)):
                    sep.append(np.linalg.norm((A[a] - A[b])[:2]))
            for c in s["candidates"]:
                rows.append({"scene": s["scene_id"], "split": sp, "name": c["name"], "valid": c["expert_collision_check_valid"], "fallback": c["expert_used_default_fallback"],
                             "n_kp": c["n_grasp_keypoints"], "in_train_instr": c["instruction_in_training_set"], "vis_front": c["visible_px_per_camera"][0],
                             "vis_right": c["visible_px_per_camera"][1], "vis_wrist": c["visible_px_per_camera"][2],
                             "dz_anchor_minus_origin": c["anchor_robot"][2] - (np.array(c["object_pose_world"]["xpos"])[2] - s["robot_base_world"][2])})
        df = pd.DataFrame(rows); sep = np.array(sep)
        anc = np.array([c["anchor_robot"] for s in sc for c in s["candidates"]]); de = np.stack(eps[eps.task == task].anchor.to_numpy())
        npz = np.load(D / task / f"{sc[0]['scene_id']}.npz")
        cf = np.concatenate([np.load(D / task / f"{s['scene_id']}.npz")["cellfrac"].reshape(len(s["candidates"]), -1) for s in sc])
        res[task] = {
            "scenes": len(sc), "samples(scene x referent)": len(df), "K_distribution": {int(k): int(Ks.count(k)) for k in sorted(set(Ks))},
            "counterfactual_pairs_per_scene_total": int(sum(k * (k - 1) // 2 for k in Ks)),
            "ordered_pairs_total": int(sum(k * (k - 1) for k in Ks)),
            "split_scene_counts": pd.Series(splits).value_counts().to_dict(),
            "pair_separation_xy_cm_pctiles_min_10_50_90": (np.percentile(sep, [0, 10, 50, 90]) * 100).round(1).tolist(),
            "frac_pairs_closer_than_3cm": float((sep < 0.03).mean()), "frac_pairs_closer_than_5cm": float((sep < 0.05).mean()),
            "label_expert_collision_check_valid_frac": float(df.valid.mean()), "label_default_fallback_count": int(df.fallback.sum()),
            "grasp_keypoints_per_object": df.n_kp.value_counts().to_dict(),
            "anchor_minus_object_origin_z_cm_mean": float(df.dz_anchor_minus_origin.mean() * 100),
            "instruction_string_in_dataset_training_set_frac": float(df.in_train_instr.mean()),
            "referent_names": df.name.value_counts().to_dict() if task == "select_fruit" else f"{df.name.nunique()} distinct cards",
            "visibility": {"front_px>0": float((df.vis_front > 0).mean()), "right_px>0": float((df.vis_right > 0).mean()), "either_static_cam_px>0": float(((df.vis_front > 0) | (df.vis_right > 0)).mean()),
                           "median_px_front": float(df.vis_front.median()), "median_token_cells_covered_per_referent(of 192)": float(np.median((cf > 0).sum(1)))},
            "anchor_xy_mean_std_cm_G1_vs_dataset": {"G1": [(anc[:, :2].mean(0) * 100).round(1).tolist(), (anc[:, :2].std(0) * 100).round(1).tolist()],
                                                    "dataset": [(de[:, :2].mean(0) * 100).round(1).tolist(), (de[:, :2].std(0) * 100).round(1).tolist()]},
            "anchor_z_cm_mean_G1_vs_dataset": [float(anc[:, 2].mean() * 100), float(de[:, 2].mean() * 100)],
            "image_array": {"rgb": list(npz["rgb"].shape), "state": npz["state"].round(3).tolist()}}
        if task == "select_poker":
            held = df[df.name.str.split("_of_").str[0].isin(HELD[task]["ranks"])]
            res[task]["heldout_identity"] = {"rule": "scenes containing any card of rank 9 (4 cards) are test-only", "scenes": int(sum(v == "test_heldout_identity" for v in splits.values()))}
        else:
            res[task]["heldout_identity"] = {"rule": "queries of kiwi/lemon never used in train/val; they remain visible as distractors", "n_held_queries": int(df.name.isin(HELD[task]["queries"]).sum())}
        import hashlib
        by_split = {}
        for sid, sp in splits.items(): by_split.setdefault(sp, set()).add(sid)
        names = sorted(by_split)
        overlap = {f"{a}&{b}": len(by_split[a] & by_split[b]) for i, a in enumerate(names) for b in names[i + 1:]}
        seeds = {sid: int(sid.split("_")[-1]) for sid in splits}
        h = {}
        for sid in splits: h.setdefault(hashlib.md5(np.load(D / task / f"{sid}.npz")["rgb"][0].tobytes()).hexdigest(), []).append(sid)
        dup_groups = [v for v in h.values() if len(v) > 1]
        cross_split_dup = sum(1 for v in dup_groups if len({splits[x] for x in v}) > 1)
        res[task]["integrity_checks"] = {"scene_id_overlap_between_splits": overlap, "all_zero": all(v == 0 for v in overlap.values()),
            "duplicate_front_camera_images": len(dup_groups), "duplicates_spanning_splits": cross_split_dup, "min_seed": min(seeds.values()),
            "max_seed": max(seeds.values()), "scenes_with_K_lt_3": int(sum(k < 3 for k in Ks)),
            "heldout_rule_violations": int(sum(1 for sid, sp in splits.items() if task == "select_poker" and sp != "test_heldout_identity" and any(n.split("_of_")[0] in HELD[task]["ranks"] for n in [c["name"] for c in json.load(open(D / task / f"{sid}.json"))["candidates"]])))}
        (D / task / "splits.json").write_text(json.dumps(splits))
    (D / "prototype_stats.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
