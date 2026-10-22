#!/usr/bin/env python
"""[v2.1HR-expansion] Verified referent table + NEW scene split for the poker/mahjong expansion (seeds 100050+). Same validity rules as v21_referent_table.
Split: per task 200 train / 50 val scenes (seeded), all counterfactual instructions of a scene share the split. The frozen 40/10 experiment is untouched."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from v21_referent_table import MIN_SEP_M, cam_params, parse_identity, project  # noqa: E402

SRC = Path("outputs/stageplan/pilot_expansion"); OUT = Path("outputs/stageplan_v21/x"); OUT.mkdir(parents=True, exist_ok=True); TASKS = ["select_poker", "select_mahjong"]; SPLIT_SEED = 777
if __name__ == "__main__":
    rows, cov, split = [], {}, {}
    cams = cam_params("select_poker")
    for task in TASKS:
        scenes = pickle.load(open(SRC / f"scenes_{task}.pkl", "rb")); assert min(s["seed"] for s in scenes) >= 100050 and max(s["seed"] for s in scenes) < 3000 + 10 ** 6
        R = []
        for s in scenes:
            cands = []
            for c in s["candidates"]:
                proj = [project(cm, c["anchor"] + s["base"]) for cm in cams]; vis = [bool(p is not None and 8 <= p[0] < 472 and 8 <= p[1] < 472) for p in proj]
                cands.append({**c, "identity": parse_identity(task, c["name"]), "proj_uv_depth": proj, "in_frame": vis, "is_target": c["name"] == s["target"]})
            A = np.stack([c["anchor"] for c in cands]); D = np.linalg.norm(A[:, None] - A[None], axis=-1) + np.eye(len(cands)) * 9
            for i, c in enumerate(cands):
                rs = ([] if any(c["in_frame"][:2]) else ["not_in_frame_fixed_cams"]) + (["position_ambiguous"] if D[i].min() < MIN_SEP_M else []); c["valid"], c["invalid_reasons"] = not rs, rs
            ck = s["checks"]; ok = bool(ck["target_in_candidates"] and ck["native_matches_live"] and ck["unique_instructions"]); nv = sum(c["valid"] for c in cands) if ok else 0
            R.append({"scene_id": f"{task}_{s['seed']}", "task": task, "seed": s["seed"], "scene_ok": ok, "n_valid": nv, "n_cand": len(cands), "base_instruction": s["base_instruction"], "target": s["target"], "candidates": cands, "state": s["state"], "base": s["base"], "imgs224": s["imgs224"], "imgs480": s["imgs480"]})
        ids = sorted(r["scene_id"] for r in R); perm = np.random.default_rng(SPLIT_SEED).permutation(len(ids)); nval = len(ids) // 5
        for j, i in enumerate(perm): split[ids[i]] = "val" if j < nval else "train"
        rows += R; cov[task] = {"scenes": len(R), "scene_ok": sum(r["scene_ok"] for r in R), "cf_scenes(>=2 valid)": sum(r["n_valid"] >= 2 for r in R), "candidates": sum(r["n_cand"] for r in R), "valid_candidates": sum(r["n_valid"] for r in R),
                                "train_scenes": sum(split[r["scene_id"]] == "train" for r in R), "val_scenes": sum(split[r["scene_id"]] == "val" for r in R), "distinct_identities": len({c["name"] for r in R for c in r["candidates"]}), "seeds": [min(r["seed"] for r in R), max(r["seed"] for r in R)]}
    pickle.dump(rows, open(OUT / "referent_table_x.pkl", "wb")); json.dump(split, open(OUT / "split_x.json", "w"), indent=1); json.dump(cov, open(OUT / "coverage_x.json", "w"), indent=1); print(json.dumps(cov, indent=1))
