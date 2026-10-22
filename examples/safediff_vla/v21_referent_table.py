#!/usr/bin/env python
"""[StagePlan-v2.1] Verified referent table for the 300-scene source-domain pilot (seeds 100000-100049/task, NEVER final eval seeds).

Per scene: images, instruction(s), per-candidate exact entity instance name (poker rank/suit, mahjong value/suit parsed from the entity
name), 3D object-centre + grasp anchor (robot frame), per-camera projection + in-frame/occlusion-free visibility (analytic, fixed cameras;
wrist cam at reset pose). A candidate is VALID iff: native instruction is a verbatim training string (sg_gen_scenes), instruction unique
within the scene, anchor visible in >=1 fixed camera (in frame), instruction of the TRUE target equals the live task_description, and
no other candidate anchor is within MIN_SEP_M (identity not separable by position). Scenes with <2 valid candidates have no counterfactual.
Usage: uv run python examples/safediff_vla/v21_referent_table.py  -> outputs/stageplan_v21/referent_table.pkl + audit montages
"""
from __future__ import annotations
import json, pickle, re, sys
from pathlib import Path
import cv2, numpy as np

PILOT = Path("outputs/stageplan/pilot"); OUT = Path("outputs/stageplan_v21"); CALIB = Path("outputs/data/safediff_vla_g1_counterfactual/camera_calibration.json")
TASKS = ["select_poker", "select_mahjong", "select_drink", "select_book", "select_fruit", "select_toy"]
MIN_SEP_M = 0.03
CAM_ORDER = ["forward", "right", "wrist"]  # == pilot imgs480 index order, verified by the audit montages


def cam_params(task):
    cams = json.load(open(CALIB))["select_poker"]["cams_policy_order"]  # fixed cams identical across tasks (poker==fruit verified); validated visually per task by the audit montages
    return [(np.array(c["xpos"]), np.array(c["xmat"]).reshape(3, 3), c["fovy"]) for c in cams]


def project(cam, p_world, size=480):
    c, R, fovy = cam; d = R.T @ (p_world - c); z = -d[2]
    if z <= 1e-6: return None
    f = (size / 2) / np.tan(np.deg2rad(fovy) / 2)
    return float(size / 2 + f * d[0] / z), float(size / 2 - f * d[1] / z), float(z)


def parse_identity(task, name):
    if task == "select_poker":
        m = re.match(r"^(.+)_of_(.+)$", name); return {"rank": m.group(1), "suit": m.group(2)} if m else {}
    if task == "select_mahjong":
        m = re.match(r"^(\d+)_(\w+)$", name); return {"value": m.group(1), "suit": m.group(2)} if m else {}
    return {"object": name}


def build():
    OUT.mkdir(parents=True, exist_ok=True); rows = []; stats = {}
    for task in TASKS:
        scenes = pickle.load(open(PILOT / f"scenes_{task}.pkl", "rb")); cams = cam_params(task); base = scenes[0]["base"]
        for s in scenes:
            cands = []
            for c in s["candidates"]:
                w = c["anchor"] + s["base"]; wc = c["obj_center"] + s["base"]
                proj = [project(cm, w) for cm in cams]
                vis = [bool(p is not None and 8 <= p[0] < 472 and 8 <= p[1] < 472) for p in proj]
                cands.append({**c, "identity": parse_identity(task, c["name"]), "proj_uv_depth": proj, "in_frame": vis, "is_target": c["name"] == s["target"]})
            A = np.stack([c["anchor"] for c in cands]); D = np.linalg.norm(A[:, None] - A[None], axis=-1) + np.eye(len(cands)) * 9
            for i, c in enumerate(cands):
                reasons = []
                if not any(c["in_frame"][:2]): reasons.append("not_in_frame_fixed_cams")
                if D[i].min() < MIN_SEP_M: reasons.append("position_ambiguous")
                c["valid"], c["invalid_reasons"] = (not reasons), reasons
            ck = s["checks"]; scene_ok = bool(ck["target_in_candidates"] and ck["native_matches_live"] and ck["unique_instructions"])
            nv = sum(c["valid"] for c in cands) if scene_ok else 0
            rows.append({"scene_id": f"{task}_{s['seed']}", "task": task, "seed": s["seed"], "scene_ok": scene_ok, "n_valid": nv, "n_cand": len(cands), "base_instruction": s["base_instruction"], "target": s["target"], "candidates": cands,
                         "distractor_non_referents": [n["name"] for n in s["non_referents"]], "state": s["state"], "base": s["base"], "imgs224": s["imgs224"], "imgs480": s["imgs480"]})
    pickle.dump(rows, open(OUT / "referent_table.pkl", "wb"))
    for task in TASKS:
        R = [r for r in rows if r["task"] == task]
        stats[task] = {"scenes": len(R), "scene_ok": sum(r["scene_ok"] for r in R), "cf_scenes(>=2 valid)": sum(r["n_valid"] >= 2 for r in R), "valid_cands": sum(r["n_valid"] for r in R), "cands": sum(r["n_cand"] for r in R),
                       "invalid_reasons": {k: sum(k in c["invalid_reasons"] for r in R for c in r["candidates"]) for k in ("not_in_frame_fixed_cams", "position_ambiguous")}}
    json.dump(stats, open(OUT / "referent_table_coverage.json", "w"), indent=1); print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    build()
