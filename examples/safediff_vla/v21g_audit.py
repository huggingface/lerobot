"""[v2.1 anchor phase] (3) anchor statistics + (5) numerical geometry verification. TRAIN-safe source data only (pilot/expansion/prop blocks, expert audit rollouts)."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from v21_common import OUT as V21  # noqa: E402
from v21g_geometry import CAMS, project, triangulate  # noqa: E402
import v21q_prop_iter as Q  # noqa: E402

X = V21 / "x"; TASKS = ("select_poker", "select_mahjong")


def st(a):
    a = np.asarray(a) * 100; n = np.linalg.norm(a, axis=1)
    return {"mean_xyz_cm": a.mean(0).round(3).tolist(), "median_xyz_cm": np.median(a, 0).round(3).tolist(), "std_xyz_cm": a.std(0).round(3).tolist(), "norm_cm median/p90/max": [round(float(np.median(n)), 3), round(float(np.percentile(n, 90)), 3), round(float(n.max()), 3)]}


def main():
    rows = pickle.load(open(X / "referent_table_x.pkl", "rb")) + Q.build_rows("A"); out = {"anchor_minus_center": {}, "by_identity_range_cm": {}, "expert_EE_minus_anchor": {}}
    for t in TASKS:
        off, by = [], {}
        for r in rows:
            if r["task"] != t: continue
            for c in r["candidates"]:
                o = c["anchor"] - c["obj_center"]; off.append(o); by.setdefault(c["name"], []).append(o)
        out["anchor_minus_center"][t] = st(off) | {"n": len(off)}; spread = {k: np.ptp(np.array(v), axis=0) * 100 for k, v in by.items()}; out["by_identity_range_cm"][t] = {"max over identities of per-identity (max-min) xyz": np.max(list(spread.values()), axis=0).round(4).tolist(), "n_identities": len(by),
            "between-identity std of per-identity mean offset (cm)": np.std([np.mean(v, 0) for v in by.values()], axis=0).__mul__(100).round(4).tolist()}
        e = [x for x in pickle.load(open(f"outputs/stageplan/pilot_expert_audit/{t}/expert/expert_audit.pkl", "rb")) if x["t_close"] is not None]; ee = [x["steps"][x["t_close"]]["ee_p"] - x["steps"][x["t_close"]]["kp"] for x in e]; out["expert_EE_minus_anchor"][t] = st(ee) | {"n": len(ee), "per-episode targets": [x["target"] for x in e]}
    # (5) numerical frame verification: GT 3D (robot frame) -> project in both cams -> triangulate -> recover; plus pixel-noise sensitivity
    rng = np.random.default_rng(0); res = {}
    for t in TASKS:
        e0, en = [], {0.5: [], 1.0: [], 1.5: [], 3.0: []}
        for r in [r for r in rows if r["task"] == t][:200]:
            for c in r["candidates"]:
                w = c["obj_center"] + r["base"]; pf, pr = project(CAMS[0], w), project(CAMS[1], w); p, g = triangulate(pf[:2], pr[:2]); e0.append(np.linalg.norm(p - w))
                for s in en: p2, _ = triangulate(np.array(pf[:2]) + rng.normal(0, s, 2), np.array(pr[:2]) + rng.normal(0, s, 2)); en[s].append(np.linalg.norm(p2 - w) * 100 if p2 is not None else np.nan)
        res[t] = {"noise-free reconstruction error (m) max": float(np.max(e0)), "triangulation error cm vs pixel noise sigma (median / p90)": {str(s): [round(float(np.nanmedian(v)), 3), round(float(np.nanpercentile(v, 90)), 3)] for s, v in en.items()}}
    out["geometry_verification"] = res; out["transforms"] = [
        {"step": "proposal 2D centre", "source": "frozen front head (g64, thr 0.10) / new right-camera head", "frame": "pixel, 480x480 original render", "units": "px"},
        {"step": "pixel -> camera ray", "source": "calibration json (fovy 45 deg, f = 240/tan(22.5deg) = 579.4 px)", "frame": "MuJoCo camera (x right, y up, -z forward)", "normalization": "unit direction"},
        {"step": "camera -> world", "source": "cam_xpos/cam_xmat from physics (fixed cameras, scene-independent)", "frame": "world", "units": "m"},
        {"step": "two-ray closest-approach midpoint", "source": "front + right fixed cameras", "frame": "world", "units": "m", "gap gate": "pairs with ray gap > 3 cm rejected"},
        {"step": "world -> robot base frame", "source": "fixed robot mount (0,-0.4,0.78) (constant of the setup, not target state)", "frame": "robot base (policy frame)", "units": "m"},
        {"step": "policy normalisation", "source": "policy.action_pos_mean/std (applied only when feeding stage variables to StagePlan-v2)", "units": "normalised"}]
    json.dump(out, open(X / "anchor_audit.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float)[:6000])

if __name__ == "__main__": main()
