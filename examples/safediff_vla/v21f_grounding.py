"""[v2.1 final] Poker/Mahjong grounding metrics + funnel split by referent correctness. Diagnostic GT (target keypoint) is read from result.json['geo'] (logging only, written after action generation)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from v21g_geometry import CAMS, project
R = Path("outputs/eval/safediff_stageplan_v21_final"); CA = {t: np.array(v["C-A"]["median_xyz_cm"]) / 100 for t, v in json.load(open("outputs/stageplan_v21/x/label_frame_sim.json")).items()}; OFF = {"select_poker": np.array([0, 0, 0.02]), "select_mahjong": np.array([0, 0.004, 0.01])}
out = {}
for t in ("select_poker", "select_mahjong"):
    rows = []
    for s in range(3000, 3020):
        r = json.load(open(R / t / f"seed{s}" / "result.json")); pi = r["provider_info"]; kp = np.array(r["geo"][0]["kp_robot"][0]); base = np.array(r["geo"][0]["base"]); g = np.array(pi["g_final"]); gt = kp + CA[t]
        c2d = project(CAMS[0], kp - OFF[t] + base)[:2]; pf = pi["front_props"]; cov = any(np.hypot(p[0] - c2d[0], p[1] - c2d[1]) <= 10 for p in pf); sel = pi.get("selected"); sel_ok = bool(sel is not None and np.hypot(pf[sel][0] - c2d[0], pf[sel][1] - c2d[1]) <= 10)
        err = float(np.linalg.norm(g - gt) * 100); sl = json.load(open(R / t / f"seed{s}" / "stage_log.json")); sn = json.load(open(R / t / f"seed{s}" / "snapshots.json"))
        z = np.array([x["obj_p"][2] for x in sn]); grasped = any(x["grasped"] for x in sn); lifted = float(z.max() - z[0]) > 0.03
        rows.append({"seed": s, "native": bool(r["success"]), "covered": cov, "ref_ok": sel_ok, "ref_ok_by_xyz3cm": bool(np.linalg.norm(g - kp) < 0.03), "err_cm": err, "fallback": pi["source"] != "grounded", "n_prop": pi["n_front"], "grasped": grasped, "lifted": lifted, "close": bool(r["close_events"]), "timeout": (not r["success"]) and r["len"] >= 500, "len": r["len"]})
    e = np.array([x["err_cm"] for x in rows]); ok = [x for x in rows if x["ref_ok"]]; bad = [x for x in rows if not x["ref_ok"]]
    out[t] = {"n": 20, "proposal_coverage": float(np.mean([x["covered"] for x in rows])), "referent_acc": float(np.mean([x["ref_ok"] for x in rows])), "referent_acc_xyz_rule_agrees": all(x["ref_ok"] == x["ref_ok_by_xyz3cm"] for x in rows), "fallback_rate": float(np.mean([x["fallback"] for x in rows])), "mean_proposals": float(np.mean([x["n_prop"] for x in rows])),
              "grasp_err_median_cm_all": float(np.median(e)), "<1cm": float(np.mean(e < 1)), "<2cm": float(np.mean(e < 2)), "<3cm": float(np.mean(e < 3)), "grasp_err_median_cm_correct_referent": float(np.median([x["err_cm"] for x in ok])) if ok else None, "wrong_referent_count": len(bad),
              "outcome": {"WRONG_REFERENT": len(bad), "CORRECT_REFERENT_BUT_EXECUTION_FAIL": sum(not x["native"] for x in ok), "success": sum(x["native"] for x in ok)}, "funnel_correct_referent": {k: sum(x[k] for x in ok) for k in ("close", "grasped", "lifted", "native")}, "n_correct": len(ok),
              "funnel_wrong_referent": {k: sum(x[k] for x in bad) for k in ("close", "grasped", "lifted", "native")}, "n_wrong": len(bad), "wrong_referent_seeds": [x["seed"] for x in bad], "native_failures": [(x["seed"], x["ref_ok"], x["timeout"]) for x in rows if not x["native"]]}
json.dump(out, open(R / "grounding_metrics.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
