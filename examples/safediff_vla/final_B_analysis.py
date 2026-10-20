#!/usr/bin/env python
"""Summary of the ORACLE-GROUNDED stage-wise execution diagnostic (`run_final_oracle_B.py`). NOT learned-policy SR.
All of these controllers use privileged simulator keypoints (get_grasped_keypoints); drink additionally an expert-derived
waypoint + pose; book an expert grasp pose. Prints per-task stage results for seeds 3000-3019.
Usage: uv run python examples/safediff_vla/final_B_analysis.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("outputs/eval/safediff_final_B")
SPEC = {"select_poker": ("g21", "ORACLE keypoint + geometry (G21 geo)"), "select_mahjong": ("g21", "ORACLE keypoint + geometry (G21 geo)"),
        "select_drink": ("g28C", "ORACLE keypoint + EXPERT waypoint + EXPERT pose, 3cm close (G28 C)"), "select_book": ("g29B", "ORACLE keypoint + EXPERT grasp pose + pose-completion close (G29 B)"),
        "select_fruit": ("g24", "ORACLE keypoint + geometry + release gate (G24)"), "select_toy": ("g24", "ORACLE keypoint + geometry + release gate (G24)")}
SEEDS = range(3000, 3020)


def rec_from_steps(R, tgt=None, S=None):
    tp = np.array([r["target_pos"] for r in R]); g = np.array([bool(r["is_grasped"]) for r in R]); ci = next((i for i, r in enumerate(R) if r["reactive_close_event"]), None)
    c0 = R[0].get("candidates_world") or {}; tgt = tgt or R[0].get("target_entity")
    oth = [k for k in c0 if k != tgt]; do = np.array([[np.linalg.norm(np.array(r["candidates_world"][k]) - np.array(c0[k])) for k in oth] for r in R]) if oth else np.zeros((len(R), 0))
    end = ci if ci is not None else len(R) - 1; dt = np.linalg.norm(tp - tp[0], axis=1) * 100
    m = {"close": ci is not None, "grasped": bool(g[ci:].any()) if ci is not None else False, "lift_cm": float((tp[ci:, 2].max() - tp[0, 2]) * 100) if ci is not None else 0.0, "min_dist_cm": (float(min(np.linalg.norm(x["ee"] - x["kp"]) for x in S[: end + 1]) * 100) if S else float(min(r["ee_to_target_dist"] for r in R[: end + 1]) * 100)),
         "collision": bool(dt[: end + 1].max() > 1.5 or (do[: end + 1].max() * 100 > 1.5 if do.size else False)), "held": int(g.sum())}
    m["reach3"] = m["min_dist_cm"] < 3; m["reach5"] = m["min_dist_cm"] < 5
    return m


def load(task, seed):
    tag = SPEC[task][0]; d = ROOT / tag / f"seed{seed}"
    if not (d / f"{task}.done").exists(): return None
    if tag == "g21":
        e = json.load(open(d / task / "geo" / "episodes_raw.json"))[0]; m = rec_from_steps(e["steps"]); m.update(native=bool(e["success"]), len=e["episode_len"]); return m
    if tag in ("g28C", "g29B"):
        p = d / task / ("C" if tag == "g28C" else "B") / ("" if tag == "g28C" else f"seed{seed}") / "g28_trace.pkl"; L = pickle.load(open(p, "rb")); e = L["episodes"][0]; m = rec_from_steps(e["rec"], e.get("target"), [x for x in L["steps"] if x["ep"] == 0]); m.update(native=bool(e["success"]), len=e["len"]); return m
    L = pickle.load(open(d / task / "g24_log.pkl", "rb")); e = L["episodes"][0]; S = [s for s in L["steps"] if s["ep"] == 0]
    g = np.array([bool(s["grasped"]) for s in S]); dxy = np.array([s["dxy_goal"] for s in S], dtype=float) * 100; ex = np.array([s["exec"] if s["exec"] is not None else np.nan for s in S], float); ci = e["close"]
    obj = np.array([s["obj"] for s in S]); ee = np.array([s["ee"] for s in S]); d_ee = np.linalg.norm(obj - ee, axis=1) * 100
    m = {"native": bool(e["success"]), "len": e["len"], "close": ci is not None, "grasped": bool(g.any()), "lift_cm": float((obj[:, 2].max() - obj[0, 2]) * 100), "min_dist_cm": float(d_ee[: (ci or len(S))].min()), "collision": None, "held": int(g.sum()),
         "transported": bool(np.nanmin(dxy) < 6), "exec_open_after_close": bool(ci is not None and (ex[ci + 2:] > 0.5).any())}
    m["reach3"] = m["min_dist_cm"] < 3; m["reach5"] = m["min_dist_cm"] < 5; return m


def main() -> None:
    out = {}
    print(f"{'task':16s}{'N':>3s}{'native':>8s}{'reach<3':>8s}{'close':>6s}{'grasp':>6s}{'lift>5':>7s}{'transp':>7s}{'execOPEN':>9s}{'collis':>7s}  controller (ORACLE-GROUNDED, not learned)")
    for t, (_, desc) in SPEC.items():
        ms = [m for s in SEEDS if (m := load(t, s))]
        for s, m in zip([s for s in SEEDS if load(t, s)], ms): m["seed"] = s
        out[t] = ms
        if not ms: print(f"{t:16s}  not evaluated"); continue
        f = lambda k: sum(bool(m[k]) for m in ms)
        tr = f"{sum(bool(m.get('transported')) for m in ms)}" if t in ("select_fruit", "select_toy") else "NA"
        eo = f"{sum(bool(m.get('exec_open_after_close')) for m in ms)}" if t in ("select_fruit", "select_toy") else "NA"
        co = "NA" if ms[0]["collision"] is None else str(f("collision"))
        print(f"{t:16s}{len(ms):3d}{f('native'):8d}{f('reach3'):8d}{f('close'):6d}{f('grasped'):6d}{sum(m['lift_cm'] > 5 for m in ms):7d}{tr:>7s}{eo:>9s}{co:>7s}  {desc}")
    Path(ROOT / "B_metrics.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
