#!/usr/bin/env python
"""G28: expert-derived references (successful scripted-expert episodes, target-relative): (1) coarse route waypoint w25 = median EE-minus-keypoint position at the first step the expert is within 25cm of the keypoint; (2) prepare point = keypoint - 0.10*(0,1,0) (the expert's own pre-grasp point, passed within ~1cm in every successful episode); (3) wrist-image reference (u,v,depth) as a function of EE-keypoint distance (1cm bins, 0-14cm; 480px scale) = median over expert episodes, used as the visual-servo set point for arm D."""
import json, pickle
from pathlib import Path
import numpy as np

OUT = Path("outputs/eval/safediff_vla_g28"); R = {}
for t in ("select_book", "select_drink"):
    E = [e for e in pickle.load(open(OUT / t / "expert" / "expert_audit.pkl", "rb")) if e["success"] and e["t_close"] is not None]; W = []; bins = {b: [] for b in range(0, 15)}
    for e in E:
        S = e["steps"]; tc = e["t_close"]; kp = S[tc]["kp"]; ee = np.array([s["ee_p"] for s in S[:tc + 1]]); d = np.linalg.norm(ee - kp, axis=1); i = int(np.argmax(d <= 0.25)); W.append(ee[i] - kp)
        for k in range(tc + 1):
            uv = S[k]["wrist_uv"]
            if uv is not None and d[k] < 0.15: bins[int(d[k] * 100)].append(uv)
    R[t] = {"w25_rel": np.median(np.array(W), axis=0).tolist(), "w25_spread_cm": (np.std(np.array(W), axis=0) * 100).tolist(), "prepare_rel": [0.0, -0.10, 0.0], "ref": {str(b): np.median(np.array(v), axis=0).tolist() for b, v in bins.items() if len(v) >= 3}, "n_expert": len(E)}
    print(t, "n", len(E), "w25 rel (m)", np.round(R[t]["w25_rel"], 3), "spread cm", np.round(R[t]["w25_spread_cm"], 1)); print("  wrist ref (dist cm: u v depth_m):", {k: [round(x, 2) for x in v] for k, v in R[t]["ref"].items()})
json.dump(R, open(OUT / "expert_refs.json", "w"), indent=1)
