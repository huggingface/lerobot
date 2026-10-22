#!/usr/bin/env python
"""[StagePlan-v2.1] Label-audit montages: candidate anchors projected onto the pilot images (target=green, distractors=orange), labelled with the
native instruction's referent. Samples: poker 50, mahjong 50, other tasks 20 each (seeded random). Numeric checks run on ALL scenes."""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from v21_referent_table import OUT, TASKS, cam_params

N = {"select_poker": 50, "select_mahjong": 50}
rows = pickle.load(open(OUT / "referent_table.pkl", "rb")); rng = np.random.default_rng(0); (OUT / "audit").mkdir(exist_ok=True); sel = {}
for task in TASKS:
    R = [r for r in rows if r["task"] == task]; k = N.get(task, 20); idx = sorted(rng.choice(len(R), size=k, replace=False).tolist()); sel[task] = [R[i]["scene_id"] for i in idx]
    tiles = []
    for i in idx:
        r = R[i]; ims = []
        for ci in range(2):
            im = cv2.cvtColor(r["imgs480"][ci].copy(), cv2.COLOR_RGB2BGR)
            for c in r["candidates"]:
                p = c["proj_uv_depth"][ci]
                if p is None: continue
                col = (0, 255, 0) if c["is_target"] else (0, 140, 255)
                cv2.circle(im, (int(p[0]), int(p[1])), 7, col, 2); cv2.putText(im, c["name"][:14], (int(p[0]) - 30, int(p[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            ims.append(im)
        t = np.concatenate(ims, 1); t = cv2.resize(t, (640, 320)); cv2.putText(t, f"{r['scene_id']} | {r['base_instruction'][:45]}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1); tiles.append(t)
    for sh in range(0, len(tiles), 6):
        grid = tiles[sh: sh + 6]; grid += [np.zeros_like(tiles[0])] * (6 - len(grid))
        cv2.imwrite(str(OUT / "audit" / f"{task}_sheet{sh // 6}.jpg"), np.concatenate([np.concatenate(grid[:2], 1), np.concatenate(grid[2:4], 1), np.concatenate(grid[4:], 1)], 0))
import json; json.dump(sel, open(OUT / "audit" / "inspected_scene_ids.json", "w"), indent=1); print({t: len(v) for t, v in sel.items()})
