#!/usr/bin/env python
"""Offline validation of the trained StagePlanHead on the episode-DISJOINT validation split (never the simulator eval seeds).
Reports, per task family (grouping only) and stage P0..P3: xyz median/mean error, fraction < 2/3/5 cm, rotation geodesic error,
and P3 release_valid precision/recall. Two frame sets: frame 0 (what the closed-loop controller uses) and random frames.
Usage: uv run python examples/safediff_vla/eval_stageplan_offline.py [--head outputs/train/safediff_stageplan_v1]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.stageplan import NUM_STAGES, SafeDiffVLAStagePlanPolicy  # noqa: E402

CKPT = "outputs/train/safediff_vla_grounded_grasp_v2_20k/checkpoints/020000/pretrained_model"
ROOT = Path("outputs/stageplan")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--head", default="outputs/train/safediff_stageplan_v1"); ap.add_argument("--per-ep-random", type=int, default=1); a = ap.parse_args()
    dev = "cuda"; labels = pickle.load(open(ROOT / "labels.pkl", "rb")); sp = json.load(open(ROOT / "split_constants.json")); val_eps = sp["val"]
    policy = SafeDiffVLAStagePlanPolicy.from_pretrained(CKPT).to(dev).eval(); policy.attach_stageplan(a.head, dev)
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT, preprocessor_overrides={"device_processor": {"device": dev}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    ds = LeRobotDataset("lerobot/vlabench_unified", episodes=val_eps, image_transforms=None, delta_timestamps=None, video_backend="torchcodec", return_uint8=True)
    ep_col = np.array(ds.hf_dataset["episode_index"]); fr_col = np.array(ds.hf_dataset["frame_index"]); rng = np.random.default_rng(0)
    idx0 = [int(np.where((ep_col == e) & (fr_col == 0))[0][0]) for e in val_eps]
    idxr = [int(rng.choice(np.where(ep_col == e)[0])) for e in val_eps for _ in range(a.per_ep_random)]
    res = {}
    for name, idx in (("frame0", idx0), ("random", idxr)):
        dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=16, num_workers=4)
        rows = []
        for b in dl:
            for k in [k for k in b if k.startswith("observation.images.") and b[k].dtype == torch.uint8]: b[k] = b[k].float() / 255.0
            eps = b["episode_index"].tolist(); bb = pre(b)
            out = policy.predict_stageplan(bb)
            P = out["poses"].cpu().numpy().astype(np.float64); rel = out["release_prob"].cpu().numpy()
            for i, e in enumerate(eps):
                l = labels[e]
                for s in range(NUM_STAGES):
                    g = np.asarray(l[f"P{s}"])
                    if np.isnan(g).any(): continue
                    dx = np.linalg.norm(P[i, s, :3] - g[:3]) * 100
                    dr = float(np.degrees((Rotation.from_euler("xyz", P[i, s, 3:]).inv() * Rotation.from_euler("xyz", g[3:])).magnitude()))
                    rows.append({"ep": e, "fam": l["fam"], "stage": s, "dxyz_cm": dx, "drot_deg": dr})
                rows.append({"ep": e, "fam": l["fam"], "stage": -1, "rel_pred": float(rel[i]), "rel_gt": bool(l["release_valid"])})
        res[name] = rows
    pickle.dump(res, open(Path(a.head) / "offline_val.pkl", "wb"))
    for name, rows in res.items():
        print(f"\n===== {name} ({len(rows)} rows) =====")
        print(f"{'family':16s}{'stage':>6s}{'n':>5s}{'med cm':>8s}{'mean cm':>9s}{'<2cm':>6s}{'<3cm':>6s}{'<5cm':>6s}{'rot med deg':>12s}")
        for fam in sorted({r["fam"] for r in rows}):
            for s in range(NUM_STAGES):
                x = [r for r in rows if r["fam"] == fam and r["stage"] == s]
                if not x: continue
                d = np.array([r["dxyz_cm"] for r in x]); rr = np.array([r["drot_deg"] for r in x])
                print(f"{fam:16s}{'P' + str(s):>6s}{len(x):5d}{np.median(d):8.1f}{d.mean():9.1f}{(d < 2).mean():6.2f}{(d < 3).mean():6.2f}{(d < 5).mean():6.2f}{np.median(rr):12.1f}")
            x = [r for r in rows if r["fam"] == fam and r["stage"] == -1]
            tp = sum(r["rel_pred"] > .5 and r["rel_gt"] for r in x); fp = sum(r["rel_pred"] > .5 and not r["rel_gt"] for r in x); fn = sum(r["rel_pred"] <= .5 and r["rel_gt"] for r in x)
            print(f"{fam:16s}{'rel':>6s}{len(x):5d}  gt_rate={np.mean([r['rel_gt'] for r in x]):.2f} precision={tp / max(tp + fp, 1):.2f} recall={tp / max(tp + fn, 1):.2f}")


if __name__ == "__main__":
    main()
