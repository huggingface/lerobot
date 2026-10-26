#!/usr/bin/env python
"""PHASE D0 step G: the CURRENT GroundedGrasp-v2 20k target head scored OFFLINE on dataset frames, same metric/units
as the D0 baselines (robot-frame xy / xyz error to the episode's grasp anchor, `find_grasp_target`'s definition).
Held-out episodes reproduce training's own rule (last ceil(10%) episodes of every task string, `make_train_eval_datasets`).
Frames: k steps before the anchor transition (k = 45, 30, 15, 5 -> inside the supervised window) and frame 0
(supervised only if the transition is within 50 steps). Dataset language (referent-bearing, "primitive: ...") is used.
No training; forward only.
"""
from __future__ import annotations

import json, math, sys
from pathlib import Path

import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, REPO_ID  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from phase_g06_counterfactual_instruction import forward_stages  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.processor.rename_processor import rename_batch_keys  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_phase_d0_training_data_audit")
KS = (45, 30, 15, 5)


def main():
    df = pd.read_json(OUT / "episodes.json"); df = df[df.has_anchor].copy()
    held = []
    for _, g in df.groupby("instruction"):
        g = g.sort_values("episode"); held += g.episode.tolist()[len(g) - math.ceil(len(g) * 0.1):]
    df["heldout"] = df.episode.isin(held)
    train = df[~df.heldout]; ho = df[df.heldout]
    print("held-out episodes per task:", ho.groupby("task").size().to_dict())
    ds = LeRobotDataset(REPO_ID)
    from_idx = ds.meta.episodes["dataset_from_index"]
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval()
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    cams = ds.meta.camera_keys
    # baselines fit on NON-held-out episodes only (the model's own training set)
    base = {}
    for task, g in train.groupby("task"):
        A = np.stack(g.anchor.to_numpy()); base[task] = {"A": A.mean(0), "C": {r: np.stack(x.anchor.to_numpy()).mean(0) for r, x in g.groupby("referent") if len(x) >= 3}}
    rows = []
    for task, g in ho.groupby("task"):
        for _, e in g.iterrows():
            tc = int(e.t_close); anc = np.array(e.anchor)
            for k in list(KS) + ["f0"]:
                f = 0 if k == "f0" else tc - k
                if f < 0: continue
                item = ds[int(from_idx[int(e.episode)]) + f]
                batch = {kk: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for kk, v in item.items()}
                for c in cams:
                    if c in batch and batch[c].dtype == torch.uint8: batch[c] = batch[c].to(torch.float32) / 255.0
                batch = pre(rename_batch_keys(batch, RENAME_MAP))
                with torch.no_grad():
                    pred = forward_stages(policy, batch)["pred_robot"]
                ee = np.array(item["observation.state"][:3])
                rows.append({"task": task, "episode": int(e.episode), "k": str(k), "steps_to_close": tc - f, "supervised": (tc - f) < 50,
                             "model_xy": float(np.linalg.norm((pred - anc)[:2])), "model_xyz": float(np.linalg.norm(pred - anc)),
                             "A_xy": float(np.linalg.norm((base[task]["A"] - anc)[:2])),
                             "C_xy": float(np.linalg.norm(((base[task]["C"].get(e.referent, base[task]["A"])) - anc)[:2])),
                             "S_ee_xy": float(np.linalg.norm((ee - anc)[:2]))})
        print(task, len([r for r in rows if r["task"] == task]), "rows", flush=True)
    pd.DataFrame(rows).to_json(OUT / "model_offline_target_error_rows.json", orient="records")
    r = pd.DataFrame(rows)
    out = {}
    for (task, k), g in r.groupby(["task", "k"]):
        out.setdefault(task, {})[k] = {"n": len(g), "supervised_frac": float(g.supervised.mean()), **{c + "_xy_mean_cm": float(g[c].mean() * 100) for c in ("model_xy", "A_xy", "C_xy", "S_ee_xy")},
                                       "model_xy_median_cm": float(g.model_xy.median() * 100), "model_xyz_mean_cm": float(g.model_xyz.mean() * 100)}
    (OUT / "model_offline_target_error.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
