#!/usr/bin/env python
"""PHASE D0 step H: is the 20k target head language-sensitive on IN-DISTRIBUTION (dataset) frames?
Held-out dataset frames (same split rule as training), image/state fixed; only the instruction text is swapped to
other REAL referents of the same task (strings that exist in the dataset). True anchor known for the original
referent: if the head used language, error to the TRUE anchor should grow when the referent is swapped (the
swapped referent is a different object elsewhere in the scene); if it ignores language, error and prediction stay put.
Forward only; no training, no rollout."""
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
rng = np.random.default_rng(0)


def main():
    df = pd.read_json(OUT / "episodes.json"); df = df[df.has_anchor]
    held = []
    for _, g in df.groupby("instruction"):
        g = g.sort_values("episode"); held += g.episode.tolist()[len(g) - math.ceil(len(g) * 0.1):]
    ho = df[df.episode.isin(held)]
    instr_by_task = {t: sorted(g.instruction.unique()) for t, g in df.groupby("task")}
    ds = LeRobotDataset(REPO_ID); from_idx = ds.meta.episodes["dataset_from_index"]
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval()
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    cams = ds.meta.camera_keys; rows = []
    for task, g in ho.groupby("task"):
        for _, e in g.iterrows():
            tc = int(e.t_close); anc = np.array(e.anchor)
            for k in (30, "f0"):
                f = 0 if k == "f0" else tc - 30
                if f < 0: continue
                item = ds[int(from_idx[int(e.episode)]) + f]
                others = [s for s in instr_by_task[task] if s != e.instruction]
                texts = {"true": e.instruction, **{f"swap{i}": rng.choice(others) for i in range(2)}}
                preds = {}
                for name, text in texts.items():
                    it = dict(item); it["task"] = text
                    batch = {kk: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for kk, v in it.items()}
                    for c in cams:
                        if c in batch and batch[c].dtype == torch.uint8: batch[c] = batch[c].to(torch.float32) / 255.0
                    with torch.no_grad():
                        preds[name] = forward_stages(policy, pre(rename_batch_keys(batch, RENAME_MAP)))["pred_robot"]
                for sw in ("swap0", "swap1"):
                    rows.append({"task": task, "k": str(k), "episode": int(e.episode),
                                 "err_true_xy": float(np.linalg.norm((preds["true"] - anc)[:2])), "err_swap_xy": float(np.linalg.norm((preds[sw] - anc)[:2])),
                                 "dpred_xy": float(np.linalg.norm((preds[sw] - preds["true"])[:2]))})
        print(task, len(rows), flush=True)
    r = pd.DataFrame(rows)
    out = {t: {k: {"n": len(x), "err_true_xy_cm": float(x.err_true_xy.mean() * 100), "err_swapped_xy_cm": float(x.err_swap_xy.mean() * 100),
                   "mean_pred_shift_xy_cm": float(x.dpred_xy.mean() * 100), "median_pred_shift_cm": float(x.dpred_xy.median() * 100),
                   "frac_shift_gt_3cm": float((x.dpred_xy > 0.03).mean())} for k, x in g.groupby("k")} for t, g in r.groupby("task")}
    (OUT / "indist_language_swap.json").write_text(json.dumps(out, indent=1)); r.to_json(OUT / "indist_language_swap_rows.json", orient="records")
    for t, v in out.items():
        for k, x in v.items():
            print(f"{t:14s} k={k:>3s} n={x['n']:3d} err true {x['err_true_xy_cm']:5.1f} -> swapped {x['err_swapped_xy_cm']:5.1f} cm | pred shift mean {x['mean_pred_shift_xy_cm']:4.1f} med {x['median_pred_shift_cm']:4.1f} cm | >3cm {x['frac_shift_gt_3cm']:.2f}")


if __name__ == "__main__":
    main()
