#!/usr/bin/env python
"""Counterfactual-language check (BEFORE any closed-loop evaluation): same observation, only the instruction changes to name
a different referent; does the predicted P1 move toward that referent? Scenes: DEV seeds 1000-1004 per task (NOT the final
eval seeds). Privileged simulator reads (entity keypoints) are used ONLY to score the prediction, never given to the policy.

Per task, per scene: candidate referents = scene entities whose name appears in the dataset's task strings for that family.
 - selection accuracy: P1(instr=c) is nearest (among candidate keypoints) to c's own keypoint
 - P1 error to own referent (cm)
 - pairwise direction cosine between the P1 shift (instr t -> c) and the true referent displacement (kp_t -> kp_c)
 - spread ratio: std of P1 over instructions / std of the true candidate keypoints (1 = fully instruction-driven, 0 = scene prior)
Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/stageplan_counterfactual.py --head outputs/train/safediff_stageplan_v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, TASK_SUBSTRINGS  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT, robot_base  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.stageplan import SafeDiffVLAStagePlanPolicy  # noqa: E402

TASKS = ["select_poker", "select_mahjong", "select_drink", "select_book", "select_fruit", "select_toy"]


def family_text(task: str) -> str:
    import glob
    t = pd.read_parquet(glob.glob("/home/user_lerobot/.cache/huggingface/hub/datasets--lerobot--vlabench_unified/snapshots/*/meta/tasks.parquet")[0]).reset_index()
    return " ".join(s.lower() for s in t["task"] if TASK_SUBSTRINGS[task] in s)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--head", default="outputs/train/safediff_stageplan_v1"); ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004]); ap.add_argument("--out", default="outputs/stageplan/counterfactual.json"); a = ap.parse_args()
    policy = SafeDiffVLAStagePlanPolicy.from_pretrained(CHECKPOINT).to("cuda").eval(); policy.attach_stageplan(a.head, "cuda")
    pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT, preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
    results = []
    for task in a.tasks:
        env_cfg = VLABenchEnv(task=task); env = make_env(env_cfg, n_envs=1, use_async_envs=False)[task][0]
        env_pre, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
        vocab = family_text(task)
        for seed in a.seeds:
            obs, _ = env.reset(seed=[seed]); e = env.unwrapped.envs[0]; t = e._env.task; ph = e._env.physics; base = robot_base(e)
            tgt, text = t.target_entity, e.task_description
            cands = {}
            for k, ent in t.entities.items():
                nm = k.replace("_", " ").lower()
                if k in ("table",) or not (nm in vocab or k.lower() in vocab): continue
                try: cands[k] = np.asarray(ent.get_grasped_keypoints(ph)[0], float) - base
                except Exception:  # noqa: BLE001
                    pass
            if tgt not in cands or len(cands) < 2: print(f"[{task} {seed}] skipped (candidates={list(cands)})"); continue
            base_obs = preprocess_observation(obs)
            pred = {}
            for c in cands:
                txt = text
                for form_t, form_c in ((tgt, c), (tgt.replace("_", " "), c.replace("_", " "))):
                    txt = txt.replace(form_t, form_c)
                o = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base_obs.items()}; o["task"] = [txt]
                batch = pre(env_pre(o))
                p = policy.predict_stageplan(batch)["poses"][0, 1, :3].float().cpu().numpy().astype(float)
                pred[c] = p
            names = list(cands); K = np.stack([cands[n] for n in names]); Pm = np.stack([pred[n] for n in names])
            D = np.linalg.norm(Pm[:, None] - K[None], axis=-1)  # [instr, kp]
            sel = float(np.mean(D.argmin(1) == np.arange(len(names)))); own = float(np.mean(np.diag(D)) * 100)
            cos = []
            for i, ti in enumerate(names):
                for j, cj in enumerate(names):
                    if i == j: continue
                    s, d = Pm[j] - Pm[i], K[j] - K[i]
                    if np.linalg.norm(d) > 0.03: cos.append(float(s @ d / (np.linalg.norm(s) * np.linalg.norm(d) + 1e-9)))
            spread = float(np.linalg.norm(Pm.std(0)) / (np.linalg.norm(K.std(0)) + 1e-9))
            results.append({"task": task, "seed": seed, "n_cand": len(names), "select_acc": sel, "p1_err_cm": own, "dir_cos_mean": float(np.mean(cos)) if cos else None, "spread_ratio": spread, "chance": 1 / len(names)})
            print(results[-1], flush=True)
        env.close()
    Path(a.out).write_text(json.dumps(results, indent=1))
    df = pd.DataFrame(results)
    print(df.groupby("task").agg(n=("seed", "size"), cands=("n_cand", "mean"), select_acc=("select_acc", "mean"), chance=("chance", "mean"), p1_err_cm=("p1_err_cm", "mean"), dir_cos=("dir_cos_mean", "mean"), spread=("spread_ratio", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
