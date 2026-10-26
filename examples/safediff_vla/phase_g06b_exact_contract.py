#!/usr/bin/env python
"""PHASE G0.6b: G0.6 re-run with the EXACT training-time text contract (supersedes the prefix-less G0.6).

Identical design to `phase_g06_counterfactual_instruction.py` (saved G0.5 step-0 images + G0.5 GT
metadata; only the instruction string varies; no env.step, no rollout), except the instruction strings
are built on top of the *wrapper's own* corrected language (`env.call("task_description")`, i.e.
"primitive: <native instruction>"), so the referent is substituted INSIDE the exact training-format
string, and every constructed string is checked for exact membership in the training dataset's task set.
Text then goes through the unchanged policy preprocessor (NewLineTaskProcessor -> tokenizer).
Conditions: task_name (OLD-EVAL input) | cand::<each real same-type candidate> (CORRECTED-EVAL contract;
the instructed target's own string = condition B, the others = counterfactual condition C) | repeat (D).
"""
from __future__ import annotations

import copy, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, STAGE2_TASK_NAMES  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402
from phase_e0_instruction_plumbing_check import TASKS_PARQUET  # noqa: E402
from phase_g06_counterfactual_instruction import fill, forward_stages, make_template  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

G05 = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_g05_clean_grounding")
OUT = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_g06b_exact_training_contract")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_strings = set(pd.read_parquet(TASKS_PARQUET).index)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to("cuda").eval()
    results = []
    for task_name in STAGE2_TASK_NAMES:
        cfg = VLABenchEnv(task=task_name)
        env = make_env(cfg, n_envs=1, use_async_envs=False)[task_name][0]
        pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CHECKPOINT,
            preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
        env_pre, _ = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config)
        for seed in range(1000, 1008):
            set_seed(seed)
            raw, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
            impl = env.envs[0]
            while not hasattr(impl, "_env"):
                impl = impl.env
            fresh_target = impl._env.task.target_entity
            fresh_text = list(env.call("task_description"))[0]          # corrected wrapper text, "primitive: ..."
            tpl, form = make_template(fresh_text, fresh_target)
            assert fill(tpl, form, fresh_target) == fresh_text
            m0 = json.loads((G05 / task_name / f"seed{seed}" / "meta.json").read_text())["steps"][0]
            tname, cands = m0["target_name"], list(m0["cand_names"])
            gt = {n: np.asarray(m0["cand_pos_world"][n], dtype=float) for n in cands}
            base = np.asarray(m0["robot_base"], dtype=float)
            raw_pp = preprocess_observation(copy.deepcopy(raw))
            for ci, k in enumerate(RENAME_MAP):
                arr = np.asarray(Image.open(G05 / task_name / f"seed{seed}" / f"step00_camera{ci + 1}.png").convert("RGB"))
                raw_pp[k] = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(raw_pp[k].dtype)
            conds = {"task_name": task_name}
            for c in cands:
                conds[f"cand::{c}"] = fill(tpl, form, c)
            conds["repeat"] = fill(tpl, form, cands[0])
            out_c = {}
            for key, text in conds.items():
                o = copy.deepcopy(raw_pp); o["task"] = [text]
                b = pre(env_pre(o))
                res = forward_stages(policy, b)
                res["text"] = text
                res["n_lang_tokens_in_batch"] = int(b["observation.language.tokens"].shape[1])
                res["in_training_strings"] = text in train_strings
                out_c[key] = res
            results.append({"task": task_name, "seed": seed, "target": tname, "cands": cands, "gt_world": {n: gt[n].tolist() for n in cands},
                            "base": base.tolist(), "conds": out_c})
            print(f"[{task_name} s{seed}] K={len(cands)} target={tname} strings_in_training="
                  f"{sum(v['in_training_strings'] for k, v in out_c.items() if k.startswith('cand::'))}/{len(cands)} "
                  f"tokens {out_c['task_name']['n_lang_tokens_in_batch']}->{out_c[f'cand::{tname}']['n_lang_tokens_in_batch']}", flush=True)
        env.close()
    lat, ser = {}, []
    for r in results:
        r2 = {k: v for k, v in r.items() if k != "conds"}; r2["conds"] = {}
        for ck, cv in r["conds"].items():
            lat[f"{r['task']}|{r['seed']}|{ck}|image_latent"] = cv["image_latent"].astype(np.float16)
            r2["conds"][ck] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cv.items() if k != "image_latent"}
        ser.append(r2)
    (OUT / "g06_raw.json").write_text(json.dumps(ser)); np.savez_compressed(OUT / "g06_image_latents.npz", **lat)
    print("saved", OUT)


if __name__ == "__main__":
    main()
