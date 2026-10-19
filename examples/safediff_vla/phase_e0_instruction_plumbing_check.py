#!/usr/bin/env python
"""PHASE E0 step 2: text-contract sanity check of the fixed VLABench language plumbing. NO rollout, NO
policy forward: reset + the exact retrieval path of `lerobot_eval.rollout`
(`env.call("task_description")` -> env_preprocessor -> preprocessor/tokenizer), for the 6 tasks.
Per (task, seed) it records and compares:
    native         `task.get_instruction()` as VLABench's own evaluator would see it
    wrapper        what `env.call("task_description")` returned (+ provenance label)
    processor_in   `observation["task"]` handed to the (env_)preprocessor
    tokenizer_out  the string decoded back from the tokenizer ids the policy actually receives
    training       is `tokenizer_out` (minus the processor's trailing newline) an EXACT string of
                   `lerobot/vlabench_unified`'s tasks.parquet (what training tokenized, also + newline)?
Usage: MUJOCO_GL=egl uv run python examples/safediff_vla/phase_e0_instruction_plumbing_check.py
"""
from __future__ import annotations

import copy, json, random, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, STAGE2_TASK_NAMES  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_e0_instruction_plumbing")
TASKS_PARQUET = "/home/user_lerobot/.cache/huggingface/hub/datasets--lerobot--vlabench_unified/snapshots/f61ad898199e746c20af3112f278cc86eb3ca3c4/meta/tasks.parquet"
G05 = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_g05_clean_grounding")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_strings = set(pd.read_parquet(TASKS_PARQUET).index)
    cfg_pol = SafeDiffVLAConfig.from_pretrained(CHECKPOINT)
    tok = AutoTokenizer.from_pretrained(cfg_pol.vlm_model_name)
    rows = []
    for task_name in STAGE2_TASK_NAMES:
        cfg = VLABenchEnv(task=task_name)
        env = make_env(cfg, n_envs=1, use_async_envs=False)[task_name][0]
        pre, _ = make_pre_post_processors(policy_cfg=cfg_pol, pretrained_path=CHECKPOINT,
            preprocessor_overrides={"device_processor": {"device": "cpu"}, "rename_observations_processor": {"rename_map": RENAME_MAP}})
        env_pre, _ = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=cfg_pol)
        for seed in (1000, 1001, 1002):
            set_seed(seed)
            raw, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
            impl = env.envs[0]
            while not hasattr(impl, "_env"):
                impl = impl.env
            st = random.getstate()
            native = impl._env.task.get_instruction(); random.setstate(st)   # what VLABench's evaluator would feed
            wrapper = list(env.call("task_description"))[0]                   # <- the line rollout() uses
            source = impl.task_description_source
            o = preprocess_observation(copy.deepcopy(raw)); o["task"] = [wrapper]
            o = env_pre(o); processor_in = o["task"][0]
            b = pre(o)
            ids = b["observation.language.tokens"][0]; m = b["observation.language.attention_mask"][0].bool()
            decoded = tok.decode(ids[m])
            rows.append({"task": task_name, "seed": seed, "native": native, "wrapper": wrapper, "source": source,
                         "processor_in": processor_in, "tokenizer_decoded": decoded, "n_tokens": int(m.sum()),
                         "decoded_minus_newline_in_training_strings": decoded.rstrip("\n") in train_strings,
                         "wrapper_equals_prefix+native": wrapper == f"primitive: {native}",
                         "prefix_count": wrapper.count("primitive:")})
            r = rows[-1]
            print(f"[{task_name} s{seed}] native={native!r}\n    wrapper={wrapper!r} ({source})\n    decoded={decoded!r} tokens={r['n_tokens']} "
                  f"in_training_strings={r['decoded_minus_newline_in_training_strings']}", flush=True)
        # scene-RNG check: the fix must not perturb scene generation (poker scene is process-deterministic)
        if task_name == "select_poker":
            set_seed(1000); raw, _ = env.reset(seed=[1000], options={NEW_ROLLOUT_OPTION: True})
            o = preprocess_observation(copy.deepcopy(raw)); diffs = []
            for ci, k in enumerate(RENAME_MAP):
                saved = np.asarray(Image.open(G05 / task_name / "seed1000" / f"step00_camera{ci + 1}.png").convert("RGB")).astype(int)
                cur = (o[k][0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(int)
                diffs.append(float(np.abs(saved - cur).mean()))
            print("poker seed1000 scene vs G0.5 saved images, mean abs px diff per camera (0 => scene RNG unperturbed):", diffs)
            rows.append({"scene_rng_check_poker_px_diff": diffs})
        env.close()
    (OUT / "e0_text_contract_log.json").write_text(json.dumps(rows, indent=1))
    body = [r for r in rows if "task" in r]
    ok = all(r["decoded_minus_newline_in_training_strings"] and r["source"] == "native_get_instruction" and r["prefix_count"] == 1 for r in body)
    print("TEXT_CONTRACT_OK (all 18: exact training string, native source, single prefix)" if ok else "TEXT_CONTRACT_VIOLATION",
          "| n_in_training_strings =", sum(r["decoded_minus_newline_in_training_strings"] for r in body), "/", len(body))


if __name__ == "__main__":
    main()
