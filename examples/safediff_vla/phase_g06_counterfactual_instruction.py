#!/usr/bin/env python
"""PHASE G0.6: counterfactual instruction-sensitivity audit (no training, no policy rollout).

Re-creates the EXACT step-0 observation of each G0.5 episode (same seed -> same scene; verified
pixel-for-pixel against the G0.5-saved step-0 PNGs and by reproducing G0.5's step-0 predicted
target) by calling only `env.reset`, never `env.step`. The observation (RGB + robot state) is then
held FIXED while ONLY the instruction string is varied, through the policy's own unmodified
preprocessor pipeline (NewLine + Tokenizer steps). Conditions per scene:

    task_name   : what the G0.5 rollouts actually fed the policy -- `env.call("task_description")`,
                  which for VLABench is the TASK NAME (e.g. "select_book"), NOT the natural-language
                  instruction (`task.instructions[0]`, never passed to the policy). Control /
                  reproduction of G0.5.
    cand::<name>: the task's own instruction template with the referent replaced by each REAL
                  same-type candidate in the scene. Templates are asserted to reproduce
                  `task.instructions[0]` exactly for the true target.
    repeat      : cand::<first> a second time (negative control A: must be identical).
    noun_only   : referent name dropped, category noun kept (negative control B; natural grammar).

Everything computed is read-only on frozen submodules, same pattern as G0/G0.5.
Usage:
    MUJOCO_GL=egl uv run python examples/safediff_vla/phase_g06_counterfactual_instruction.py
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _multitask_common import RENAME_MAP, STAGE2_TASK_NAMES  # noqa: E402
from eval_grounded_grasp_v2_monitor_baseline import CHECKPOINT, robot_base  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.configs import VLABenchEnv  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy  # noqa: E402
from lerobot.policies.safediff_vla.utils import IMAGE_MODALITY, TEXT_MODALITY  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
G05 = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_g05_clean_grounding")
OUT = Path("outputs/eval/safediff_vla_grounded_grasp_v2_phase_g06_counterfactual_instruction")
CAMERA_KEYS = ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]
NOUN_ONLY = {  # natural-grammar referent-free variants of each task's own template
    "select_poker": "Please pick the poker",
    "select_book": "Please take the book",
    "select_mahjong": "Pick up the mahjong",
    "select_fruit": "Put the fruit into the {container}",
    "select_toy": "Put the toy into the {container}",
    "select_drink": "Take out the drink from the {container}",
}


def make_template(instr: str, target: str) -> tuple[str, str]:
    """Return (template with '{X}', form) where form in {'raw','space'} is how the name appears."""
    for form, txt in (("raw", target), ("space", target.replace("_", " "))):
        if txt in instr:
            return instr.replace(txt, "{X}"), form
    raise ValueError(f"target {target!r} not found in instruction {instr!r}")


def fill(template: str, form: str, name: str) -> str:
    return template.replace("{X}", name if form == "raw" else name.replace("_", " "))


@torch.no_grad()
def forward_stages(policy: SafeDiffVLAPolicy, batch: dict) -> dict:
    tokens, pad, mod = policy._encode_multimodal_latent(batch)
    tqe, vg = policy.target_query_extractor, policy.visual_grounding_cross_attn
    valid = pad.to(torch.bool)
    text_valid = (mod == TEXT_MODALITY) & valid
    text_valid |= (text_valid.sum(1, keepdim=True) == 0) & valid
    q = tqe.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)
    pooled, t_attn = tqe.attn(q, tokens, tokens, key_padding_mask=~text_valid, need_weights=True, average_attn_weights=False)
    lang_q = pooled.squeeze(1)
    img_valid = (mod == IMAGE_MODALITY) & valid
    img_valid |= (img_valid.sum(1, keepdim=True) == 0) & valid
    grounded, i_attn = vg.attn(lang_q.unsqueeze(1), tokens, tokens, key_padding_mask=~img_valid, need_weights=True, average_attn_weights=False)
    feat = grounded.squeeze(1)
    xyz = policy.target_point_head(feat)
    xyz_m = (xyz * policy.action_pos_std + policy.action_pos_mean)[0].cpu().numpy()
    img_idx = torch.where(mod[0] == IMAGE_MODALITY)[0]
    txt_idx = torch.where(text_valid[0])[0]
    return {
        "pred_robot": xyz_m,
        "lang_q": lang_q[0].cpu().numpy(),
        "feat": feat[0].cpu().numpy(),
        "text_latent_mean": tokens[0, txt_idx].mean(0).cpu().numpy(),  # post-transformer text tokens, pooled
        "image_latent_mean": tokens[0, img_idx].mean(0).cpu().numpy(),
        "image_latent": tokens[0, img_idx].cpu().numpy(),  # [192, D]
        "img_attn": i_attn[0, :, 0, :][:, img_idx.cpu()].mean(0).cpu().numpy(),
        "text_attn": t_attn[0, :, 0, :][:, txt_idx.cpu()].mean(0).cpu().numpy(),
        "n_text_tokens": int(txt_idx.numel()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=STAGE2_TASK_NAMES)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1000, 1008)))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT).to(args.device).eval()
    results = []
    for task_name in args.tasks:
        cfg = VLABenchEnv(task=task_name)
        env = make_env(cfg, n_envs=1, use_async_envs=False)[task_name][0]
        pre, _ = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=CHECKPOINT,
            preprocessor_overrides={"device_processor": {"device": args.device}, "rename_observations_processor": {"rename_map": RENAME_MAP}},
        )
        env_pre, _ = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config)
        for seed in args.seeds:
            set_seed(seed)
            raw, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
            impl = env.envs[0]
            while not hasattr(impl, "_env"):
                impl = impl.env
            e = impl._env
            task = e.task
            # Template: from THIS reset's own instruction/target (a per-task constant string pattern).
            fresh_target = task.target_entity
            fresh_instr = task.instructions[0] if isinstance(task.instructions, list) else str(task.instructions)
            tpl, form = make_template(fresh_instr, fresh_target)
            assert fill(tpl, form, fresh_target) == fresh_instr
            # Scene = the G0.5-SAVED step-0 observation (RGB PNGs) + G0.5's GT metadata. A fresh
            # `env.reset(seed)` does not regenerate the identical scene for every task (see px_diff).
            g05_meta = json.loads((G05 / task_name / f"seed{seed}" / "meta.json").read_text())
            m0 = g05_meta["steps"][0]
            tname, cands = m0["target_name"], list(m0["cand_names"])
            gt = {n: np.asarray(m0["cand_pos_world"][n], dtype=float) for n in cands}
            base = np.asarray(m0["robot_base"], dtype=float)
            s0_instr = fill(tpl, form, tname)
            container = None
            if task_name in ("select_fruit", "select_toy"):
                container = fresh_instr.split(" into the ")[-1]
            elif task_name == "select_drink":
                container = fresh_instr.split(" from the ")[-1]

            from PIL import Image  # noqa: PLC0415

            obs_np = preprocess_observation(copy.deepcopy(raw))
            px_diff = []
            saved_imgs = {}
            for ci, k in enumerate(RENAME_MAP):
                arr = np.asarray(Image.open(G05 / task_name / f"seed{seed}" / f"step00_camera{ci + 1}.png").convert("RGB"))
                saved_imgs[k] = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                cur = (obs_np[k][0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                px_diff.append(float(np.abs(arr.astype(int) - cur.astype(int)).mean()))
            raw_pp = preprocess_observation(copy.deepcopy(raw))
            for k, v in saved_imgs.items():
                raw_pp[k] = v.to(raw_pp[k].dtype)

            conds = {"task_name": task_name}
            for c in cands:
                conds[f"cand::{c}"] = fill(tpl, form, c)
            conds["repeat"] = fill(tpl, form, cands[0])
            conds["noun_only"] = NOUN_ONLY[task_name].format(container=container)
            out_c = {}
            for key, text in conds.items():
                o = copy.deepcopy(raw_pp)
                o["task"] = [text]
                o = env_pre(o)
                b = pre(o)
                res = forward_stages(policy, b)
                res["text"] = text
                res["n_lang_tokens_in_batch"] = int(b["observation.language.tokens"].shape[1])
                out_c[key] = res
            results.append({"task": task_name, "seed": seed, "target": tname, "s0_instruction": s0_instr, "cands": cands,
                            "gt_world": {n: gt[n].tolist() for n in cands}, "base": base.tolist(), "px_diff_vs_g05": px_diff, "conds": out_c})
            print(f"[{task_name} s{seed}] K={len(cands)} px_diff={np.round(px_diff, 3).tolist()} tokens(task_name)={out_c['task_name']['n_lang_tokens_in_batch']} "
                  f"tokens(instr)={out_c[f'cand::{tname}']['n_lang_tokens_in_batch']} instr={s0_instr!r}", flush=True)
        env.close()

    # serialise (arrays -> lists; keep image_latent as npz for stage analysis)
    lat = {}
    ser = []
    for r in results:
        r2 = {k: v for k, v in r.items() if k != "conds"}
        r2["conds"] = {}
        for ck, cv in r["conds"].items():
            lat[f"{r['task']}|{r['seed']}|{ck}|image_latent"] = cv["image_latent"].astype(np.float16)
            r2["conds"][ck] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cv.items() if k != "image_latent"}
        ser.append(r2)
    (OUT / "g06_raw.json").write_text(json.dumps(ser))
    np.savez_compressed(OUT / "g06_image_latents.npz", **lat)
    print("saved", OUT)


if __name__ == "__main__":
    main()
