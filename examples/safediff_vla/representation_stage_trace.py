#!/usr/bin/env python
"""Read-only representation-stage trace: with image/state FROZEN (one env.reset(), zero
env.step() -- no rollout), vary ONLY the instruction text and compare, at every representation
stage between the tokenizer and the decoder's cross-attention memory:
  1. tokenizer output (text token ids + attention mask)
  2. VLM text-token embeddings (pre-fusion, i.e. before the VLM's self-attention pass)
  3. VLM image-token embeddings (pre-fusion)
  4. full multimodal token hidden states [B,N,D] (post VLM self-attention, the whole prefix)
  5. _encode_multimodal_latent's actual return value
  6. the pooled latent that feeds TargetPointHead/state-predictor heads (_pooled_latent's output)
  7. the tensor fed to TemporalActionDecoder's cross-attention memory (decoder.latent_projection
     applied to stage 4/5)

No model/execution/architecture code is modified -- calls existing (including private/internal)
methods directly for introspection, same pattern as this session's other diagnostics.

Usage:
    uv run python examples/safediff_vla/representation_stage_trace.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.envs import make_env, make_env_pre_post_processors
from lerobot.envs.configs import VLABenchEnv
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
from lerobot.envs.vlabench import VLABenchEnv as VLABenchEnvImpl
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.safediff_vla.modeling_safediff_vla import SafeDiffVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.random_utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

CHECKPOINT = "outputs/train/safediff_vla_temporal_decoder_padfix_20k/checkpoints/020000/pretrained_model"
TASK = "select_poker"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}
SEED = 1000


def pretty_identity(identity: str) -> str:
    rank, _, suit = identity.partition("_of_")
    return f"{rank} of {suit}"


def capture_scene(seed: int) -> dict[str, Any]:
    env_cfg = VLABenchEnv(task=TASK)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[TASK][0]
    env_impl: VLABenchEnvImpl = env.envs[0]
    try:
        observation, _ = env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
        cards = [n for n, e in env_impl._env.task.entities.items() if type(e).__name__ == "Poker"]
        obs_t = preprocess_observation(observation)
        obs_t = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs_t.items()}
    finally:
        env.close()
    return {"obs_t": obs_t, "cards": cards}


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def trace_instruction(
    policy: SafeDiffVLAPolicy, env_preprocessor, preprocessor, obs_t_base: dict, instruction: str
) -> dict[str, Any]:
    obs_t = dict(obs_t_base)
    obs_t["task"] = [instruction]
    obs_t = env_preprocessor(obs_t)
    obs_norm = preprocessor(obs_t)

    backbone = policy.backbone
    model = backbone.model  # VLAFlowMatching

    with torch.no_grad():
        backbone_batch = policy._prepare_backbone_batch(obs_norm)

        # ---- Stage 1: tokenizer output ----
        lang_tokens = backbone_batch[OBS_LANGUAGE_TOKENS]
        lang_masks = backbone_batch[OBS_LANGUAGE_ATTENTION_MASK]

        # ---- Stage 2: VLM text-token embeddings (pre-fusion) ----
        lang_emb = model.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])

        # ---- Stage 3: VLM image-token embeddings (pre-fusion) ----
        images, img_masks = backbone.prepare_images(backbone_batch)
        img_embs = []
        for img in images:
            e = model.vlm_with_expert.embed_image(img)
            e = e * math.sqrt(e.shape[-1])
            img_embs.append(e)
        img_emb_cat = torch.cat(img_embs, dim=1)  # [B, n_images*patches(+special), D] -- special tokens not embedded here (see note)

        # ---- Stage 4 / 5: full multimodal hidden states = _encode_multimodal_latent's return ----
        latent_tokens, latent_pad_mask = policy._encode_multimodal_latent(obs_norm)

        # ---- Stage 6: pooled latent (TargetPointHead / subgoal head input) ----
        pooled = policy._pooled_latent(latent_tokens, latent_pad_mask)

        # ---- Stage 7: decoder cross-attention memory ----
        decoder_memory = policy.decoder.latent_projection(latent_tokens)

    # index bookkeeping for slicing stage-4/5/7's concatenated prefix into image-only vs
    # text-only sub-ranges, replicating embed_prefix's exact construction order.
    add_special = model.add_image_special_tokens
    per_image_tokens = img_embs[0].shape[1]  # patch count per camera (special tokens counted separately)
    special_per_image = 2 if add_special else 0  # start(2 ids) + end(1 id) -> but start/end are each ONE embedded row here (start uses embed_language_tokens on a 2-token id pair -> 1 row per call in embed_prefix? see note)
    n_images = len(images)
    n_real_lang = int(lang_masks[0].sum().item())

    return {
        "instruction": instruction,
        "stage1_token_ids": lang_tokens[0].cpu().tolist(),
        "stage1_attention_mask": lang_masks[0].cpu().tolist(),
        "stage1_n_real_tokens": n_real_lang,
        "stage2_text_emb": lang_emb[0].float().cpu().numpy(),  # [T_text_padded, D]
        "stage2_real_mask": lang_masks[0].cpu().numpy().astype(bool),
        "stage3_image_emb": img_emb_cat[0].float().cpu().numpy(),  # [n_images*patches, D]
        "stage4_5_full_tokens": latent_tokens[0].float().cpu().numpy(),  # [N, D]
        "stage4_5_pad_mask": latent_pad_mask[0].cpu().numpy().astype(bool) if latent_pad_mask is not None else None,
        "stage6_pooled": pooled[0].float().cpu().numpy(),  # [D]
        "stage7_decoder_memory": decoder_memory[0].float().cpu().numpy(),  # [N, D']
        "n_images": n_images,
        "per_image_patch_tokens": per_image_tokens,
        "add_image_special_tokens": bool(add_special),
        "n_real_lang_tokens": n_real_lang,
    }


def compare_multi_token(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: [N, D] (same N). Per-token cosine/L2, then averaged, plus a flattened whole-tensor comparison."""
    per_token_cos = [cos_sim(a[i], b[i]) for i in range(a.shape[0])]
    per_token_l2 = np.linalg.norm(a - b, axis=-1)
    return {
        "shape": list(a.shape),
        "mean_per_token_cosine_sim": float(np.mean(per_token_cos)),
        "min_per_token_cosine_sim": float(np.min(per_token_cos)),
        "mean_per_token_l2": float(np.mean(per_token_l2)),
        "max_per_token_l2": float(np.max(per_token_l2)),
        "flattened_cosine_sim": cos_sim(a.flatten(), b.flatten()),
        "flattened_l2": float(np.linalg.norm(a.flatten() - b.flatten())),
    }


def compare_vector(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "shape": list(a.shape),
        "cosine_sim": cos_sim(a, b),
        "l2": float(np.linalg.norm(a - b)),
    }


def main() -> None:
    global CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/eval/safediff_vla_representation_stage_trace")
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    logger.info("=== loading policy from %s (unmodified) ===", CHECKPOINT)
    policy = SafeDiffVLAPolicy.from_pretrained(CHECKPOINT)
    policy = policy.to(args.device)
    policy.eval()
    assert policy.config.architecture == "temporal_decoder"

    env_cfg = VLABenchEnv(task=TASK)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    logger.info("=== scene seed=%d: single reset, zero steps ===", args.seed)
    scene = capture_scene(args.seed)
    cards = scene["cards"][:3]
    instructions = [f"primitive: Please pick the poker {pretty_identity(c)}" for c in cards]
    logger.info("instructions: %s", instructions)

    traces = {}
    for instr in instructions:
        logger.info("tracing: %s", instr)
        traces[instr] = trace_instruction(policy, env_preprocessor, preprocessor, scene["obs_t"], instr)

    # ---- per-stage pairwise comparisons ----
    pairs = list(itertools.combinations(instructions, 2))
    stage_results = {"stage1_tokenizer": [], "stage2_text_emb": [], "stage3_image_emb": [], "stage4_5_full": [], "stage6_pooled": [], "stage7_decoder_memory": []}

    example = traces[instructions[0]]
    n_images = example["n_images"]
    per_image_tokens = example["per_image_patch_tokens"]
    add_special = example["add_image_special_tokens"]

    for a_instr, b_instr in pairs:
        ta, tb = traces[a_instr], traces[b_instr]

        # Stage 1: token id comparison (Hamming-style, padded-length may differ -> compare on min length)
        ids_a, ids_b = ta["stage1_token_ids"], tb["stage1_token_ids"]
        min_len = min(len(ids_a), len(ids_b))
        n_diff_ids = sum(1 for i in range(min_len) if ids_a[i] != ids_b[i])
        stage_results["stage1_tokenizer"].append(
            {
                "pair": (a_instr, b_instr),
                "n_real_tokens_a": ta["stage1_n_real_tokens"],
                "n_real_tokens_b": tb["stage1_n_real_tokens"],
                "n_token_ids_differing_over_common_length": n_diff_ids,
                "common_length_compared": min_len,
            }
        )

        # Stage 2: text embeddings -- restrict to REAL (non-pad) tokens on the shorter side.
        mask_a, mask_b = ta["stage2_real_mask"], tb["stage2_real_mask"]
        real_a = ta["stage2_text_emb"][mask_a]
        real_b = tb["stage2_text_emb"][mask_b]
        k = min(len(real_a), len(real_b))
        stage_results["stage2_text_emb"].append({"pair": (a_instr, b_instr), **compare_multi_token(real_a[:k], real_b[:k])})

        # Stage 3: image embeddings -- SAME image, so should be near-identical (sanity check).
        stage_results["stage3_image_emb"].append(
            {"pair": (a_instr, b_instr), **compare_multi_token(ta["stage3_image_emb"], tb["stage3_image_emb"])}
        )

        # Stage 4/5: full multimodal tokens -- whole-sequence, plus a per-region breakdown
        # (image-token region vs text-token region vs the trailing state token) since embed_prefix
        # concatenates in that fixed order: [n_images*per_image_tokens image patches][real lang
        # tokens][state token(s)]. Using the pad mask (not hand-derived special-token counts) to
        # find the real (non-padding-to-prefix_length) sequence length.
        full_a, full_b = ta["stage4_5_full_tokens"], tb["stage4_5_full_tokens"]
        pad_mask_a = ta["stage4_5_pad_mask"]
        real_len_a = int(pad_mask_a.sum()) if pad_mask_a is not None else full_a.shape[0]
        n_img_tok = n_images * per_image_tokens
        n_lang_a, n_lang_b = ta["n_real_lang_tokens"], tb["n_real_lang_tokens"]
        n_lang_common = min(n_lang_a, n_lang_b)  # only meaningful if equal; reported either way

        def region_profile(arr_a: np.ndarray, arr_b: np.ndarray) -> dict:
            per_tok_l2 = np.linalg.norm(arr_a[:real_len_a] - arr_b[:real_len_a], axis=-1)
            img = per_tok_l2[:n_img_tok]
            txt = per_tok_l2[n_img_tok : n_img_tok + n_lang_common]
            state = per_tok_l2[n_img_tok + max(n_lang_a, n_lang_b) :]
            return {
                "image_tokens": {"n": len(img), "mean_l2": float(img.mean()), "max_l2": float(img.max())},
                "text_tokens": {"n": len(txt), "mean_l2": float(txt.mean()) if len(txt) else None, "max_l2": float(txt.max()) if len(txt) else None},
                "state_token": {"n": len(state), "mean_l2": float(state.mean()) if len(state) else None},
                "top10_highest_l2_token_indices": np.argsort(-per_tok_l2)[:10].tolist(),
            }

        full_result = compare_multi_token(full_a[:real_len_a], full_b[:real_len_a])
        full_result["region_profile"] = region_profile(full_a, full_b)
        stage_results["stage4_5_full"].append({"pair": (a_instr, b_instr), "whole_sequence": full_result})

        # Stage 6: pooled vector.
        stage_results["stage6_pooled"].append({"pair": (a_instr, b_instr), **compare_vector(ta["stage6_pooled"], tb["stage6_pooled"])})

        # Stage 7: decoder memory (full sequence, same real_len as stage 4/5, same region profile).
        mem_a, mem_b = ta["stage7_decoder_memory"], tb["stage7_decoder_memory"]
        mem_result = compare_multi_token(mem_a[:real_len_a], mem_b[:real_len_a])
        mem_result["region_profile"] = region_profile(mem_a, mem_b)
        stage_results["stage7_decoder_memory"].append({"pair": (a_instr, b_instr), **mem_result})

    result = {
        "checkpoint": CHECKPOINT,
        "seed": args.seed,
        "instructions": instructions,
        "n_images": n_images,
        "per_image_patch_tokens": per_image_tokens,
        "add_image_special_tokens": add_special,
        "n_real_lang_tokens_per_instruction": {i: traces[i]["stage1_n_real_tokens"] for i in instructions},
        "token_budget_note": (
            f"~{n_images * per_image_tokens} image patch tokens vs "
            f"{min(traces[i]['stage1_n_real_tokens'] for i in instructions)}-"
            f"{max(traces[i]['stage1_n_real_tokens'] for i in instructions)} real text tokens per "
            "instruction -- the ratio the masked-mean pooling at _pooled_latent averages over."
        ),
        "stage_results": stage_results,
    }
    (out_dir / "representation_stage_trace.json").write_text(json.dumps(result, indent=2))
    logger.info(json.dumps({k: v for k, v in result.items() if k != "stage_results"}, indent=2))
    for stage, entries in stage_results.items():
        logger.info("=== %s ===", stage)
        for e in entries:
            logger.info(json.dumps(e, indent=2, default=str))
    logger.info("wrote %s", out_dir / "representation_stage_trace.json")


if __name__ == "__main__":
    main()
