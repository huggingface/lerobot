#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Producer (run in the ORIGINAL gr00t env): dump original GR00T N1.7 outputs + inputs.

The original NVIDIA ``gr00t`` package pins ``transformers==4.57.3`` (py3.10) and its
model-config dataclasses are incompatible with the ``transformers==5.x`` that the
LeRobot GR00T N1.7 integration requires. The two implementations therefore cannot be
imported in the same Python process. To keep the parity comparison FAIR, we run the
original model in its native env here and serialize, PER EMBODIMENT TAG:

  * the exact pre-processed/collated model inputs (so the LeRobot side consumes the
    byte-identical tensors -- same image preprocessing, tokenization, normalization),
  * the random seed used right before the flow-matching sampler,
  * the raw ``action_pred`` tensor returned by ``model.get_action`` (normalized space,
    before any per-implementation action decoding).

Inputs are built GENERICALLY from the checkpoint metadata (no per-tag hardcoding):
state keys + dims come from ``statistics.json``; video + language keys come from the
processor's per-embodiment modality configs. This lets us test many embodiment tags
from the SAME checkpoint and confirm the LeRobot integration is not overfit to
``libero_sim``.

The companion pytest (run in the LeRobot env) loads each .npz, replays the identical
inputs + seed through the LeRobot GR00T N1.7 model, and asserts the outputs match.

Usage:
    .venv-original/bin/python tests/policies/groot/utils/dump_original_n1_7.py \
        --ckpt <path-to-GR00T-N1.7-LIBERO/libero_10> \
        --out-dir tests/policies/groot/artifacts \
        [--tags libero_sim,oxe_droid_relative_eef_relative_joint,...] \
        [--device cuda] [--seed 42]

If --tags is omitted, every embodiment present in the checkpoint statistics is dumped.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

IMAGE_SIZE = 256
BATCH_SIZE = 2
PROMPT = "pick up the black bowl and place it on the plate"


def load_statistics(ckpt: str) -> dict:
    with open(os.path.join(ckpt, "statistics.json")) as f:
        return json.load(f)


def make_observation(seed: int, video_keys, lang_key, state_spec):
    """Build a dummy observation dict generically from the embodiment metadata."""
    rng = np.random.default_rng(seed)
    video = {
        k: rng.integers(0, 256, (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        for k in video_keys
    }
    # One ndarray per state key, shape (B, T=1, key_dim); dim taken from statistics.
    # Keys with dim 0 (e.g. disabled eef on some embodiments) are still emitted as
    # present-but-empty so the processor's state transform finds every expected key.
    state = {k: rng.standard_normal((BATCH_SIZE, 1, dim)).astype(np.float32) for k, dim in state_spec}
    language = {lang_key: [[PROMPT] for _ in range(BATCH_SIZE)]}
    return {"video": video, "state": state, "language": language}


def dump_one_tag(policy, fair_model, tag, modality_cfg, state_spec, args, out_path):
    from gr00t.data.types import MessageType

    video_keys = modality_cfg["video"].modality_keys
    lang_key = modality_cfg["language"].modality_keys[0]
    observation = make_observation(args.seed, video_keys, lang_key, state_spec)

    # Point the policy preprocessing at this embodiment (mirrors Gr00tPolicy.__init__).
    policy.embodiment_tag = type(policy.embodiment_tag)(tag)
    policy.modality_configs = {
        k: v for k, v in policy.processor.get_modality_configs()[tag].items() if k != "rl_info"
    }
    policy.language_key = policy.modality_configs["language"].modality_keys[0]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    unbatched = policy._unbatch_observation(observation)
    processed = []
    for obs in unbatched:
        vla = policy._to_vla_step_data(obs)
        processed.append(policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": vla}]))
    collated = policy.collate_fn(processed)

    def to_dev(x):
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            return x.to(args.device, torch.float32)
        if isinstance(x, torch.Tensor):
            return x.to(args.device)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        return x

    collated = {k: to_dev(v) for k, v in collated.items()}

    torch.manual_seed(args.seed)
    with torch.inference_mode():
        out = fair_model.get_action(**collated)
    action_pred = out["action_pred"].float().cpu().numpy()

    flat, meta = {}, {}

    def flatten(prefix, obj):
        if isinstance(obj, torch.Tensor):
            arr = obj.float().cpu().numpy() if torch.is_floating_point(obj) else obj.cpu().numpy()
            flat[f"in::{prefix}"] = arr
            meta[f"in::{prefix}"] = str(obj.dtype)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                flatten(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, (list, tuple)):
            flat[f"in::{prefix}"] = np.array(obj, dtype=object)
        else:
            flat[f"in::{prefix}"] = np.array(obj)

    flatten("", collated)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        action_pred=action_pred,
        seed=np.array(args.seed),
        device=np.array(args.device),
        embodiment_tag=np.array(tag),
        meta_keys=np.array(list(meta.keys()), dtype=object),
        meta_dtypes=np.array(list(meta.values()), dtype=object),
        **flat,
    )
    print(f"[{tag}] action_pred {action_pred.shape} -> {out_path.name} ({os.path.getsize(out_path)} B)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True, help="directory for per-tag .npz files")
    ap.add_argument("--tags", default="", help="comma-separated embodiment tags (default: all in stats)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from transformers import AutoConfig, AutoModel

    stats = load_statistics(args.ckpt)
    requested = [t.strip() for t in args.tags.split(",") if t.strip()] or list(stats.keys())

    # Load the policy once (for its processor/preprocessing) on any valid tag.
    bootstrap_tag = "libero_sim" if "libero_sim" in stats else requested[0]
    policy = Gr00tPolicy(embodiment_tag=bootstrap_tag, model_path=args.ckpt, device=args.device)
    all_modality = policy.processor.get_modality_configs()

    # Load a FAIR model (SDPA + fp32) once and reuse across tags. Otherwise the
    # original checkpoint default (flash_attention_2 + bf16) introduces kernel/rounding
    # noise vs the LeRobot env (which has no flash_attn and runs SDPA).
    cfg = AutoConfig.from_pretrained(args.ckpt, trust_remote_code=True)
    cfg.use_flash_attention = False
    cfg.load_bf16 = False
    fair_model = AutoModel.from_pretrained(args.ckpt, config=cfg, trust_remote_code=True)
    fair_model.to(device=args.device, dtype=torch.float32)
    fair_model.eval()

    out_dir = Path(args.out_dir)
    done, skipped = [], []
    for tag in requested:
        if tag not in stats or tag not in all_modality:
            print(f"[skip] {tag}: not present in checkpoint statistics/modality configs")
            skipped.append(tag)
            continue
        state_spec = [(k, len(v["min"])) for k, v in stats[tag]["state"].items()]
        try:
            dump_one_tag(
                policy,
                fair_model,
                tag,
                all_modality[tag],
                state_spec,
                args,
                out_dir / f"original_n1_7_{tag}.npz",
            )
            done.append(tag)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {tag}: {type(exc).__name__}: {exc}")
            skipped.append(tag)

    print(f"\nDumped {len(done)} tags: {done}")
    if skipped:
        print(f"Skipped/failed {len(skipped)} tags: {skipped}")


if __name__ == "__main__":
    main()
