#!/usr/bin/env python3
"""Extract the (FAST-pretrained) SmolVLM2 VLM from an A1 stage-1 smolvla_ki
checkpoint and re-save it in plain HF format.

A1 stage-2 then loads it via `--policy.vlm_model_name=<outdir>
--policy.load_vlm_weights=true`, freezes it (`train_expert_only=true`), and
trains a fresh flow expert on top — i.e. "pretrain VLM on action tokens, then
hard-freeze and train the action head", the arm-A1 recipe.

Usage:  python scripts/extract_vlm.py <stage1_pretrained_model_dir> <out_dir>
"""

import sys

from lerobot.policies.smolvla_ki.modeling_smolvla_ki import SmolVLAKIPolicy


def main():
    if len(sys.argv) != 3:
        print("usage: extract_vlm.py <stage1_ckpt_dir> <out_dir>")
        sys.exit(2)
    ckpt, outdir = sys.argv[1], sys.argv[2]
    print(f"[extract_vlm] loading stage-1 policy from {ckpt} on CPU ...", flush=True)
    policy = SmolVLAKIPolicy.from_pretrained(ckpt)
    policy = policy.to("cpu")
    vlm = policy.model.vlm_with_expert.vlm
    n = sum(p.numel() for p in vlm.parameters()) / 1e6
    print(f"[extract_vlm] saving VLM ({n:.0f}M params) -> {outdir}", flush=True)
    vlm.save_pretrained(outdir)
    # processor (tokenizer + image processor) so stage-2 can load the dir cleanly
    policy.model.vlm_with_expert.processor.save_pretrained(outdir)
    print(f"[extract_vlm] done -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
