#!/usr/bin/env python3
# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Merge a PEFT (LoRA) adapter checkpoint back into the base LingBot-VLA 2.0 weights.

The lerobot PEFT training path saves adapter-only checkpoints (adapter_config.json +
adapter_model.safetensors next to the policy config). Loading them for inference goes
through `PeftModel`, which adds small per-layer adapter GEMMs on top of the base forward.
Merging folds the adapters into the base weights so the exported checkpoint is
byte-format-identical to a normal fine-tuned checkpoint — the CUDA-graph denoise path
and all inference acceleration apply unchanged.

The merge is computed in float32 (base weights upcast, `W + B @ A * scale`) and the
result is cast back to the checkpoint dtype, so the numerical delta vs the unmerged
adapter model is bf16 rounding only (~1e-3 max abs on actions, well under the bf16
reassociation floor).

Usage:
    python -m lerobot.policies.lingbot_vla_v2.scripts.export_merged \
        --adapter outputs/train/.../checkpoints/004000/pretrained_model \
        --output ./lingbot-vla-v2-robotwin-lora-merged
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path,
                        help="Directory with adapter_config.json + adapter_model.safetensors (the lerobot "
                             "pretrained_model dir of a PEFT checkpoint).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for the merged standard checkpoint.")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Local Qwen3-VL tokenizer/processor dir. Overrides the checkpoint's "
                             "embedded tokenizer_path when that is a Hub id and the machine is offline.")
    args = parser.parse_args()

    import torch
    from peft import PeftConfig, PeftModel

    from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import (
        LingbotVLAV2Config as LeRobotLingbotVLAV2Config,
    )
    from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy

    adapter_dir = args.adapter
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"{adapter_dir} has no adapter_config.json — not a PEFT checkpoint?")

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_path = peft_config.base_model_name_or_path
    if not base_path or not Path(base_path).is_dir():
        raise FileNotFoundError(
            f"adapter_config.json points at base model '{base_path}', which is not a local directory. "
            "Re-point base_model_name_or_path to the converted base checkpoint."
        )

    # Load the base policy in float32 so the merge arithmetic is exact; the original
    # dtype is restored before saving. Prefer the adapter checkpoint's own config:
    # it carries the training-time overrides (robot_config slots, image rename map)
    # that the base model's config predates — exporting with the base config would
    # silently resurrect dropped feature slots and break downstream preprocessing.
    config_src = adapter_dir if (adapter_dir / "config.json").is_file() else base_path
    config = LeRobotLingbotVLAV2Config.from_pretrained(config_src)
    orig_dtype = getattr(config, "dtype", "bfloat16")
    config.dtype = "float32"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.pretrained_path = None
    if args.tokenizer_path:
        config.tokenizer_path = args.tokenizer_path
    policy = LingbotVLAV2Policy.from_pretrained(base_path, config=config)

    peft_model = PeftModel.from_pretrained(policy, adapter_dir)
    merged = peft_model.merge_and_unload()

    merged.to(getattr(torch, orig_dtype))
    merged.config.dtype = orig_dtype
    merged.config.use_peft = False
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output)

    # Carry over the processor pipelines so the merged checkpoint stays self-contained.
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        src = adapter_dir / name
        if src.is_file():
            shutil.copy(src, args.output / name)

    print(f"Merged adapter {adapter_dir} into base {base_path} -> {args.output} (dtype={orig_dtype})")


if __name__ == "__main__":
    main()
