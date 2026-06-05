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

"""Convert a released LingBot-VA HuggingFace checkpoint to LeRobot format.

The released checkpoints are diffusers-style directories with ``transformer/``, ``vae/``,
``text_encoder/`` and ``tokenizer/`` sub-folders. This script:

  1. loads the (sharded) ``transformer/`` weights with the vendored ``WanTransformer3DModel``;
  2. builds a :class:`LingBotVAConfig` for the target benchmark variant;
  3. instantiates a :class:`LingBotVAPolicy` and copies the transformer weights into it
     (near-identity: the only key change is the ``transformer.`` prefix);
  4. saves the LeRobot policy (``model.safetensors`` + ``config.json``) and its processors.

Packaging decision: only the trainable ~5B transformer is bundled into the LeRobot
``model.safetensors``. The frozen VAE + UMT5 text encoder + tokenizer (~20 GB) are NOT
copied; instead ``config.wan_pretrained_path`` records where to lazily pull them from at
load time (defaults to the source repo/dir). Pass ``--bundle-frozen`` to additionally copy
those sub-folders next to the converted checkpoint and point ``wan_pretrained_path`` at it.

Example (LIBERO-Long, the LIBERO eval gate):

    python -m lerobot.policies.lingbot_va.convert_lingbot_va_checkpoints \
        --checkpoint robbyant/lingbot-va-posttrain-libero-long \
        --variant libero \
        --output_dir outputs/lingbot_va_libero_long

Requires a CUDA GPU with enough RAM/VRAM to materialize the transformer; run on Linux.
"""

import argparse
import shutil
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
from lerobot.policies.lingbot_va.processor_lingbot_va import make_lingbot_va_pre_post_processors
from lerobot.policies.lingbot_va.wan_transformer import WanTransformer3DModel
from lerobot.utils.constants import ACTION, OBS_IMAGES

# Per-benchmark variant presets (camera keys + action layout). Values mirror the upstream
# configs (wan_va/configs/va_*_cfg.py).
VARIANTS = {
    "libero": {
        "obs_cam_keys": [f"{OBS_IMAGES}.image", f"{OBS_IMAGES}.image2"],
        "height": 128,
        "width": 128,
        "action_per_frame": 4,
        "frame_chunk_size": 4,
        "attn_window": 30,
        "num_inference_steps": 20,
        "action_num_inference_steps": 50,
        "guidance_scale": 5.0,
        "action_guidance_scale": 1.0,
        "snr_shift": 5.0,
        "action_snr_shift": 0.05,
        "used_action_channel_ids": list(range(7)),
        # 7-DoF: agentview + eye-in-hand, single arm. Quantiles are the config defaults.
        "image_shape": (3, 256, 256),
    },
    "robotwin": {
        "obs_cam_keys": [
            f"{OBS_IMAGES}.cam_high",
            f"{OBS_IMAGES}.cam_left_wrist",
            f"{OBS_IMAGES}.cam_right_wrist",
        ],
        "height": 256,
        "width": 320,
        "action_per_frame": 16,
        "frame_chunk_size": 2,
        "attn_window": 72,
        "num_inference_steps": 25,
        "action_num_inference_steps": 50,
        "guidance_scale": 5.0,
        "action_guidance_scale": 1.0,
        "snr_shift": 5.0,
        "action_snr_shift": 1.0,
        # RoboTwin is dual-arm; set the used channels / quantiles to match the deployed config.
        "used_action_channel_ids": list(range(14)),
        "image_shape": (3, 256, 256),
    },
}


def _transformer_dir(checkpoint: str) -> str:
    """Return the path/repo that ``WanTransformer3DModel.from_pretrained`` should read."""
    p = Path(checkpoint)
    if p.is_dir():
        return str(p / "transformer")
    return checkpoint  # HF repo id; use subfolder kwarg below


def load_source_transformer(checkpoint: str, dtype: torch.dtype) -> WanTransformer3DModel:
    p = Path(checkpoint)
    if p.is_dir():
        return WanTransformer3DModel.from_pretrained(
            str(p / "transformer"), torch_dtype=dtype, attn_mode="torch"
        )
    return WanTransformer3DModel.from_pretrained(
        checkpoint, subfolder="transformer", torch_dtype=dtype, attn_mode="torch"
    )


def build_config(variant: str, wan_pretrained_path: str, dtype: str) -> LingBotVAConfig:
    preset = VARIANTS[variant]
    n_used = len(preset["used_action_channel_ids"])
    kwargs = {
        "wan_pretrained_path": wan_pretrained_path,
        "dtype": dtype,
        "obs_cam_keys": preset["obs_cam_keys"],
        "height": preset["height"],
        "width": preset["width"],
        "action_per_frame": preset["action_per_frame"],
        "frame_chunk_size": preset["frame_chunk_size"],
        "attn_window": preset["attn_window"],
        "num_inference_steps": preset["num_inference_steps"],
        "action_num_inference_steps": preset["action_num_inference_steps"],
        "guidance_scale": preset["guidance_scale"],
        "action_guidance_scale": preset["action_guidance_scale"],
        "snr_shift": preset["snr_shift"],
        "action_snr_shift": preset["action_snr_shift"],
        "used_action_channel_ids": preset["used_action_channel_ids"],
        "device": "cpu",
    }
    if variant != "libero":
        # LIBERO keeps the config default quantiles; other variants need their own. Until the
        # exact per-channel quantiles are wired in, use a neutral [-1, 1] mapping (no rescale).
        kwargs["action_q01"] = [-1.0] * n_used
        kwargs["action_q99"] = [1.0] * n_used
    cfg = LingBotVAConfig(**kwargs)
    # Populate input/output features (cameras + action) so validate_features passes.
    img_shape = preset["image_shape"]
    cfg.input_features = {
        k: PolicyFeature(type=FeatureType.VISUAL, shape=img_shape) for k in preset["obs_cam_keys"]
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(n_used,))}
    cfg.validate_features()
    return cfg


def convert(
    checkpoint: str, variant: str, output_dir: str, dtype: str, bundle_frozen: bool, push_to: str | None
):
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Decide where frozen modules will be pulled from at load time.
    if bundle_frozen:
        wan_pretrained_path = str(out)
        _copy_frozen_subfolders(checkpoint, out)
    else:
        wan_pretrained_path = checkpoint

    print(f"Building LingBot-VA config for variant '{variant}' (frozen modules from: {wan_pretrained_path})")
    cfg = build_config(variant, wan_pretrained_path, dtype)

    print("Loading source transformer weights ...")
    src = load_source_transformer(checkpoint, torch_dtype)
    src_sd = src.state_dict()

    print("Instantiating LingBotVAPolicy and copying transformer weights ...")
    # Build the policy without triggering frozen-module download by constructing directly.
    policy = LingBotVAPolicy(cfg)
    # Near-identity remap: source transformer keys -> policy "transformer.*".
    remapped = {f"transformer.{k}": v for k, v in src_sd.items()}
    missing, unexpected = policy.load_state_dict(remapped, strict=False)
    _log_load_keys(missing, unexpected)
    policy = policy.to(torch_dtype)

    print(f"Saving converted policy to {out}")
    policy.save_pretrained(out)

    preprocessor, postprocessor = make_lingbot_va_pre_post_processors(cfg, dataset_stats=None)
    preprocessor.save_pretrained(out)
    postprocessor.save_pretrained(out)

    if push_to:
        print(f"Pushing to the Hub: {push_to}")
        policy.push_to_hub(push_to)
        preprocessor.push_to_hub(push_to)
        postprocessor.push_to_hub(push_to)

    print("Done.")


def _copy_frozen_subfolders(checkpoint: str, out: Path):
    p = Path(checkpoint)
    if not p.is_dir():
        from huggingface_hub import snapshot_download

        p = Path(snapshot_download(checkpoint, allow_patterns=["vae/*", "text_encoder/*", "tokenizer/*"]))
    for sub in ("vae", "text_encoder", "tokenizer"):
        src_sub = p / sub
        if src_sub.is_dir():
            shutil.copytree(src_sub, out / sub, dirs_exist_ok=True)
            print(f"  bundled {sub}/")


def _log_load_keys(missing, unexpected):
    # The source transformer should account for every "transformer.*" key in the policy.
    if missing:
        print(
            f"  [load_state_dict] {len(missing)} missing keys (expected: none for transformer). Sample: {missing[:5]}"
        )
    if unexpected:
        print(f"  [load_state_dict] {len(unexpected)} unexpected keys. Sample: {unexpected[:5]}")
    if not missing and not unexpected:
        print("  [load_state_dict] perfect match (near-identity remap).")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="HF repo id or local diffusers-style directory.")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS.keys()))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--bundle-frozen",
        action="store_true",
        help="Copy the frozen vae/text_encoder/tokenizer next to the checkpoint instead of lazy-pulling.",
    )
    parser.add_argument(
        "--push_to_hub", default=None, help="Optional HF repo id to push the converted policy to."
    )
    args = parser.parse_args()
    convert(
        checkpoint=args.checkpoint,
        variant=args.variant,
        output_dir=args.output_dir,
        dtype=args.dtype,
        bundle_frozen=args.bundle_frozen,
        push_to=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
