# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit conversion of released two-camera LIBERO bundles to native LeRobot."""

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from lerobot.policies import make_pre_post_processors

from .configuration_longwam import LongWAMConfig
from .modeling_longwam import LongWAMPolicy


def config_from_release(bundle: Path, *, device: str = "cpu") -> LongWAMConfig:
    """Validate the supported release contract without executing YAML targets."""
    with (bundle / "config.yaml").open() as handle:
        source = yaml.safe_load(handle)
    data, model = source["data"]["train"], source["model"]
    processor = data["processor"]
    if data["concat_multi_camera"] != "horizontal" or data["video_size"] != [224, 448]:
        raise ValueError("This converter currently supports the released two-camera LIBERO layout only")
    if [x["key"] for x in data["shape_meta"]["images"]] != ["image", "wrist_image"]:
        raise ValueError("Unexpected camera order")
    if processor["norm_default_mode"] != "min/max" or processor.get("norm_exception_mode"):
        raise ValueError("Unsupported normalization recipe")
    if processor.get("use_stepwise_action_norm") or processor.get("action_state_transforms"):
        raise ValueError("Unsupported stepwise normalization or action/state transform")
    architectures: dict[str, dict[str, Any]] = {
        "video_dit_config": {
            "hidden_dim": 3072,
            "ffn_dim": 14336,
            "in_dim": 48,
            "out_dim": 48,
            "num_layers": 30,
            "num_heads": 24,
            "attn_head_dim": 128,
            "text_dim": 4096,
            "freq_dim": 256,
            "patch_size": [1, 2, 2],
            "video_attention_mask_mode": "per_frame_causal",
            "action_conditioned": False,
        },
        "action_dit_config": {
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_layers": 30,
            "num_heads": 24,
            "attn_head_dim": 128,
            "text_dim": 4096,
            "freq_dim": 256,
        },
    }
    for expert, expected in architectures.items():
        for key, value in expected.items():
            if model[expert].get(key) != value:
                raise ValueError(f"Unsupported architecture: {expert}.{key}")
    for scheduler in ("video_scheduler", "action_scheduler"):
        if model[scheduler] != {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}:
            raise ValueError(f"Unsupported {scheduler}")

    def setting(name, default=None):
        return model.get("longwam_" + name, model.get("arwam_" + name, default))

    past = int(data["past_obs_size"])
    future = int(data["num_frames"]) - past - 1
    if setting("past_obs_size") != past or setting("num_imagine_frames") * 16 != future:
        raise ValueError("Inconsistent video history/future metadata")
    if model["proprio_dim"] != 8 or model["action_dit_config"]["action_dim"] != 7:
        raise ValueError("Expected LIBERO state/action dimensions 8/7")
    return LongWAMConfig(
        device=device,
        dtype=torch.bfloat16,
        past_obs_steps=past,
        future_video_steps=future,
        video_stride=int(data["action_video_freq_ratio"]),
        action_horizon=int(data["action_chunk"]),
        inference_action_horizon=80,
        n_action_steps=10,
        execution="sync",
        num_inference_steps=int(source.get("eval_num_inference_steps", 20)),
        denoising="cod" if setting("joint_denoise", False) else "idm",
        imagine_sigma=float(setting("imagine_sigma")),
        imagine_steps=int(setting("imagine_infer_steps")),
        tokenizer_max_len=int(data["context_len"]),
        load_text_encoder=True,
        camera_keys=("observation.images.image", "observation.images.image2"),
        toggle_action_dimensions=[-1],
        use_gradient_checkpointing=False,
        push_to_hub=False,
    )


def release_statistics(bundle: Path) -> dict[str, dict[str, torch.Tensor]]:
    with (bundle / "dataset_stats.json").open() as handle:
        source = json.load(handle)
    result = {}
    for source_key, target_key, dimension in (("state", "observation.state", 8), ("action", "action", 7)):
        result[target_key] = {
            name: torch.tensor(source[source_key]["default"]["global_" + name], dtype=torch.float32)
            for name in ("min", "max")
        }
        low, high = result[target_key]["min"], result[target_key]["max"]
        if low.shape != (dimension,) or high.shape != low.shape:
            raise ValueError(f"Invalid normalization shape for {target_key}")
        if not torch.isfinite(low).all() or not torch.isfinite(high).all() or (high < low).any():
            raise ValueError(f"Invalid normalization bounds for {target_key}")
    return result


def convert_release(bundle: Path, output: Path, *, backbone: Path, tokenizer_model: Path) -> None:
    """Strictly load experts and frozen assets, then save one native checkpoint."""
    if output.exists():
        raise FileExistsError("Choose a new output directory; never overwrite a checkpoint")
    for path in (
        bundle / "model.pt",
        backbone / "Wan2.2_VAE.pth",
        backbone / "models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_model / "google/umt5-xxl",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    cfg = config_from_release(bundle)
    stats = release_statistics(bundle)
    cfg.model_id, cfg.tokenizer_model_id = str(backbone), str(tokenizer_model)
    policy = LongWAMPolicy(cfg, load_initial_weights=False, load_backbone_assets=True)
    policy.model.load_checkpoint(str(bundle / "model.pt"), strict=True)
    # The complete native artifact no longer needs local initialization paths.
    cfg.model_id, cfg.tokenizer_model_id = "Wan-AI/Wan2.2-TI2V-5B", "Wan-AI/Wan2.1-T2V-1.3B"
    policy.save_pretrained(output)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    pre.save_pretrained(output, config_filename="policy_preprocessor.json")
    post.save_pretrained(output, config_filename="policy_postprocessor.json")
    policy.close()
