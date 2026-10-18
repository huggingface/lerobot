# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the unchanged reference core and load explicit initialization weights."""

from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from lerobot.utils.import_utils import require_package

from .configuration_longwam import LongWAMConfig


def build_core_model(
    config: LongWAMConfig,
    *,
    load_initial_weights: bool,
    tokenizer_assets: str | Path | None = None,
    load_backbone_assets: bool = False,
) -> torch.nn.Module:
    require_package("transformers", extra="longwam")
    if load_initial_weights and config.action_init == "mapped":
        if not config.action_init_path:
            raise ValueError(
                "Fresh mapped-action training requires action_init_path; use --policy.path for a complete policy"
            )
        if not Path(config.action_init_path).expanduser().is_file():
            raise FileNotFoundError("action_init_path must identify an existing mapped ActionDiT file")
    from ._core.long_wam import LongWAM

    video: dict[str, Any] = {
        "has_image_input": False,
        "patch_size": [1, 2, 2],
        "in_dim": 48,
        "hidden_dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "eps": 1e-6,
        "seperated_timestep": True,
        "require_clip_embedding": False,
        "require_vae_embedding": False,
        "fuse_vae_embedding_in_latents": True,
        "use_gradient_checkpointing": config.use_gradient_checkpointing,
        "video_attention_mask_mode": "per_frame_causal",
        "action_conditioned": False,
        "action_dim": config.action_dim,
        "action_group_causal_mask_mode": "group_diagonal",
    }
    action: dict[str, Any] = {
        "action_dim": config.action_dim,
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 24,
        "attn_head_dim": 128,
        "num_layers": 30,
        "text_dim": 4096,
        "freq_dim": 256,
        "eps": 1e-6,
        "use_gradient_checkpointing": config.use_gradient_checkpointing,
    }
    action_path = (
        str(Path(config.action_init_path).expanduser().resolve()) if config.action_init_path else None
    )
    if not load_initial_weights and not load_backbone_assets:
        # Native safetensors contain the complete model, including frozen modules.
        # Do not download or reload the multi-GB Wan/Robot initialization bundles.
        from ._core.action_dit import ActionDiT
        from ._core.mot import MoT
        from ._core.wan_video_dit import WanVideoDiT
        from ._core.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
        from ._core.wan_video_vae import WanVideoVAE38

        tokenizer = None
        if config.load_text_encoder:
            if tokenizer_assets is None or not Path(tokenizer_assets).is_dir():
                raise FileNotFoundError("A complete native Long-WAM checkpoint must include tokenizer/")
            tokenizer = HuggingfaceTokenizer(
                str(tokenizer_assets),
                seq_len=config.tokenizer_max_len,
                clean="whitespace",
                local_files_only=True,
            )
        video_expert = WanVideoDiT(**video).to(dtype=config.dtype)
        action_expert = ActionDiT(**action).to(dtype=config.dtype)
        return LongWAM(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=MoT(
                {"video": video_expert, "action": action_expert},
                mot_checkpoint_mixed_attn=config.use_gradient_checkpointing,
            ),
            vae=WanVideoVAE38().to(dtype=config.dtype),
            text_encoder=WanTextEncoder().to(dtype=config.dtype) if config.load_text_encoder else None,
            tokenizer=tokenizer,
            text_dim=4096,
            proprio_dim=config.proprio_dim,
            device=config.device,
            torch_dtype=config.dtype,
            longwam_past_obs_size=config.past_obs_steps,
            longwam_num_imagine_frames=config.future_video_steps // 16,
            longwam_imagine_sigma=config.imagine_sigma,
            longwam_imagine_infer_steps=config.imagine_steps,
            longwam_joint_denoise=config.denoising == "cod",
            loss_lambda_video=config.loss_lambda_video,
            loss_lambda_action=config.loss_lambda_action,
        )
    core = LongWAM.from_wan22_pretrained(
        device=config.device,
        torch_dtype=config.dtype,
        model_id=config.model_id,
        tokenizer_model_id=config.tokenizer_model_id,
        tokenizer_max_len=config.tokenizer_max_len,
        load_text_encoder=config.load_text_encoder,
        proprio_dim=config.proprio_dim,
        redirect_common_files=False,
        video_dit_config=video,
        action_dit_config=action,
        action_dit_pretrained_path=action_path,
        skip_video_dit_load_from_pretrain=True,
        skip_action_dit_load_from_pretrain=not load_initial_weights or config.action_init == "random",
        mot_checkpoint_mixed_attn=config.use_gradient_checkpointing,
        longwam_past_obs_size=config.past_obs_steps,
        longwam_num_imagine_frames=config.future_video_steps // 16,
        longwam_imagine_sigma=config.imagine_sigma,
        longwam_imagine_infer_steps=config.imagine_steps,
        longwam_joint_denoise=config.denoising == "cod",
        loss_lambda_video=config.loss_lambda_video,
        loss_lambda_action=config.loss_lambda_action,
    )
    if load_initial_weights:
        path = hf_hub_download(
            config.video_init_repo, config.video_init_filename, revision=config.video_init_revision
        )
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("mot"), dict):
            raise ValueError("Robot video initialization requires a mot state dictionary")
        prefix = "mixtures.video."
        state = {key[len(prefix) :]: value for key, value in payload["mot"].items() if key.startswith(prefix)}
        core.video_expert.load_state_dict(state, strict=True)
    return core
