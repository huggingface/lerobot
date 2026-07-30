# Copyright (C) 2026 Tencent.  All rights reserved.
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

# ruff: noqa: B007, N806, SIM102

"""Hy-VLA flow-matching model."""

import copy
import json
import math
import os
import sys

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from transformers import AutoConfig

from ..configuration_hy_vla import HyVLAConfig
from .hunyuan_vl_mot import (
    HunYuanVLMoTConfig,
)
from .modeling_dual_tower import (
    HyDualTower,
    HyDualTowerConfig,
)

# ---------------------------------------------------------------------------
# VLM AutoConfig loading: returns a bundled ``HunYuanVLMoTConfig``
# (text_config + vision_config nested form) so downstream dual_tower
# construction has a single config schema to worry about. There are two
# checkpoint flavours the loader handles:
#
#   (a) Self-contained VLA ckpt (the released layout):
#       embeds ``vlm_config_dict`` with ``model_type=hunyuan_vl_mot`` and a
#       populated ``text_config`` block. We instantiate ``HunYuanVLMoTConfig``
#       directly from the embedded dict -- no disk / network access.
#
#   (b) Bare VLM directory or HF Hub repo id (e.g. ``tencent/HY-Embodied-0.5``):
#       resolved via ``AutoConfig.from_pretrained``. The HY-Embodied-0.5
#       release ships an ``auto_map`` whose target file is not bundled, so
#       we pin ``trust_remote_code=False`` and, on the resulting ValueError,
#       fall back to reading ``config.json`` by hand (after stripping
#       ``auto_map``) and routing through ``HunYuanVLMoTConfig``.
#
# Returns: a ``HunYuanVLMoTConfig`` (always).
# ---------------------------------------------------------------------------


def _load_vlm_autoconfig(config_or_path):
    """Load the upstream VLM ``AutoConfig`` and return a ``HunYuanVLMoTConfig``.

    Accepts either:
      * a ``HyVLAConfig`` instance -- in which case ``config.vlm_config_dict``
        is the authoritative source: no disk / network access is needed.
      * a string ``model_path`` (local dir or HF repo id) -- in which case
        we first try plain ``AutoConfig.from_pretrained``; on failure caused
        by a broken ``auto_map`` (typical for HY-Embodied-0.5), we fall back
        to reading ``config.json`` as a dict and stripping ``auto_map``.
    """
    # ---- Path 1: embedded vlm_config_dict (self-contained VLA ckpt) ----
    if isinstance(config_or_path, HyVLAConfig) or hasattr(config_or_path, "vlm_config_dict"):
        cfg = config_or_path
        embedded = getattr(cfg, "vlm_config_dict", None)
        if embedded:
            data = dict(embedded)
            mt = data.get("model_type")
            if mt != "hunyuan_vl_mot" or "text_config" not in data:
                raise ValueError(
                    "vlm_config_dict embedded in HyVLAConfig is not in the "
                    "expected nested schema (model_type='hunyuan_vl_mot' "
                    f"with a 'text_config' "
                    f"block, got model_type={mt!r}, "
                    f"has_text_config={'text_config' in data})."
                )
            print(
                "[modeling_hy_vla] VLM AutoConfig loaded from embedded "
                "vlm_config_dict (nested hunyuan_vl_mot schema).",
                file=sys.stderr,
                flush=True,
            )
            data.pop("model_type", None)
            return HunYuanVLMoTConfig(**data)
        # Fall through: raw-VLM bootstrap (``pretrain_source`` in
        # {``vlm``, ``scratch``}); resolve from ``cfg.vlm_model_path``.
        model_path = cfg.vlm_model_path
        if not model_path:
            raise ValueError(
                "_load_vlm_autoconfig: HyVLAConfig has no "
                "``vlm_config_dict`` AND no ``vlm_model_path``. "
                "Self-contained ckpts must embed ``vlm_config_dict``; "
                "raw-VLM bootstrap flows must set ``vlm_model_path``."
            )
    else:
        model_path = config_or_path
    # ---- Path 2: AutoConfig.from_pretrained (locally registered class) ----
    # ``trust_remote_code=False`` is required so transformers raises a
    # deterministic ValueError on broken ``auto_map`` entries (which our
    # except block below repairs) instead of prompting on stdin.
    try:
        loaded = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        if isinstance(loaded, HunYuanVLMoTConfig):
            return loaded
        raise TypeError(
            f"AutoConfig at {model_path!r} dispatched to "
            f"{type(loaded).__name__}, expected HunYuanVLMoTConfig. The "
            "LeRobot HunYuanVLMoTConfig must be registered before loading."
        )
    except (OSError, ValueError) as exc:
        msg = str(exc).lower()
        if (
            "auto_map" not in msg
            and "does not appear to have a file named" not in msg
            and "trust_remote_code" not in msg
        ):
            raise

    # ---- Path 3: read config.json by hand (auto_map strip) -----------
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        cfg_path = hf_hub_download(repo_id=model_path, filename="config.json")

    with open(cfg_path, encoding="utf-8") as fp:
        data = json.load(fp)

    data.pop("auto_map", None)
    if data.get("model_type") != "hunyuan_vl_mot":
        raise ValueError(
            f"VLM config.json at {cfg_path!r} has "
            f"model_type={data.get('model_type')!r}, expected "
            "'hunyuan_vl_mot'. The loader requires the upstream "
            "HY-Embodied schema."
        )
    data.pop("model_type", None)
    return HunYuanVLMoTConfig(**data)


def _get_safe_dtype(dtype: torch.dtype, device: str | torch.device) -> torch.dtype:
    """Return ``dtype`` clamped to one supported on ``device``.

    MPS does not support float64; everything else does.
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    return dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = _get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma_alpha_dist = torch.distributions.Gamma(alpha, 1)
    gamma_beta_dist = torch.distributions.Gamma(beta, 1)

    x = gamma_alpha_dist.sample((bsize,)).to(device)
    y = gamma_beta_dist.sample((bsize,)).to(device)
    return x / (x + y)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


class HyVLAFlowMatching(nn.Module):
    """Hy-VLA flow-matching action expert.

    Owns the dual-tower (VLM + action expert) and the flow-matching
    training and sampling logic used by :class:`HyVLAPolicy`.

    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │action│        │
    │  ┌──────────►│expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x N   │        │
    │ │         │  └▲──▲──┘        │
    │ │   VLM   │   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config, language_tokenizer):
        super().__init__()
        self.config = config
        self.language_tokenizer = language_tokenizer

        # Self-contained checkpoints read the VLM config from
        # ``self.config.vlm_config_dict``; fresh models resolve it from
        # ``self.config.vlm_model_path``.
        vlm_inner_config = _load_vlm_autoconfig(self.config)

        # Expert config = VLM config with ``hidden_size`` overridden by
        # ``proj_width``. Released ckpt: ``hidden_size=1024`` (vs the VLM's
        # 2048) and ``intermediate_size=2048``; everything else (layers,
        # heads, vocab, rope) is shared with the VLM.
        expert_inner_config = copy.deepcopy(vlm_inner_config)
        expert_inner_config.hidden_size = self.config.proj_width
        expert_inner_config.intermediate_size = 2048
        if hasattr(expert_inner_config, "dense_list"):
            expert_inner_config.dense_list = [self.config.proj_width, 0]

        dual_tower_config = HyDualTowerConfig(
            vlm_config=vlm_inner_config,
            expert_config=expert_inner_config,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            config=self.config,  # outer HyVLAConfig (kept for proj_width etc.)
        )
        self.dual_tower = HyDualTower(dual_tower_config)

        # Projections are float32
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.set_requires_grad()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for the dual-tower transformer processing.

        Layout (per sample):
            <bos><hy_User>
            for each image:
                <vision_start>
                image_patch_grid interleaved with <vision_split> at the end of every row
                <vision_end>
            language_tokens
        """
        embs = []
        pad_masks = []
        att_masks = []
        modality_mask = []

        # Special tokens (BOS / role / vision boundaries / split / assistant)
        img = images[0]
        # add <｜hy_begin▁of▁sentence｜><｜hy_User｜>
        bos_token = torch.full(
            (img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_begin▁of▁sentence｜>")
        )
        bos_token = bos_token.to(img.device)
        bos_emb = self.dual_tower.embed_language_tokens(bos_token)
        hy_user_token = torch.full(
            (img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_User｜>")
        )
        hy_user_token = hy_user_token.to(img.device)
        hy_user_emb = self.dual_tower.embed_language_tokens(hy_user_token)

        # add <｜hy_place▁holder▁no▁666｜> vision_start_token
        vision_start_token = torch.full(
            (img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_place▁holder▁no▁666｜>")
        )
        vision_start_token = vision_start_token.to(img.device)
        vision_start_emb = self.dual_tower.embed_language_tokens(vision_start_token)

        # add <｜hy_place▁holder▁no▁666｜> vision_end_token
        vision_end_token = torch.full(
            (img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_place▁holder▁no▁667｜>")
        )
        vision_end_token = vision_end_token.to(img.device)
        vision_end_emb = self.dual_tower.embed_language_tokens(vision_end_token)

        # add <｜hy_place▁holder▁no▁666｜> vision_split_token
        vision_split_token = torch.full(
            (img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_place▁holder▁no▁671｜>")
        )
        vision_split_token = vision_split_token.to(img.device)
        vision_split_emb = self.dual_tower.embed_language_tokens(vision_split_token)

        # 1. Add [bos_token, hy_user_token]
        embs.extend([bos_emb, hy_user_emb])
        pad_masks.append(torch.ones((images[0].shape[0], 2), dtype=torch.bool, device=images[0].device))
        att_masks.extend([1, 1])
        modality_mask.extend([False, False])

        # Track image-token index ranges so the visual-segment attention mask
        # tweak (see ``_apply_visual_segment_mask``) can address them later.
        image_idx_ranges = []  # per-row patch ranges (excludes split tokens)
        image_full_ranges = []  # full per-image span (patches + split rows)

        # 2. Add vision_start + image patches with row-wise split tokens + vision_end
        for i, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):
            bs = img.shape[0]

            # vision_start
            embs.append(vision_start_emb)
            pad_masks.append(torch.ones((bs, 1), dtype=torch.bool, device=img.device))
            att_masks.append(1)
            modality_mask.append(False)

            # embed image (bs, num_patches, emb_dim)
            img_emb = self.dual_tower.embed_image(img).to(dtype=torch.bfloat16)
            num_patches, emb_dim = img_emb.shape[1], img_emb.shape[2]
            grid_size = int(num_patches**0.5)
            assert grid_size * grid_size == num_patches, "num_patches must be square"

            img_emb_grid = img_emb.view(bs, grid_size, grid_size, emb_dim)
            split_expanded = vision_split_emb.unsqueeze(1).expand(bs, grid_size, 1, emb_dim)
            img_emb_with_split = torch.cat([img_emb_grid, split_expanded], dim=2)
            img_emb_with_split = img_emb_with_split.view(bs, -1, emb_dim)
            embs.append(img_emb_with_split)

            row_len = grid_size + 1
            total_img_tokens = grid_size * row_len
            start_idx = len(att_masks)

            # Per-row patch ranges (exclude the trailing split token of each row).
            row_ranges = [
                (start_idx + r * row_len, start_idx + r * row_len + grid_size) for r in range(grid_size)
            ]
            image_idx_ranges.extend(row_ranges)

            # Full span of this image's visual segment (patches + split tokens).
            image_full_ranges.append((start_idx, start_idx + total_img_tokens))

            att_masks.extend([1] * total_img_tokens)
            # Each grid row: ``grid_size`` patch tokens (modality=True) + 1 split token (False).
            modality_mask.extend(([True] * grid_size + [False] * 1) * grid_size)

            img_mask_expanded = img_mask[:, None].expand(bs, total_img_tokens)
            pad_masks.append(img_mask_expanded)

            # vision_end
            embs.append(vision_end_emb)
            pad_masks.append(torch.ones((bs, 1), dtype=torch.bool, device=img.device))
            att_masks.append(1)
            modality_mask.append(False)

        # 3. Language tokens
        lang_emb = self.dual_tower.embed_language_tokens(lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks.extend([1] * num_lang_embs)
        modality_mask.extend([False] * num_lang_embs)

        # 4. Stack into tensors
        bsize = images[0].shape[0]
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1).to(torch.bool)

        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, -1)

        modality_mask = torch.tensor(modality_mask, dtype=torch.bool, device=pad_masks.device)
        modality_mask = modality_mask[None, :].expand(bsize, -1)

        return embs, pad_masks, att_masks, modality_mask, image_idx_ranges, image_full_ranges

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions and timestep for the action expert.

        Emits a single absolute state token from ``state`` (passed through
        ``state_proj`` and cast to bf16), then the action / time embedding
        block. The state token shares one attention block with the action
        chunk: leading ``att_masks=1`` followed by ``0`` for the action
        tokens.
        """
        embs = []
        pad_masks = []
        att_masks = []
        modality_mask = []

        # --- State token ----------------------------------------------------
        assert state is not None, "embed_suffix: ``state`` is required."
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        # (B, D) -> (B, 1, D)
        state_block = state_emb[:, None, :]
        embs.append(state_block)

        bsize = state_block.shape[0]
        T_state = state_block.shape[1]
        device = state_block.device

        state_mask = torch.ones(bsize, T_state, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # All state tokens share one attention block: leading 1, rest 0.
        # Mirrors the action-chunk wiring further down.
        att_masks += [1] + [0] * (T_state - 1)
        modality_mask += [True] * T_state

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=torch.bfloat16)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions.to(torch.bfloat16))  # torch.float32 -> bf16

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)  # torch.float32

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))
        modality_mask += [True] * (self.config.n_action_steps)

        embs = torch.cat(embs, dim=1)  # torch.bfloat16
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        modality_mask = torch.tensor(modality_mask, dtype=torch.bool, device=pad_masks.device)
        modality_mask = modality_mask[None, :].expand(bsize, len(modality_mask))

        return embs, pad_masks, att_masks, modality_mask

    def _apply_visual_segment_mask(
        self,
        att_2d_masks,
        image_idx_ranges,
        image_full_ranges,
    ):
        """In-place rewrite the visual-segment portion of ``att_2d_masks``.

        Two scopes are selectable via ``self.config.visual_segment_isolation``:

        * ``False`` -- *patch-only* (default, backward-compatible):
          1. collect every image's ``image_idx_ranges`` (image-patch tokens,
             excluding the per-row split tokens) and zero out their pairwise
             visibility;
          2. inside each image's ``image_full_range``, set the image-patch
             tokens to be bidirectionally visible.
          Image-patch / split-row tokens still see segment-external tokens
          via the causal mask, which differs slightly from the VLM-time
          eager MoT attention behaviour.

        * ``True`` -- *full-segment isolation* (matches
          eager MoT attention): for each image's
          ``image_full_range`` (image patches + split / newline rows,
          excluding ``vision_start`` / ``vision_end``):
          1. clear all visibility on the rows of those tokens;
          2. enable bidirectional visibility within the segment.
          The released RoboTwin post-train ckpt was trained under this mode,
          so reproducing it requires ``visual_segment_isolation=True`` in
          ``config.json``.

        Args:
            att_2d_masks: ``(B, S, S)`` bool tensor; modified in place.
            image_idx_ranges: per-row image-patch ``[start, end)`` ranges
                (excluding split tokens).
            image_full_ranges: per-image ``[start, end)`` ranges covering
                image patches plus split / newline rows.
        """
        if getattr(self.config, "visual_segment_isolation", False):
            # Full-segment isolation: rewrite each image_full_range as a
            # self-contained bidirectional block.
            for img_full_start, img_full_end in image_full_ranges:
                full_range_idx = torch.arange(img_full_start, img_full_end, device=att_2d_masks.device)
                # Clear outward visibility for image-patch + split rows.
                att_2d_masks[:, full_range_idx, :] = False
                # Re-enable visibility within the segment.
                att_2d_masks[:, full_range_idx[:, None], full_range_idx[None, :]] = True
            return

        # Patch-only (default): only adjust image-patch tokens; split rows
        # stay on the causal pathway.
        # Step 1: clear pairwise visibility between every image-patch token
        # (this also drops the causal-pathway visibility between them).
        all_img_indices = []
        for s, e in image_idx_ranges:
            all_img_indices.extend(range(s, e))
        if all_img_indices:
            idx = torch.tensor(all_img_indices, device=att_2d_masks.device)
            att_2d_masks[:, idx[:, None], idx[None, :]] = False

        # Step 2: re-enable bidirectional visibility among image-patch
        # tokens that belong to the same image.
        for img_full_start, img_full_end in image_full_ranges:
            img_indices = []
            for s, e in image_idx_ranges:
                if s >= img_full_start and e <= img_full_end:
                    img_indices.extend(range(s, e))
            if img_indices:
                idx = torch.tensor(img_indices, device=att_2d_masks.device)
                att_2d_masks[:, idx[:, None], idx[None, :]] = True

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state=None,
        actions=None,
        noise=None,
        time=None,
        lang_token_labels=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        losses_flow = None
        losses_ntp = None

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            modality_mask_prefix,
            image_idx_ranges,
            image_full_ranges,
        ) = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # action, text + action
        if actions is not None:
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)

            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            suffix_embs, suffix_pad_masks, suffix_att_masks, modality_mask_suffix = self.embed_suffix(
                state,
                x_t,
                time,
            )

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # text only
        else:
            suffix_embs = None
            pad_masks = torch.cat([prefix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Adjust visual-segment attention according to the configured scope.
        self._apply_visual_segment_mask(att_2d_masks, image_idx_ranges, image_full_ranges)

        (prefix_out, suffix_out), _, att_vis_output, _ = self.dual_tower.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            modality_masks=[modality_mask_prefix, modality_mask_suffix],
        )

        # Flow matching prediction
        if actions is not None:
            suffix_out = suffix_out[:, -self.config.n_action_steps :]
            v_t = self.action_out_proj(suffix_out)  # torch.float32 -> bf16
            losses_flow = F.mse_loss(u_t.float(), v_t.float(), reduction="none")  # bf16 -> torch.float32

        # Next-token prediction
        if lang_token_labels is not None:
            attention_mask = None
            logits = self.dual_tower.vlm.language_model.lm_head(prefix_out)

            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., -self.config.tokenizer_max_length : -1, :]
            shift_labels = lang_token_labels[..., 1:]

            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()

            # Flatten the tokens
            losses_ce = nn.CrossEntropyLoss(
                reduction="none",
                ignore_index=self.dual_tower.vlm.config.ignore_index,
            )

            flat_logits = shift_logits.view(-1, self.dual_tower.vlm.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            losses_ntp = losses_ce(flat_logits, flat_labels)

        return losses_flow, losses_ntp

    # @torch.compile(mode="reduce-overhead")
    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None, vis_attn=False
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            modality_mask_prefix,
            image_idx_ranges,
            image_full_ranges,
        ) = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Adjust visual-segment attention according to the configured scope.
        self._apply_visual_segment_mask(prefix_att_2d_masks, image_idx_ranges, image_full_ranges)

        # Compute image and language key value cache
        (prefix_out, _), past_key_values, _, _ = self.dual_tower.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            modality_masks=[modality_mask_prefix, None],
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t, att_vis_output = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt

        if vis_attn:
            # Strip non-patch tokens from att_vis_output, leaving the
            # contiguous (B, H, suffix_len, num_patches * num_views)
            # tensor that downstream visualisation tooling expects.
            all_img_indices = []
            for s, e in image_idx_ranges:
                all_img_indices.extend(range(s, e))
            img_idx_tensor = torch.tensor(all_img_indices, dtype=torch.long, device=device)

            cleaned_att = []
            for layer_att in att_vis_output:
                cleaned_att.append(layer_att[:, :, :, img_idx_tensor])
            return x_t, cleaned_att

        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        # IMPORTANT: copy the past_key_values, or its size will increase during n-step denoise.
        past_key_values_vlm = copy.deepcopy(past_key_values)

        suffix_embs, suffix_pad_masks, suffix_att_masks, modality_mask_suffix = self.embed_suffix(
            state,
            x_t,
            timestep,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _, att_vis_output, _ = self.dual_tower.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values_vlm,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            modality_masks=[None, modality_mask_suffix],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        v_t = self.action_out_proj(suffix_out)  # bf16 -> torch.float32
        return v_t, att_vis_output


__all__ = ["HyVLAFlowMatching"]
