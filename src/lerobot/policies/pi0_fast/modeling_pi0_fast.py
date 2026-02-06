#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import builtins
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.utils.import_utils import _scipy_available, _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _scipy_available:
    from scipy.fftpack import idct
else:
    idct = None

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoTokenizer
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    PaliGemmaForConditionalGeneration = None
    AutoTokenizer = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)


class ActionSelectKwargs(TypedDict, total=False):
    temperature: float | None


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PI0FastPaliGemma(nn.Module):
    """PaliGemma model for PI0Fast"""

    def __init__(
        self,
        vlm_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            # prefix_output to be used for the language head
            # shape: [batch_size, seq_len, hidden_size] with hidden_size = 2048
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        return [prefix_output, suffix_output], prefix_past_key_values


class PI0FastPytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI0Fast PyTorch model."""

    def __init__(
        self,
        config: PI0FastConfig,
        rtc_processor: RTCProcessor | None = None,
        paligemma_tokenizer: "AutoTokenizer | None" = None,
    ):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor
        self._paligemma_tokenizer = paligemma_tokenizer

        paligemma_config = get_gemma_config(config.paligemma_variant)

        self.paligemma_with_expert = PI0FastPaliGemma(
            paligemma_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions_fast = torch.compile(self.sample_actions_fast, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

        msg = """An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""

        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        # Call the proper gradient_checkpointing_enable() method with use_reentrant=False for better memory efficiency
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logging.info("Enabled gradient checkpointing for PI0FastPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        # Call the proper gradient_checkpointing_disable() method
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing_disable()
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing_disable()
        logging.info("Disabled gradient checkpointing for PI0FastPytorch model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks, dtype=None):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        result = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        if dtype is not None:
            result = result.to(dtype=dtype)
        return result

    def embed_prefix_fast(
        self,
        images,
        img_masks,
        tokens,
        masks,
        fast_action_tokens=None,
        fast_action_masks=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Embed images, language tokens, and FAST action tokens.

        Attention pattern:
        - Images + Language: bidirectional among themselves
        - FAST: attend to images + language, causal among themselves

        Args:
            images: List of image tensors
            img_masks: List of image masks
            tokens: Language instruction tokens
            masks: Attention masks for tokens
            fast_action_tokens: FAST action tokens (discrete token IDs)
            fast_action_masks: Padding masks for FAST action tokens

        Returns:
            embs: Concatenated embeddings [images, tokens, fast_action_tokens]
            pad_masks: Padding masks
            att_masks: 2D attention mask
            total_T_images: Total number of image tokens
            num_fast_embs: Number of FAST action token embeddings
        """
        embs = []
        pad_masks = []
        att_mask_segments = []
        total_t_images = 0
        num_fast_embs = 0

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_mask_segments.append(("image", num_img_embs))
            total_t_images += num_img_embs

        # Process language instruction tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_mask_segments.append(("language", num_lang_embs))

        # Process FAST action tokens (discrete token IDs)
        if fast_action_tokens is not None:

            def fast_action_embed_func(fast_action_tokens):
                fast_emb = self.paligemma_with_expert.embed_language_tokens(fast_action_tokens)
                fast_emb_dim = fast_emb.shape[-1]
                return fast_emb * math.sqrt(fast_emb_dim)

            fast_action_emb = self._apply_checkpoint(fast_action_embed_func, fast_action_tokens)
            embs.append(fast_action_emb)

            num_fast_embs = fast_action_tokens.shape[1]
            pad_masks.append(fast_action_masks)
            att_mask_segments.append(("fast", num_fast_embs))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        # Create custom 2D attention mask:
        # - Images + Language: bidirectional among themselves
        # - FAST: attend to images + language, causal among themselves
        att_masks = self._create_custom_attention_mask_fast(att_mask_segments, pad_masks, bsize)

        return embs, pad_masks, att_masks, total_t_images, num_fast_embs

    def _create_custom_attention_mask_fast(self, att_mask_segments, pad_masks, bsize):
        """Create custom 2D attention mask.

        Attention rules:
        - Images + Language: bidirectional among themselves
        - FAST: attend to images + language, causal among themselves
        """
        total_len = sum(length for _, length in att_mask_segments)
        device = pad_masks.device

        att_2d_masks = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=device)

        positions = []
        current_pos = 0
        for seg_type, seg_len in att_mask_segments:
            positions.append((seg_type, current_pos, current_pos + seg_len))
            current_pos += seg_len

        for _i, (query_type, query_start, query_end) in enumerate(positions):
            for _j, (key_type, key_start, key_end) in enumerate(positions):
                # Images and Language can attend to each other bidirectionally
                if (
                    query_type in ["image", "language"]
                    and key_type in ["image", "language"]
                    or query_type == "fast"
                    and key_type in ["image", "language"]
                ):
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = True

                # FAST tokens attend causally to themselves
                elif query_type == "fast" and key_type == "fast":
                    fast_len = query_end - query_start
                    causal_mask = torch.tril(torch.ones(fast_len, fast_len, dtype=torch.bool, device=device))
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = causal_mask[None, :, :]

        # Apply padding masks
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        att_2d_masks = att_2d_masks & pad_2d_masks

        return att_2d_masks

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        fast_action_tokens,
        fast_action_masks,
    ) -> dict:
        """Forward pass for PI0Fast.

        This implements the Pi0FAST training objective: predict next action token
        using cross-entropy loss.

        Args:
            images: List of image tensors
            img_masks: List of image masks
            tokens: Language instruction tokens
            masks: Attention masks for tokens
            fast_action_tokens: Discrete action token IDs [B, max_action_tokens]
            fast_action_masks: Padding masks for fast action tokens [B, max_action_tokens]

        Returns:
            Dictionary with 'fast_loss' and 'loss' keys
        """
        if fast_action_tokens is None or fast_action_masks is None:
            raise ValueError("fast_action_tokens and fast_action_masks are required for FAST-only mode")

        # Embed prefix with FAST tokens
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_t_images, num_fast_embs = (
            self.embed_prefix_fast(
                images,
                img_masks,
                tokens,
                masks,
                fast_action_tokens=fast_action_tokens,
                fast_action_masks=fast_action_masks,
            )
        )

        # Convert embeddings to bfloat16 if needed
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # for next-token prediction, input tokens [0:T-1] to predict tokens [1:T]
        input_embs = prefix_embs
        input_pad_masks = prefix_pad_masks
        input_att_masks = prefix_att_masks

        position_ids = torch.cumsum(input_pad_masks, dim=1) - 1
        att_2d_4d = self._prepare_attention_masks_4d(input_att_masks, dtype=input_embs.dtype)

        # forward pass through paligemma (language model)
        (prefix_out, _), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[input_embs, None],  # No suffix/action expert
            use_cache=False,
            adarms_cond=[None, None],
        )

        # Get logits for FAST action tokens using the FAST LM head
        # only compute logits for the positions that predict FAST tokens
        lm_head = self.paligemma_with_expert.paligemma.lm_head

        # Targets are the FAST action tokens
        fast_targets = fast_action_tokens  # (B, num_fast_embs)

        # extract logits for FAST token prediction
        fast_hidden = prefix_out[:, -fast_targets.shape[1] :, :]
        fast_logits_for_pred = lm_head(fast_hidden)  # (B, num_fast_embs, gemma_vocab_size)

        # Shift left for next-step prediction and shift target
        # logits[:, i] predicts targets[:, i+1]
        fast_logits_for_pred = fast_logits_for_pred[:, :-1, :]  # shift logits left
        fast_targets = fast_targets[:, 1:]  # shift targets right
        fast_action_masks = fast_action_masks[:, 1:]  # shift masks to match targets

        # compute cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        fast_logits_flat = fast_logits_for_pred.reshape(-1, fast_logits_for_pred.size(-1))
        fast_targets_flat = fast_targets.reshape(-1)

        fast_loss_per_token = loss_fct(fast_logits_flat, fast_targets_flat)
        fast_loss_per_token = fast_loss_per_token.reshape(fast_targets.shape)

        # apply mask and compute mean loss
        masked_fast_loss = fast_loss_per_token * fast_action_masks.float()
        fast_loss = masked_fast_loss.sum() / fast_action_masks.sum().clamp(min=1)

        return {
            "ce_loss": fast_loss,
            "loss": fast_loss,
        }

    @torch.no_grad()
    def sample_actions_fast(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps=None,
        temperature=0.0,
    ) -> torch.Tensor:
        """
        Inefficient but safe autoregressive decoding for FAST tokens.
        Matches the pattern of _generate_subtask_tokens.
        TODO: jadechoghari, should we move this logic to PI0FastPolicy class?
        """
        if max_decoding_steps is None:
            max_decoding_steps = self.config.max_action_tokens

        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.paligemma_with_expert.paligemma.lm_head

        # add bos token after tokens
        bos_token = torch.full(
            (bsize, 1), self._paligemma_tokenizer.bos_token_id, dtype=torch.long, device=device
        )
        tokens = torch.cat([tokens, bos_token], dim=1)
        masks = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)

        # 1. Initial Embedding (matches training prefix)
        # prefix_embs will include [Images, Language Prompt, BOS]
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_t_images, _ = self.embed_prefix_fast(
            images, img_masks, tokens, masks, fast_action_tokens=None, fast_action_masks=None
        )

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        generated_action_tokens = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)

        # 2. Decoding Loop (each step re-computes full sequence)
        for t in range(max_decoding_steps):
            # always re-calculate position IDs from the current pad mask
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            att_4d = self._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)

            # full forward pass (no kv cache)
            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=att_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                adarms_cond=[None, None],
            )

            # predict next token from the very last sequence position
            last_logits = lm_head(prefix_out[:, -1:, :])  # (B, 1, vocab_size)

            if temperature > 0:
                probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

            generated_action_tokens[:, t] = next_token.squeeze(-1)

            # 3. Update sequence for next iteration (unless it's the last step)
            if t < max_decoding_steps - 1:
                # embed the newly generated token
                next_token_emb = self.paligemma_with_expert.embed_language_tokens(next_token)
                next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
                if prefix_embs.dtype == torch.bfloat16:
                    next_token_emb = next_token_emb.to(dtype=torch.bfloat16)

                # append to embeddings
                prefix_embs = torch.cat([prefix_embs, next_token_emb], dim=1)

                # update padding mask (new token is always valid/1)
                prefix_pad_masks = torch.cat(
                    [prefix_pad_masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1
                )

                # update 2d attention mask: grow the matrix
                old_len = prefix_att_masks.shape[1]
                new_len = old_len + 1
                new_att_masks = torch.zeros((bsize, new_len, new_len), dtype=torch.bool, device=device)
                new_att_masks[:, :old_len, :old_len] = prefix_att_masks
                # new token attends to all non-padding tokens in the updated sequence
                new_att_masks[:, -1, :] = prefix_pad_masks
                prefix_att_masks = new_att_masks
        return generated_action_tokens

    @torch.no_grad()
    def sample_actions_fast_kv_cache(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps=None,
        temperature=0.0,
    ) -> torch.Tensor:
        """
        Optimized autoregressive decoding for FAST tokens using KV Caching.
        """
        if max_decoding_steps is None:
            max_decoding_steps = self.config.max_action_tokens

        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.paligemma_with_expert.paligemma.lm_head

        # --- 1. PREFILL PHASE ---
        # Process Images + Text Prompt + BOS token once to populate the KV cache.

        # Add BOS token to the prompt
        bos_token = torch.full(
            (bsize, 1), self._paligemma_tokenizer.bos_token_id, dtype=torch.long, device=device
        )
        tokens_in = torch.cat([tokens, bos_token], dim=1)
        masks_in = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)

        # Embed prefix [Images, Language, BOS]
        # fast_action_tokens=None means we are just embedding the condition (images+text)
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_t_images, _ = self.embed_prefix_fast(
            images, img_masks, tokens_in, masks_in, fast_action_tokens=None, fast_action_masks=None
        )

        # Ensure correct precision (bfloat16/float32)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Create position IDs (cumsum of mask - 1)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Create 4D mask for the prefix
        att_4d = self._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)

        # Forward pass (Prefill) with use_cache=True
        # We only pass [prefix_embs, None] because we aren't using the suffix (expert) model yet
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,  # Enable caching
            adarms_cond=[None, None],
        )

        # Sample the first action token from the last logit of the prefix
        last_logits = lm_head(prefix_out[:, -1:, :])  # (B, 1, V)
        if temperature > 0:
            probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

        # Initialize storage for generated tokens
        generated_action_tokens = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)
        generated_action_tokens[:, 0] = next_token.squeeze(-1)

        # Track valid tokens mask (0 for pad, 1 for valid)
        # We need this to tell the new token what it can attend to (images + text + past actions)
        current_pad_mask = prefix_pad_masks

        # --- 2. DECODING PHASE ---
        # Generate remaining tokens one by one using the cache.

        for t in range(1, max_decoding_steps):
            # Embed the single previous token
            # We use embed_language_tokens directly to avoid overhead of full prefix embedding
            next_token_emb = self.paligemma_with_expert.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            if prefix_embs.dtype == torch.bfloat16:
                next_token_emb = next_token_emb.to(dtype=torch.bfloat16)

            # Update Pad Mask: append 1s for the new valid token
            new_column = torch.ones((bsize, 1), dtype=torch.bool, device=device)
            current_pad_mask = torch.cat([current_pad_mask, new_column], dim=1)

            # Update Position IDs for the single new token
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()

            # Create Attention Mask for the single new step
            # The new token attends to all valid tokens in history (captured by current_pad_mask).
            # Shape becomes (B, 1, 1, Total_Len) which works with HF's cache logic.
            step_att_mask = self._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1), dtype=next_token_emb.dtype
            )

            # Forward pass (Decoding step)
            # input_embeds is just the new token (B, 1, D)
            (step_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,  # Pass updated cache
                inputs_embeds=[next_token_emb, None],
                use_cache=True,
                adarms_cond=[None, None],
            )

            # Sample next token
            last_logits = lm_head(step_out[:, -1:, :])
            if temperature > 0:
                probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

            generated_action_tokens[:, t] = next_token.squeeze(-1)

        return generated_action_tokens


class PI0FastPolicy(PreTrainedPolicy):
    """PI0Fast Policy for LeRobot."""

    config_class = PI0FastConfig
    name = "pi0_fast"

    def __init__(
        self,
        config: PI0FastConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Load tokenizers first
        try:
            from transformers import AutoProcessor, AutoTokenizer

            # Load FAST tokenizer
            self.action_tokenizer = AutoProcessor.from_pretrained(
                config.action_tokenizer_name, trust_remote_code=True
            )

            # Load PaliGemma tokenizer for token conversion
            self._paligemma_tokenizer = AutoTokenizer.from_pretrained(
                config.text_tokenizer_name, trust_remote_code=True, add_eos_token=True, add_bos_token=False
            )

            logging.info("Loaded FAST tokenizer for action detokenization")
        except Exception as e:
            logging.error(f"Failed to load FAST tokenizer for action detokenization: {e}")
            logging.error("Tokenizer loading is required for proper policy initialization; aborting.")
            raise RuntimeError("Failed to load required tokenizers for PI0FastPolicy initialization") from e

        # Initialize the core PI0Fast model
        self.init_rtc_processor()
        self.model = PI0FastPytorch(
            config, rtc_processor=self.rtc_processor, paligemma_tokenizer=self._paligemma_tokenizer
        )

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI0Fast model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Now manually load and remap the state dict
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    use_auth_token=kwargs.get("use_auth_token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)
            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                    if remap_count <= 10:  # Only print first 10 to avoid spam
                        print(f"Remapped: {key} -> {new_key}")
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not remap state dict keys: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _paligemma_tokens_to_act_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Converts PaliGemma tokens back to action tokens (inverse of _act_tokens_to_paligemma_tokens).

        Args:
            tokens: PaliGemma token IDs

        Returns:
            Action token IDs
        """
        return self._paligemma_tokenizer.vocab_size - 1 - self.config.fast_skip_tokens - tokens

    def decode_actions_with_fast(
        self, token_ids: list[int], time_horizon: int, action_dim: int, relaxed_decoding: bool = True
    ) -> np.ndarray:
        """
        Decodes action token IDs back to continuous action values using the FAST tokenizer.

        Args:
            token_ids: List of token IDs to decode.
            time_horizon: The number of timesteps for actions.
            action_dim: The dimensionality of each action.
            relaxed_decoding: Whether to use relaxed decoding (allows partial sequences).

        Returns:
            A numpy array representing the decoded actions.
        """
        decoded_actions = []

        for token in token_ids:
            try:
                decoded_tokens = self.action_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.action_tokenizer.min_token

                if relaxed_decoding:
                    # expected sequence length
                    expected_seq_len = time_horizon * action_dim
                    diff = expected_seq_len - decoded_dct_coeff.shape[0]

                    # apply truncation if too long
                    if diff < 0:
                        decoded_dct_coeff = decoded_dct_coeff[:expected_seq_len]  # truncate on the right

                    # apply padding if too short
                    elif diff > 0:
                        decoded_dct_coeff = np.pad(
                            decoded_dct_coeff, (0, diff), mode="constant", constant_values=0
                        )

                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
                assert decoded_dct_coeff.shape == (
                    time_horizon,
                    action_dim,
                ), (
                    f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({time_horizon}, {action_dim})"
                )

            except Exception as e:
                logging.warning(f"Error decoding tokens: {e}")
                logging.warning(f"Tokens: {token}")
                decoded_dct_coeff = np.zeros((time_horizon, action_dim))

            decoded_actions.append(
                idct(decoded_dct_coeff / self.action_tokenizer.scale, axis=0, norm="ortho")
            )

        return np.stack(decoded_actions)

    def detokenize_actions(self, tokens: torch.Tensor, action_horizon: int, action_dim: int) -> torch.Tensor:
        """
        Detokenizes action tokens back to continuous actions.

        This method converts predicted action tokens from the model back to continuous action values
        using the FAST tokenizer. It handles the conversion from PaliGemma token space to action token
        space, then decodes the action tokens to continuous values using DCT decoding.

        Args:
            tokens: The input tensor of tokenized outputs. Shape: (B, seq_len) or (seq_len,)
            action_horizon: The number of timesteps for actions.
            action_dim: The dimensionality of each action.

        Returns:
            The continuous action tensor. Shape: (B, action_horizon, action_dim) or (action_horizon, action_dim)
        """
        if self.action_tokenizer is None or self._paligemma_tokenizer is None:
            raise ValueError(
                "Action tokenizer not initialized. Make sure fast_only=True in config and tokenizers loaded successfully."
            )

        # Handle single sample (add batch dimension)
        single_sample = tokens.dim() == 1
        if single_sample:
            tokens = tokens.unsqueeze(0)

        # Convert token IDs to token strings
        decoded_tokens = [self._paligemma_tokenizer.convert_ids_to_tokens(seq.tolist()) for seq in tokens]
        # Get the token sequence for "Action: " to remove it
        action_prefix_ids = self._paligemma_tokenizer.encode("Action: ", add_special_tokens=False)
        action_prefix_tokens = self._paligemma_tokenizer.convert_ids_to_tokens(action_prefix_ids)
        action_prefix_len = len(action_prefix_tokens)

        # Clean tokens by removing everything after the first "|" (end-of-action marker)
        # and removing all occurrences of "Action: " token sequence
        # assert that beginning contain "Action: "
        if self.config.validate_action_token_prefix:
            for token_seq in decoded_tokens:
                assert len(token_seq) >= 2 and token_seq[0] == "Action" and token_seq[1] == ":", (
                    f"Token sequence does not start with ['Action', ':']: {token_seq}"
                )

        cleaned_tokens = []
        for token_seq in decoded_tokens:
            # Remove everything after "|"
            if "|" in token_seq:
                token_seq = token_seq[: token_seq.index("|")]

            # Remove all occurrences of "Action: " token sequence
            i = 0
            while i <= len(token_seq) - action_prefix_len:
                if token_seq[i : i + action_prefix_len] == action_prefix_tokens:
                    # Found a match, remove it
                    token_seq = token_seq[:i] + token_seq[i + action_prefix_len :]
                else:
                    i += 1

            cleaned_tokens.append(token_seq)

        # Convert token strings back to IDs
        raw_action_tokens = [
            torch.tensor(
                self._paligemma_tokenizer.convert_tokens_to_ids(token_seq),
                dtype=torch.long,
                device=tokens.device,
            )
            for token_seq in cleaned_tokens
        ]

        # Convert PaliGemma tokens to action tokens
        action_tokens = [
            self._paligemma_tokens_to_act_tokens(raw_action_token) for raw_action_token in raw_action_tokens
        ]

        # Decode action tokens to continuous actions
        actions = self.decode_actions_with_fast(
            action_tokens, time_horizon=action_horizon, action_dim=action_dim
        )

        # Convert to tensor and return
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=tokens.device)

        # Remove batch dimension if input was single sample
        if single_sample:
            actions_tensor = actions_tensor.squeeze(0)

        return actions_tensor

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)

        # FAST-only mode: use autoregressive decoding
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Get decoding parameters
        temperature = self.config.temperature
        max_decoding_steps = self.config.max_decoding_steps

        # Sample action tokens autoregressively
        if self.config.use_kv_cache:
            action_tokens = self.model.sample_actions_fast_kv_cache(
                images,
                img_masks,
                tokens,
                masks,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
            )
        else:
            action_tokens = self.model.sample_actions_fast(
                images,
                img_masks,
                tokens,
                masks,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
            )

        # Detokenize action tokens to continuous actions
        action_horizon = self.config.n_action_steps
        action_dim = self.config.output_features[ACTION].shape[0]

        continuous_actions = self.detokenize_actions(
            action_tokens, action_horizon=action_horizon, action_dim=action_dim
        )

        return continuous_actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training."""

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)

        # Get FAST action tokens from batch
        fast_action_tokens = batch.get(ACTION_TOKENS)  # (B, max_action_tokens)
        fast_action_masks = batch.get(ACTION_TOKEN_MASK)  # (B, max_action_tokens)

        # Use full language tokens (no separation into high_level_task and subtask)
        tokens = batch.get(OBS_LANGUAGE_TOKENS)
        masks = batch.get(OBS_LANGUAGE_ATTENTION_MASK)

        if fast_action_tokens is None or fast_action_masks is None:
            raise ValueError(
                f"PI0Fast requires {ACTION_TOKENS} and {ACTION_TOKEN_MASK} to be present in the batch"
            )

        loss_dict = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            fast_action_tokens,
            fast_action_masks,
        )

        loss = loss_dict["loss"]
        detailed_loss_dict = {
            "loss": loss.item(),
            "ce_loss": loss_dict["ce_loss"].item(),
        }
        return loss, detailed_loss_dict
