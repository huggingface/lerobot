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

import logging
import math
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn

from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE
from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import CONFIG_MAPPING
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    PaliGemmaForConditionalGeneration = None

def _load_get_gemma_config():
    try:
        module = import_module("lerobot.policies.pi05.modeling_pi05")
    except ModuleNotFoundError:
        module = import_module("lerobot.policies.pi05_full.modeling_pi05")
    return module.get_gemma_config


get_gemma_config = _load_get_gemma_config()


PI05_VLM_KEY_PREFIX = "paligemma_with_expert.paligemma."
PI05_PROJECTION_DIM = 2048


@dataclass
class RECAPValueNetworkConfig:
    """Configuration for the standalone RECAP value network."""

    paligemma_variant: str = "gemma_300m"
    precision: Literal["bfloat16", "float32"] = "float32"
    image_size: int = 224
    max_state_dim: int = 32
    freeze_vision_encoder: bool = False
    freeze_backbone: bool = False
    freeze_embeddings: bool = False
    num_value_bins: int = 201
    dropout: float = 0.1
    pretrained_path: str | None = None


class RECAPValueNetwork(nn.Module):
    """
    Standalone distributional value network for RECAP.

    The network predicts a categorical value distribution over bins in [-1, 0]
    and recovers a scalar expected value from this distribution.
    """

    def __init__(self, config: RECAPValueNetworkConfig):
        super().__init__()
        if PaliGemmaForConditionalGeneration is None or CONFIG_MAPPING is None:
            raise ImportError("transformers is required to instantiate RECAPValueNetwork.")

        if config.pretrained_path and config.paligemma_variant != "gemma_2b":
            logging.warning(
                f"pretrained_path is set but paligemma_variant={config.paligemma_variant!r}; "
                "pi0.5 base uses gemma_2b for its VLM — overriding to gemma_2b."
            )
            config.paligemma_variant = "gemma_2b"

        self.config = config
        gemma_config = get_gemma_config(config.paligemma_variant)

        projection_dim = PI05_PROJECTION_DIM if config.pretrained_path else gemma_config.width

        paligemma_config_hf = CONFIG_MAPPING["paligemma"]()
        paligemma_config_hf._vocab_size = 257152  # noqa: SLF001
        paligemma_config_hf.image_token_index = 257152
        paligemma_config_hf.text_config.hidden_size = gemma_config.width
        paligemma_config_hf.text_config.intermediate_size = gemma_config.mlp_dim
        paligemma_config_hf.text_config.num_attention_heads = gemma_config.num_heads
        paligemma_config_hf.text_config.head_dim = gemma_config.head_dim
        paligemma_config_hf.text_config.num_hidden_layers = gemma_config.depth
        paligemma_config_hf.text_config.num_key_value_heads = gemma_config.num_kv_heads
        paligemma_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        paligemma_config_hf.text_config.torch_dtype = "float32"
        paligemma_config_hf.text_config.vocab_size = 257152
        paligemma_config_hf.vision_config.image_size = config.image_size
        paligemma_config_hf.vision_config.intermediate_size = 4304
        paligemma_config_hf.vision_config.projection_dim = projection_dim
        paligemma_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        paligemma_config_hf.vision_config.torch_dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGeneration(config=paligemma_config_hf)

        if config.precision == "bfloat16":
            self.paligemma = self.paligemma.to(dtype=torch.bfloat16)
        elif config.precision == "float32":
            self.paligemma = self.paligemma.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid precision: {config.precision}")

        if config.freeze_backbone:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False
        else:
            if config.freeze_vision_encoder:
                vision_tower = self._get_vision_tower()
                vision_tower.eval()
                for param in vision_tower.parameters():
                    param.requires_grad = False
            if config.freeze_embeddings:
                embed = self.paligemma.get_input_embeddings()
                for param in embed.parameters():
                    param.requires_grad = False

        self.state_proj = nn.Linear(config.max_state_dim, gemma_config.width)

        self.fusion_head = nn.Sequential(
            nn.Linear(gemma_config.width, gemma_config.width),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(gemma_config.width, gemma_config.width),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(gemma_config.width, config.num_value_bins)

        self.register_buffer(
            "value_bin_support",
            torch.linspace(-1.0, 0.0, config.num_value_bins, dtype=torch.float32),
            persistent=True,
        )

        if config.pretrained_path:
            self._load_pretrained_vlm_weights(config.pretrained_path)

    def _load_pretrained_vlm_weights(self, pretrained_path: str) -> None:
        """Load VLM weights from a pretrained pi0.5 checkpoint into the PaliGemma backbone."""
        from safetensors.torch import load_file
        from transformers.utils import cached_file

        logging.info(f"Loading pretrained VLM weights from {pretrained_path}")

        resolved_file = cached_file(pretrained_path, "model.safetensors")
        full_state_dict = load_file(resolved_file)

        vlm_state_dict: dict[str, Tensor] = {}
        for key, value in full_state_dict.items():
            if not key.startswith(PI05_VLM_KEY_PREFIX):
                continue
            new_key = key[len("paligemma_with_expert."):]
            vlm_state_dict[new_key] = value

        lm_head_key = "paligemma.lm_head.weight"
        embed_key = "paligemma.model.language_model.embed_tokens.weight"
        if lm_head_key in vlm_state_dict and embed_key not in vlm_state_dict:
            vlm_state_dict[embed_key] = vlm_state_dict[lm_head_key].clone()

        missing, unexpected = self.load_state_dict(vlm_state_dict, strict=False)

        expected_missing = [
            k
            for k in missing
            if k.startswith(("state_proj.", "fusion_head.", "value_head.", "value_bin_support"))
        ]
        truly_missing = [k for k in missing if k not in expected_missing]

        loaded_count = len(vlm_state_dict) - len(unexpected)
        logging.info(
            f"Pretrained VLM weights: loaded {loaded_count} tensors, "
            f"{len(expected_missing)} expected-missing (state/fusion/value heads), "
            f"{len(truly_missing)} unexpectedly missing, "
            f"{len(unexpected)} unexpected."
        )
        if truly_missing:
            logging.warning(f"Unexpectedly missing keys: {truly_missing[:10]}")
        if unexpected:
            logging.warning(f"Unexpected keys (not loaded): {unexpected[:10]}")

    def _get_paligemma_backbone(self):
        if hasattr(self.paligemma, "model"):
            return self.paligemma.model
        return self.paligemma

    def _get_language_model(self):
        backbone = self._get_paligemma_backbone()
        if hasattr(backbone, "language_model"):
            return backbone.language_model
        if hasattr(self.paligemma, "language_model"):
            return self.paligemma.language_model
        raise AttributeError("Unable to locate PaliGemma language model module.")

    def _get_vision_tower(self):
        backbone = self._get_paligemma_backbone()
        if hasattr(backbone, "vision_tower"):
            return backbone.vision_tower
        if hasattr(self.paligemma, "vision_tower"):
            return self.paligemma.vision_tower
        raise AttributeError("Unable to locate PaliGemma vision tower module.")

    def _extract_image_embeddings(self, image_features_output) -> Tensor:
        """Normalize image-feature outputs across transformers versions."""
        if isinstance(image_features_output, Tensor):
            return image_features_output

        projected = getattr(image_features_output, "pooler_output", None)
        if isinstance(projected, Tensor):
            return projected

        fallback = getattr(image_features_output, "last_hidden_state", None)
        if isinstance(fallback, Tensor):
            return fallback

        raise TypeError(
            "Unsupported image feature output type from PaliGemma get_image_features: "
            f"{type(image_features_output)}"
        )

    def _prepare_attention_masks_4d(self, att_2d_masks: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        result = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        if dtype is not None:
            result = result.to(dtype=dtype)
        return result

    def _build_prefix_embeddings(
        self,
        images: Tensor,
        state: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, int, int]:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape [B, N_cam, 3, H, W] or [B, 3, H, W], got {tuple(images.shape)}")

        batch_size, n_cams = images.shape[:2]
        flat_images = images.reshape(batch_size * n_cams, *images.shape[2:])
        image_features_output = self._get_paligemma_backbone().get_image_features(
            flat_images,
            return_dict=True,
        )
        flat_img_emb = self._extract_image_embeddings(image_features_output)
        if flat_img_emb.ndim == 2:
            flat_img_emb = flat_img_emb.unsqueeze(1)
        if flat_img_emb.ndim != 3:
            raise ValueError(
                "Expected image embeddings with shape [B*N_cam, T_img, D] "
                f"or [B*N_cam, D], got {tuple(flat_img_emb.shape)}"
            )

        img_token_len = flat_img_emb.shape[1]
        img_emb = flat_img_emb.reshape(batch_size, n_cams * img_token_len, flat_img_emb.shape[-1])
        image_token_len = img_emb.shape[1]
        image_mask = torch.ones(
            batch_size,
            image_token_len,
            dtype=torch.bool,
            device=img_emb.device,
        )

        lang_emb = self.paligemma.get_input_embeddings()(input_ids)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        text_mask = attention_mask.bool()

        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3:
            raise ValueError(
                f"Expected state with shape [B, D_state] or [B, N_state, D_state], got {tuple(state.shape)}"
            )
        if state.shape[0] != batch_size:
            raise ValueError(
                f"State batch size {state.shape[0]} must match image batch size {batch_size}"
            )
        if state.shape[-1] != self.config.max_state_dim:
            raise ValueError(
                f"Expected state dim {self.config.max_state_dim}, got {state.shape[-1]}"
            )

        state = state.to(device=img_emb.device, dtype=self.state_proj.weight.dtype)
        state_emb = self.state_proj(state).to(dtype=lang_emb.dtype)
        state_token_len = state_emb.shape[1]
        state_mask = torch.ones(
            batch_size,
            state_token_len,
            dtype=torch.bool,
            device=img_emb.device,
        )

        full_embs = torch.cat((img_emb, lang_emb, state_emb), dim=1)
        full_mask = torch.cat((image_mask, text_mask, state_mask), dim=1)
        return full_embs, full_mask, image_token_len, state_token_len

    def forward(self, images: Tensor, state: Tensor, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        """
        Args:
            images: [B, N_cam, 3, H, W] or [B, 3, H, W]
            state: [B, D_state] or [B, N_state, D_state]
            input_ids: [B, T]
            attention_mask: [B, T]
        Returns:
            Dictionary with value logits, probabilities and expected value.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids with shape [B, T], got {tuple(input_ids.shape)}")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

        input_ids = input_ids.long()
        attention_mask = attention_mask.bool()

        prefix_embs, prefix_pad_mask, _image_token_len, state_token_len = self._build_prefix_embeddings(
            images=images,
            state=state,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        att_2d_masks = prefix_pad_mask[:, None, :] & prefix_pad_mask[:, :, None]
        att_4d_masks = self._prepare_attention_masks_4d(att_2d_masks, dtype=prefix_embs.dtype)
        position_ids = torch.cumsum(prefix_pad_mask, dim=1) - 1

        prefix_output = self._get_language_model().forward(
            inputs_embeds=prefix_embs,
            attention_mask=att_4d_masks,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state

        if state_token_len <= 0:
            raise ValueError(f"Expected at least one state token, got {state_token_len}")
        state_hidden = prefix_output[:, -state_token_len:, :]
        value_feat = state_hidden[:, -1, :]

        fusion_dtype = self.fusion_head[0].weight.dtype
        value_head_dtype = self.value_head.weight.dtype
        fused = self.fusion_head(value_feat.to(dtype=fusion_dtype))
        value_logits = self.value_head(fused.to(dtype=value_head_dtype))

        value_probs = torch.softmax(value_logits.float(), dim=-1)
        expected_value = (value_probs * self.value_bin_support).sum(dim=-1, keepdim=True)

        return {
            "value_logits": value_logits,
            "value_probs": value_probs,
            "expected_value": expected_value,
        }


if __name__ == "__main__":
    model_config = RECAPValueNetworkConfig()
    model = RECAPValueNetwork(model_config)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
