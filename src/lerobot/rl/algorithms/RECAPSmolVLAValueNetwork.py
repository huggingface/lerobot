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
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoConfig, AutoModelForImageTextToText, SmolVLMForConditionalGeneration
else:
    AutoConfig = None
    AutoModelForImageTextToText = None
    SmolVLMForConditionalGeneration = None


@dataclass
class RECAPSmolVLAValueNetworkConfig:
    """Configuration for the standalone RECAP SmolVLA value network."""

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False
    precision: Literal["bfloat16", "float32"] = "float32"
    image_size: int = 512
    max_state_dim: int = 32
    freeze_vision_encoder: bool = False
    freeze_backbone: bool = False
    num_vlm_layers: int = 16
    num_value_bins: int = 201
    dropout: float = 0.1


class RECAPSmolVLAValueNetwork(nn.Module):
    """
    Standalone distributional value network for RECAP using a SmolVLM backbone.

    The network predicts a categorical value distribution over bins in [-1, 0]
    and recovers a scalar expected value from this distribution.
    """

    value_bin_support: Tensor

    def __init__(self, config: RECAPSmolVLAValueNetworkConfig):
        super().__init__()
        if (
            AutoConfig is None
            or AutoModelForImageTextToText is None
            or SmolVLMForConditionalGeneration is None
        ):
            raise ImportError("transformers is required to instantiate RECAPSmolVLAValueNetwork.")

        self.config = config
        model_dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float32

        if config.load_vlm_weights:
            logging.info(f"Loading SmolVLM weights from {config.vlm_model_name}")
            self.smolvlm = AutoModelForImageTextToText.from_pretrained(
                config.vlm_model_name,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True,
            )
        else:
            logging.info(f"Building SmolVLM from config {config.vlm_model_name}")
            base_config = AutoConfig.from_pretrained(config.vlm_model_name)
            self.smolvlm = SmolVLMForConditionalGeneration(config=base_config)
            self.smolvlm = self.smolvlm.to(dtype=model_dtype)  # ty: ignore[missing-argument]

        text_model = self._get_text_model()
        if config.num_vlm_layers > 0:
            total_layers = len(text_model.layers)
            if config.num_vlm_layers > total_layers:
                raise ValueError(
                    f"num_vlm_layers={config.num_vlm_layers} exceeds model depth {total_layers}"
                )
            text_model.layers = text_model.layers[: config.num_vlm_layers]
            logging.info(f"Using first {len(text_model.layers)} SmolVLM text layers for value network")

        hidden_size = int(self.smolvlm.config.text_config.hidden_size)
        self.state_proj = nn.Linear(config.max_state_dim, hidden_size)

        if config.freeze_backbone:
            self.smolvlm.eval()
            for param in self.smolvlm.parameters():
                param.requires_grad = False
        elif config.freeze_vision_encoder:
            vision_model = self._get_vision_model()
            vision_model.eval()
            for param in vision_model.parameters():
                param.requires_grad = False

        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(hidden_size, config.num_value_bins)

        self.register_buffer(
            "value_bin_support",
            torch.linspace(-1.0, 0.0, config.num_value_bins, dtype=torch.float32),
            persistent=True,
        )

    def _get_vlm_backbone(self):
        if hasattr(self.smolvlm, "model"):
            return self.smolvlm.model
        return self.smolvlm

    def _get_text_model(self):
        backbone = self._get_vlm_backbone()
        if hasattr(backbone, "text_model"):
            return backbone.text_model
        raise AttributeError("Unable to locate SmolVLM text model module.")

    def _get_vision_model(self):
        backbone = self._get_vlm_backbone()
        if hasattr(backbone, "vision_model"):
            return backbone.vision_model
        raise AttributeError("Unable to locate SmolVLM vision model module.")

    def _extract_last_hidden_state(self, output) -> Tensor:
        if isinstance(output, Tensor):
            return output
        hidden_state = getattr(output, "last_hidden_state", None)
        if isinstance(hidden_state, Tensor):
            return hidden_state
        raise TypeError(f"Unsupported output type while extracting hidden states: {type(output)}")

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
            raise ValueError(
                f"Expected images with shape [B, N_cam, 3, H, W] or [B, 3, H, W], got {tuple(images.shape)}"
            )

        batch_size, n_cams = images.shape[:2]
        image_height, image_width = images.shape[-2], images.shape[-1]

        connector = self._get_vlm_backbone().connector
        scale_factor = int(getattr(connector, "scale_factor", 1))
        vision_config = self._get_vision_model().config
        patch_size = int(getattr(vision_config, "patch_size", 1))
        required_multiple = max(1, scale_factor * patch_size)
        if image_height % required_multiple != 0 or image_width % required_multiple != 0:
            raise ValueError(
                "SmolVLM image sizes must be divisible by "
                f"{required_multiple} (patch_size={patch_size}, connector_scale_factor={scale_factor}). "
                f"Got HxW={image_height}x{image_width}."
            )

        flat_images = images.reshape(batch_size * n_cams, *images.shape[2:])
        flat_images = flat_images.to(dtype=torch.float32)
        # SmolVLM uses SigLIP-like image normalization in [-1, 1].
        flat_images = flat_images.mul(2.0).sub(1.0)

        vision_dtype = next(self._get_vision_model().parameters()).dtype
        image_hidden_states = self._extract_last_hidden_state(
            self._get_vision_model()(
                pixel_values=flat_images.to(dtype=vision_dtype),
                patch_attention_mask=None,
            )
        )
        projected = self._get_vlm_backbone().connector(image_hidden_states)

        if projected.ndim == 2:
            projected = projected.unsqueeze(1)
        if projected.ndim != 3:
            raise ValueError(
                "Expected image embeddings with shape [B*N_cam, T_img, D] "
                f"or [B*N_cam, D], got {tuple(projected.shape)}"
            )

        img_token_len = projected.shape[1]
        img_emb = projected.reshape(batch_size, n_cams * img_token_len, projected.shape[-1])
        image_token_len = img_emb.shape[1]
        image_mask = torch.ones(
            batch_size,
            image_token_len,
            dtype=torch.bool,
            device=img_emb.device,
        )

        lang_emb = self.smolvlm.get_input_embeddings()(input_ids)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        text_mask = attention_mask.bool()

        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3:
            raise ValueError(
                f"Expected state with shape [B, D_state] or [B, N_state, D_state], got {tuple(state.shape)}"
            )
        if state.shape[0] != batch_size:
            raise ValueError(f"State batch size {state.shape[0]} must match image batch size {batch_size}")
        if state.shape[-1] != self.config.max_state_dim:
            raise ValueError(f"Expected state dim {self.config.max_state_dim}, got {state.shape[-1]}")

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
        position_ids = torch.cumsum(prefix_pad_mask, dim=1) - 1
        position_ids = position_ids.masked_fill(~prefix_pad_mask, 0).long()

        text_model = self._get_text_model()
        text_dtype = next(text_model.parameters()).dtype
        text_model_inputs = {
            "inputs_embeds": prefix_embs.to(dtype=text_dtype),
            "attention_mask": prefix_pad_mask,
            "use_cache": False,
        }
        try:
            text_model_inputs["position_ids"] = position_ids
            prefix_output = text_model.forward(**text_model_inputs).last_hidden_state
        except TypeError:
            text_model_inputs.pop("position_ids", None)
            prefix_output = text_model.forward(**text_model_inputs).last_hidden_state

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
    model_config = RECAPSmolVLAValueNetworkConfig()
    model = RECAPSmolVLAValueNetwork(model_config)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
