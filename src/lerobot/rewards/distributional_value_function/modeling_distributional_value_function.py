# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Modeling for RECAP's distributional value function.

Paper: "π*0.6: a VLA That Learns From Experience" (Physical Intelligence, 2025)
       https://pi.website/blog/pistar06
Architecture source of truth: "π0.6 Model Card", Section 2 (Model Design)
       https://website.pi-asset.com/pi06star/PI06_model_card.pdf

Implements the distributional value function V^{pi_ref}(o_t, l) from Section IV-A.
It adapts the native Gemma3 multimodal VLM design to π0.6's smaller ~670M scale:
448px SigLIP images are pooled to 256 soft tokens, RMS-normalized, projected
into Gemma3-270M, and processed with bidirectional image / causal text
attention. A final learned value-query token supplies the 201-bin readout.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs.types import FeatureType
from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_distributional_value_function import DistributionalVFConfig
from .processor_distributional_value_function import IMAGE_MASK_SUFFIX

if TYPE_CHECKING or _transformers_available:
    from transformers import (
        Gemma3Config,
        Gemma3ForCausalLM,
        Gemma3ForConditionalGeneration,
        SiglipVisionModel,
    )
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3MultiModalProjector,
        create_causal_mask_mapping,
    )
else:
    Gemma3Config = None  # type: ignore[assignment]
    Gemma3ForCausalLM = None  # type: ignore[assignment]
    Gemma3ForConditionalGeneration = None  # type: ignore[assignment]
    Gemma3MultiModalProjector = None  # type: ignore[assignment]
    SiglipVisionModel = None  # type: ignore[assignment]
    create_causal_mask_mapping = None  # type: ignore[assignment]


class ValueHead(nn.Module):
    """Categorical value projection: hidden state → bin logits.

    The 2-layer MLP topology is adapted from Robometer's prediction head:
    Linear → LayerNorm → GELU → Dropout → Linear. Unlike Robometer's progress
    and success heads, this head predicts RECAP's 201-bin MC-return
    distribution over [-1, 0] from the final value-query representation.

    Also holds the ``bin_centers`` buffer used to compute E[V] = Σ p_i · c_i.
    """

    def __init__(
        self,
        hidden_size: int,
        num_bins: int,
        v_min: float,
        v_max: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bins = num_bins

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_bins),
        )

        self.register_buffer("bin_centers", torch.linspace(v_min, v_max, num_bins), persistent=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project hidden state to value logits. Returns [B, num_bins]."""
        hidden_states = hidden_states.to(self.mlp[0].weight.dtype)
        return self.mlp(hidden_states)


class DistributionalVFRewardModel(PreTrainedRewardModel):
    """Distributional value function model for RECAP.

    Predicts V^{pi_ref}(o_t, l) as a categorical distribution over B bins (default 201).
    Trained with cross-entropy on HL-Gauss or Dirac delta targets centered on
    per-task normalized Monte Carlo returns.

    Architecture: adapted from the native Gemma3 multimodal VLM using a
    448px SigLIP2-so400m vision tower, Gemma3 multimodal projector, and
    Gemma3-270M language backbone. Each camera is represented by 256 soft
    image tokens. Image tokens are bidirectional, text is causal, and a final
    one-way value query supplies the hidden state consumed by the value head.
    """

    name = "distributional_value_function"
    config_class = DistributionalVFConfig

    def __init__(self, config: DistributionalVFConfig, **kwargs) -> None:
        require_package("transformers", extra="recap")
        super().__init__(config)
        self.config = config

        if config.vlm_pretrained_path:
            aligned_vlm = Gemma3ForConditionalGeneration.from_pretrained(config.vlm_pretrained_path)
            self._vlm_config = aligned_vlm.config
            self.vision_encoder = aligned_vlm.model.vision_tower
            self.multi_modal_projector = aligned_vlm.model.multi_modal_projector
            self.language_model = aligned_vlm.model.language_model
        else:
            self.vision_encoder = SiglipVisionModel.from_pretrained(config.siglip_path)
            gemma3 = Gemma3ForCausalLM.from_pretrained(config.gemma3_path)
            self.language_model = gemma3.model

            # Adapt Gemma3's native multimodal connector to the 270M text
            # backbone and π0.6's 448px input layout.
            vision_config = copy.deepcopy(self.vision_encoder.config)
            vision_config.image_size = config.image_resolution[0]
            text_config = copy.deepcopy(gemma3.config)
            self._vlm_config = Gemma3Config(
                vision_config=vision_config,
                text_config=text_config,
                mm_tokens_per_image=config.num_image_tokens,
            )
            self.multi_modal_projector = Gemma3MultiModalProjector(self._vlm_config)
            nn.init.normal_(
                self.multi_modal_projector.mm_input_projection_weight,
                mean=0.0,
                std=self._vlm_config.initializer_range,
            )

        self._validate_vlm_config()
        self.gemma3_hidden = self._vlm_config.text_config.hidden_size  # 640

        # One-way suffix query, analogous to PI05's suffix/action tokens.
        self.value_query = nn.Embedding(1, self.gemma3_hidden)
        nn.init.normal_(self.value_query.weight, std=0.02)

        # Value head: value-query hidden state → MLP → num_bins logits
        self.value_head = ValueHead(
            hidden_size=self.gemma3_hidden,
            num_bins=config.num_value_bins,
            v_min=config.value_support_min,
            v_max=config.value_support_max,
            dropout=config.value_dropout,
        )

        # HL-Gauss sigma for soft targets
        bin_width = (config.value_support_max - config.value_support_min) / (config.num_value_bins - 1)
        self.hl_gauss_sigma = float(config.hl_gauss_sigma_ratio * bin_width)

        # Apply freezing
        self._set_requires_grad()

    @property
    def gemma3(self) -> nn.Module:
        """Backward-compatible access to the Gemma3 text backbone."""
        return self.language_model

    def _validate_vlm_config(self) -> None:
        """Validate the π0.6 448px → 256-token multimodal layout."""
        vision_config = self._vlm_config.vision_config
        image_size = self.config.image_resolution[0]
        if vision_config.image_size != image_size:
            raise ValueError(
                f"VLM vision image_size ({vision_config.image_size}) does not match "
                f"DistributionalVFConfig.image_resolution ({image_size})"
            )

        patches_per_side = image_size // vision_config.patch_size
        tokens_per_side = int(self.config.num_image_tokens**0.5)
        if tokens_per_side**2 != self.config.num_image_tokens:
            raise ValueError("num_image_tokens must be a perfect square")
        if patches_per_side % tokens_per_side:
            raise ValueError(
                f"{patches_per_side} patches/side cannot be evenly pooled to {tokens_per_side} tokens/side"
            )
        if self._vlm_config.mm_tokens_per_image != self.config.num_image_tokens:
            raise ValueError(
                f"VLM emits {self._vlm_config.mm_tokens_per_image} image tokens, "
                f"expected {self.config.num_image_tokens}"
            )

    def _set_requires_grad(self) -> None:
        if self.config.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()

        if self.config.freeze_language_model:
            for param in self.language_model.parameters():
                param.requires_grad = False
            self.language_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.vision_encoder.eval()
        if self.config.freeze_language_model:
            self.language_model.eval()
        return self

    def get_optim_params(self) -> list[dict]:
        """Optimizer param groups with per-component learning rates."""
        vision_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("vision_encoder"):
                vision_params.append(param)
            else:
                other_params.append(param)

        base_lr = self.config.get_optimizer_preset().lr
        return [
            {"params": other_params},
            {"params": vision_params, "lr": base_lr * self.config.vision_encoder_lr_multiplier},
        ]

    def embed_image(self, image: Tensor) -> Tensor:
        """Embed images with π0.6's Gemma3 visual connector.

        Args:
            image: [batch_size, channels, height, width] preprocessed image in [-1, 1].

        Returns:
            [B, 256, gemma3_hidden] pooled, normalized, projected features.
        """
        vision_dtype = next(self.vision_encoder.parameters()).dtype
        image_features = self.vision_encoder(
            pixel_values=image.to(dtype=vision_dtype),
            interpolate_pos_encoding=True,
        ).last_hidden_state
        projected_features = self.multi_modal_projector(image_features)
        if projected_features.shape[1] != self.config.num_image_tokens:
            raise RuntimeError(
                f"Expected {self.config.num_image_tokens} image tokens, got {projected_features.shape[1]}"
            )
        return projected_features

    def embed_text(self, token_ids: Tensor) -> Tensor:
        """Embed text using Gemma3's embedding table (includes sqrt(d) scaling).

        Args:
            token_ids: [B, seq_len] integer token IDs.

        Returns:
            [B, seq_len, gemma3_hidden] text embeddings.
        """
        return self.language_model.embed_tokens(token_ids)

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        text_embeddings: Tensor,
        text_padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build [image soft tokens..., text] plus masks.

        Returns:
            embs: [B, total_prefix_len, hidden_dim]
            pad_masks: [B, total_prefix_len] boolean
            token_type_ids: [B, total_prefix_len], 1=image and 0=text
        """
        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []
        token_types: list[Tensor] = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.embed_image(img)
            bsize, num_image_tokens = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_image_tokens))
            token_types.append(torch.ones(bsize, num_image_tokens, dtype=torch.long, device=img_emb.device))

        embs.append(text_embeddings)
        pad_masks.append(text_padding_mask)
        token_types.append(torch.zeros_like(text_padding_mask, dtype=torch.long))

        return torch.cat(embs, dim=1), torch.cat(pad_masks, dim=1), torch.cat(token_types, dim=1)

    def hl_gauss_target(self, target_value: Tensor) -> Tensor:
        """HL-Gauss soft target distribution.

        Places a Gaussian N(target, sigma^2) over the bin support and computes
        per-bin probabilities as CDF differences at bin edges, normalized to sum to 1.

        Reference: Farebrother et al. 2024, "Stop Regressing: Training Value
        Functions via Classification for Scalable Deep RL", Section 3.1.
        arXiv:2403.03950

        Args:
            target_value: [batch_size] or [batch_size, 1] target values.

        Returns:
            [batch_size, num_value_bins] target probability distribution.
        """
        if target_value.ndim == 2:
            target_value = target_value.squeeze(-1)

        target_value = target_value.to(dtype=self.value_head.bin_centers.dtype)

        # Bin edges: half a bin-width outside the first/last center
        bin_width = (self.config.value_support_max - self.config.value_support_min) / (
            self.config.num_value_bins - 1
        )
        support_edges = torch.linspace(
            self.config.value_support_min - bin_width / 2,
            self.config.value_support_max + bin_width / 2,
            self.config.num_value_bins + 1,
            device=target_value.device,
            dtype=target_value.dtype,
        )

        # CDF of N(target, sigma^2) evaluated at each edge
        cdf_at_edges = 0.5 * (
            1.0
            + torch.erf(
                (support_edges.unsqueeze(0) - target_value.unsqueeze(-1))
                / (self.hl_gauss_sigma * math.sqrt(2))
            )
        )  # [batch_size, num_bins + 1]

        # Normalize: z = cdf(max_edge) - cdf(min_edge)
        normalization_constant = (cdf_at_edges[:, -1] - cdf_at_edges[:, 0]).unsqueeze(-1).clamp(min=1e-10)

        # Bin probabilities = differences of consecutive CDF values, normalized
        bin_probabilities = (cdf_at_edges[:, 1:] - cdf_at_edges[:, :-1]) / normalization_constant

        return bin_probabilities

    def dirac_delta_target(self, target_value: Tensor) -> Tensor:
        """Dirac delta (C51) projection: split probability between two nearest bins.

        Standard distributional RL projection from Bellemare et al. 2017.
        "A Distributional Perspective on Reinforcement Learning"
        arXiv:1707.06887

        Args:
            target_value: [batch_size] or [batch_size, 1] target values.

        Returns:
            [batch_size, num_value_bins] target probability distribution.
        """
        if target_value.ndim == 2:
            target_value = target_value.squeeze(-1)

        target_value = target_value.clamp(self.config.value_support_min, self.config.value_support_max)
        target_value = target_value.to(dtype=self.value_head.bin_centers.dtype)

        bin_width = self.value_head.bin_centers[1] - self.value_head.bin_centers[0]
        normalized_position = (target_value - self.config.value_support_min) / bin_width
        lower_bin_idx = normalized_position.floor().long().clamp(0, self.config.num_value_bins - 1)
        upper_bin_idx = normalized_position.ceil().long().clamp(0, self.config.num_value_bins - 1)

        weight_upper = normalized_position - lower_bin_idx.float()
        weight_lower = upper_bin_idx.float() - normalized_position

        same_bin = lower_bin_idx == upper_bin_idx
        weight_upper = torch.where(same_bin, torch.zeros_like(weight_upper), weight_upper)
        weight_lower = torch.where(same_bin, torch.ones_like(weight_lower), weight_lower)

        batch_size = target_value.shape[0]
        target_distribution = torch.zeros(batch_size, self.config.num_value_bins, device=target_value.device)
        batch_indices = torch.arange(batch_size, device=target_value.device)
        target_distribution[batch_indices, lower_bin_idx] += weight_lower
        target_distribution[batch_indices, upper_bin_idx] += weight_upper

        return target_distribution

    def one_hot_target(self, target_value: Tensor) -> Tensor:
        """One-hot target for terminal states (exact return, no smoothing).

        Args:
            target_value: [batch_size] or [batch_size, 1] target values.

        Returns:
            [batch_size, num_value_bins] one-hot distribution at the nearest bin.
        """
        if target_value.ndim == 2:
            target_value = target_value.squeeze(-1)
        target_value = target_value.to(dtype=self.value_head.bin_centers.dtype)
        nearest_bin_idx = torch.argmin(
            torch.abs(self.value_head.bin_centers.unsqueeze(0) - target_value.unsqueeze(-1)), dim=-1
        )
        return F.one_hot(nearest_bin_idx, num_classes=self.config.num_value_bins).to(
            dtype=self.value_head.bin_centers.dtype
        )

    def compute_target_distribution(
        self,
        target_value: Tensor,
        is_terminal: Tensor,
        method: str = "hl_gauss",
        use_one_hot_terminal: bool = True,
    ) -> Tensor:
        """Compute target distribution using configured method.

        Args:
            target_value: [batch_size] scalar return targets
            is_terminal: [batch_size] boolean terminal flags
            method: "hl_gauss" or "dirac_delta"
            use_one_hot_terminal: if True, terminal states get one-hot targets
                (exact return, no smoothing). If False, all states use the same method.

        Returns:
            [batch_size, num_value_bins] target probability distribution
        """
        if method == "hl_gauss":
            base_distribution = self.hl_gauss_target(target_value)
        elif method == "dirac_delta":
            base_distribution = self.dirac_delta_target(target_value)
        else:
            raise ValueError(f"Unknown target method: {method}. Use 'hl_gauss' or 'dirac_delta'.")

        if not use_one_hot_terminal:
            return base_distribution

        terminal_distribution = self.one_hot_target(target_value)
        return torch.where(is_terminal[:, None].bool(), terminal_distribution, base_distribution)

    def _get_vlm_readout(self, batch: dict[str, Tensor]) -> Tensor:
        """Run Gemma3 image-bidirectional/text-causal attention plus value query."""
        images, img_masks, token_ids, text_pad_mask = self._get_model_inputs(batch)

        text_embs = self.embed_text(token_ids)
        prefix_embs, prefix_pad_masks, prefix_token_types = self.embed_prefix(
            images, img_masks, text_embs, text_pad_mask
        )

        if self.config.stop_gradient_to_vlm:
            prefix_embs = prefix_embs.detach()

        batch_size, prefix_len = prefix_pad_masks.shape
        device = prefix_embs.device
        model_dtype = next(self.language_model.parameters()).dtype

        query_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        query_emb = self.value_query(query_ids)
        hidden_states = torch.cat([prefix_embs, query_emb], dim=1)

        query_pad_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        pad_masks = torch.cat([prefix_pad_masks, query_pad_mask], dim=1)
        token_type_ids = torch.cat(
            [prefix_token_types, torch.zeros(batch_size, 1, dtype=torch.long, device=device)],
            dim=1,
        )

        prefix_position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
        query_position_ids = prefix_pad_masks.sum(dim=1, keepdim=True).long()
        position_ids = torch.cat([prefix_position_ids, query_position_ids], dim=1).clamp_min(0)

        if hidden_states.dtype != model_dtype:
            hidden_states = hidden_states.to(model_dtype)

        attention_masks = create_causal_mask_mapping(
            self._vlm_config,
            inputs_embeds=hidden_states,
            attention_mask=pad_masks,
            past_key_values=None,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            is_training=self.training,
        )
        outputs = self.language_model(
            inputs_embeds=hidden_states,
            attention_mask=attention_masks,
            position_ids=position_ids,
            use_cache=False,
        )
        return outputs.last_hidden_state[:, -1, :]

    def _vlm_forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Run the VLM and value head.

        Returns:
            (value_logits [B, num_bins], predicted_value [B, 1])
        """
        readout = self._get_vlm_readout(batch)
        value_logits = self.value_head(readout)
        value_probs = F.softmax(value_logits, dim=-1)
        predicted_value = (value_probs * self.value_head.bin_centers.to(dtype=value_probs.dtype)).sum(
            dim=-1, keepdim=True
        )
        return value_logits, predicted_value

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        """Training forward pass — cross-entropy loss on MC return targets."""
        mc_return = batch["mc_return"]
        is_terminal = batch["is_terminal"]

        value_logits, predicted_value = self._vlm_forward(batch)

        # Compute target distribution from MC returns
        target_dist = self.compute_target_distribution(
            mc_return,
            is_terminal,
            method=self.config.target_method,
            use_one_hot_terminal=self.config.use_one_hot_terminal,
        )

        # Cross-entropy loss between predicted and target distributions (Eq. 1 in pi*0.6 paper)
        log_probs = F.log_softmax(value_logits, dim=-1)
        loss = -(target_dist * log_probs).sum(dim=-1).mean()

        # Diagnostic metrics
        clamped_return = (
            mc_return.float().view(-1).clamp(self.config.value_support_min, self.config.value_support_max)
        )
        bin_width = self.value_head.bin_centers[1] - self.value_head.bin_centers[0]
        normalized_position = (clamped_return - self.config.value_support_min) / bin_width
        lower_bin_idx = normalized_position.floor().long().clamp(0, self.config.num_value_bins - 1)
        upper_bin_idx = normalized_position.ceil().long().clamp(0, self.config.num_value_bins - 1)

        dist_to_lower = normalized_position - lower_bin_idx.float()
        dist_to_upper = upper_bin_idx.float() - normalized_position
        same_bin = lower_bin_idx == upper_bin_idx
        dist_to_lower = torch.where(same_bin, torch.zeros_like(dist_to_lower), dist_to_lower)
        dist_to_upper = torch.where(same_bin, torch.ones_like(dist_to_upper), dist_to_upper)

        pred_bin = value_logits.argmax(dim=-1)
        best_target_bin = torch.where(dist_to_upper >= dist_to_lower, lower_bin_idx, upper_bin_idx)

        acc_best = (pred_bin == best_target_bin).float().mean().item()
        acc_neighbor = ((pred_bin == lower_bin_idx) | (pred_bin == upper_bin_idx)).float().mean().item()

        min_bin_dist = torch.min((pred_bin - lower_bin_idx).abs(), (pred_bin - upper_bin_idx).abs()).float()
        mae = (min_bin_dist * bin_width).mean().item()

        output_dict: dict[str, Any] = {
            "loss": loss.item(),
            "predicted_value_mean": predicted_value.mean().item(),
            "mc_return_mean": mc_return.mean().item(),
            "acc_best": acc_best,
            "acc_neighbor": acc_neighbor,
            "mae": mae,
        }

        return loss, output_dict

    def _get_model_inputs(
        self, batch: dict[str, Tensor]
    ) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor]:
        """Extract images, masks, token_ids, text_pad_mask from a preprocessed batch."""
        image_keys = [k for k, v in self.config.input_features.items() if v.type == FeatureType.VISUAL]
        images = [batch[k] for k in image_keys]
        img_masks = [batch[k + IMAGE_MASK_SUFFIX].bool() for k in image_keys]
        token_ids = batch[OBS_LANGUAGE_TOKENS]
        text_pad_mask = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
        return images, img_masks, token_ids, text_pad_mask

    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute V(s) for a batch of observations. Used for advantage scoring.

        Args:
            batch: preprocessed batch with images, masks, and tokenized text.

        Returns:
            [batch_size] tensor of predicted values V(s).
        """
        _, predicted_value = self._vlm_forward(batch)
        return predicted_value.squeeze(-1)
