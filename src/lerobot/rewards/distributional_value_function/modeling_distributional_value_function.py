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

Implements the distributional value function V^{pi_ref}(o_t, l) from Section IV-A.
Architecture: the paper uses a 670M-parameter Gemma 3 VLM (Figure 3) —
SigLIP2-so400m (27 layers, 1152-dim) + Gemma3-270M (18 layers, 640-dim),
with a [CLS] token readout predicting a categorical distribution over
B=201 discrete value bins in [-1, 0]. This implementation uses a 2-layer
MLP value head (Linear→LN→GELU→Dropout→Linear) inspired by Robometer
(Chen et al., 2025).
"""

from __future__ import annotations

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
    OPENPI_ATTENTION_MASK_VALUE as _ATTENTION_MASK_VALUE,
)
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_distributional_value_function import DistributionalVFConfig
from .processor_distributional_value_function import IMAGE_MASK_SUFFIX

if TYPE_CHECKING or _transformers_available:
    from transformers import Gemma3ForCausalLM, SiglipVisionModel
else:
    Gemma3ForCausalLM = None  # type: ignore[assignment]
    SiglipVisionModel = None  # type: ignore[assignment]


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
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
    return att_2d_masks & pad_2d_masks


class ValueHead(nn.Module):
    """Categorical value projection: hidden state → bin logits.

    2-layer MLP: Linear → LayerNorm → GELU → Dropout → Linear.
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

    Architecture: monolithic VLM — SigLIP2-so400m + Gemma3-270M (~670M params).
    Multi-camera images are encoded by SigLIP2 (256 patches each), projected to
    Gemma3's hidden dim, concatenated with tokenized language, and processed by
    all 18 Gemma3 transformer layers. A [CLS] token appended at the end provides
    the value readout via a 2-layer MLP head.
    """

    name = "distributional_value_function"
    config_class = DistributionalVFConfig

    def __init__(self, config: DistributionalVFConfig, **kwargs) -> None:
        require_package("transformers", extra="recap")
        super().__init__(config)
        self.config = config

        self.vision_encoder = SiglipVisionModel.from_pretrained(config.siglip_path)
        siglip_hidden = self.vision_encoder.config.hidden_size  # 1152

        self.gemma3 = Gemma3ForCausalLM.from_pretrained(config.gemma3_path)
        self.gemma3_hidden = self.gemma3.config.hidden_size  # 640

        # Fresh image projection: SigLIP2 1152-dim → Gemma3 640-dim
        self.image_proj = nn.Linear(siglip_hidden, self.gemma3_hidden, bias=True)
        nn.init.normal_(self.image_proj.weight, std=0.02)
        nn.init.zeros_(self.image_proj.bias)

        # Learnable [CLS] token — appended to the sequence before Gemma3.
        # nn.Embedding (not nn.Parameter) for FSDP compatibility.
        self.cls_embedding = nn.Embedding(1, self.gemma3_hidden)
        nn.init.normal_(self.cls_embedding.weight, std=0.02)

        # Value head: MLP projection → num_bins logits
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

    def _set_requires_grad(self) -> None:
        if self.config.freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()

        if self.config.freeze_language_model:
            for param in self.gemma3.parameters():
                param.requires_grad = False
            self.gemma3.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.vision_encoder.eval()
        if self.config.freeze_language_model:
            self.gemma3.eval()
        return self

    def embed_image(self, image: Tensor) -> Tensor:
        """Embed images: SigLIP2 → projection → [B, num_patches, gemma3_hidden].

        Args:
            image: [batch_size, channels, height, width] preprocessed image in [-1, 1].

        Returns:
            [B, 256, gemma3_hidden] projected image features.
        """
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        feats = self.vision_encoder(pixel_values=image).last_hidden_state
        return self.image_proj(feats)

    def embed_text(self, token_ids: Tensor) -> Tensor:
        """Embed text using Gemma3's embedding table (includes sqrt(d) scaling).

        Args:
            token_ids: [B, seq_len] integer token IDs.

        Returns:
            [B, seq_len, gemma3_hidden] text embeddings.
        """
        return self.gemma3.model.embed_tokens(token_ids)

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        text_embeddings: Tensor,
        text_padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build prefix: [img1_patches, img2_patches, ..., lang_tokens].

        All prefix tokens use bidirectional attention (att_mask=0).

        Returns:
            embs: [B, total_prefix_len, hidden_dim]
            pad_masks: [B, total_prefix_len] boolean
        """
        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.embed_image(img)
            bsize, num_patches = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_patches))

        embs.append(text_embeddings)
        pad_masks.append(text_padding_mask)

        return torch.cat(embs, dim=1), torch.cat(pad_masks, dim=1)

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

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        """Training forward pass — computes cross-entropy loss against MC return targets.

        The batch is expected to be preprocessed by the processor pipeline.
        Keys expected in batch:
            - observation.images.*: [B, C, H, W] preprocessed images
            - observation.language_tokens: [B, seq_len] tokenized task prompt
            - observation.language_attention_mask: [B, seq_len] padding mask
            - mc_return: [B] normalized Monte Carlo return targets in (-1, 0)
            - is_terminal: [B] boolean terminal flags

        Returns:
            (loss, output_dict) where loss is scalar cross-entropy
        """
        images, img_masks, token_ids, text_pad_mask = self._get_model_inputs(batch)
        mc_return = batch["mc_return"]
        is_terminal = batch["is_terminal"]

        text_embs = self.embed_text(token_ids)
        prefix_embs, prefix_pad_masks = self.embed_prefix(images, img_masks, text_embs, text_pad_mask)

        # VLM forward: prefix + [CLS] through Gemma3, then value head
        batch_size = prefix_embs.shape[0]
        device = prefix_embs.device

        if self.config.stop_gradient_to_vlm:
            prefix_embs = prefix_embs.detach()

        cls_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        cls_emb = self.cls_embedding(cls_ids)
        hidden_states = torch.cat([prefix_embs, cls_emb], dim=1)

        cls_pad = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        pad_masks = torch.cat([prefix_pad_masks, cls_pad], dim=1)

        prefix_att = torch.zeros(batch_size, prefix_embs.shape[1], dtype=torch.long, device=device)
        cls_att = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        att_masks = torch.cat([prefix_att, cls_att], dim=1)

        att_2d = make_att_2d_masks(pad_masks, att_masks)
        model_dtype = next(self.gemma3.parameters()).dtype
        att_4d = torch.where(
            att_2d[:, None, :, :],
            torch.tensor(0.0, dtype=model_dtype, device=device),
            torch.tensor(_ATTENTION_MASK_VALUE, dtype=model_dtype, device=device),
        )

        position_ids = torch.cumsum(pad_masks.long(), dim=1) - 1

        if hidden_states.dtype != model_dtype:
            hidden_states = hidden_states.to(model_dtype)

        outputs = self.gemma3.model(
            inputs_embeds=hidden_states,
            attention_mask=att_4d,
            position_ids=position_ids,
        )
        cls_hidden_state = outputs.last_hidden_state[:, -1, :]

        value_logits = self.value_head(cls_hidden_state)
        value_probs = F.softmax(value_logits, dim=-1)
        predicted_value = (value_probs * self.value_head.bin_centers.to(dtype=value_probs.dtype)).sum(
            dim=-1, keepdim=True
        )

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
        images, img_masks, token_ids, text_pad_mask = self._get_model_inputs(batch)
        text_embs = self.embed_text(token_ids)
        prefix_embs, prefix_pad_masks = self.embed_prefix(images, img_masks, text_embs, text_pad_mask)

        # VLM forward: prefix + [CLS] through Gemma3, then value head
        batch_size = prefix_embs.shape[0]
        device = prefix_embs.device

        if self.config.stop_gradient_to_vlm:
            prefix_embs = prefix_embs.detach()

        cls_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        cls_emb = self.cls_embedding(cls_ids)
        hidden_states = torch.cat([prefix_embs, cls_emb], dim=1)

        cls_pad = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        pad_masks = torch.cat([prefix_pad_masks, cls_pad], dim=1)

        prefix_att = torch.zeros(batch_size, prefix_embs.shape[1], dtype=torch.long, device=device)
        cls_att = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        att_masks = torch.cat([prefix_att, cls_att], dim=1)

        att_2d = make_att_2d_masks(pad_masks, att_masks)
        model_dtype = next(self.gemma3.parameters()).dtype
        att_4d = torch.where(
            att_2d[:, None, :, :],
            torch.tensor(0.0, dtype=model_dtype, device=device),
            torch.tensor(_ATTENTION_MASK_VALUE, dtype=model_dtype, device=device),
        )

        position_ids = torch.cumsum(pad_masks.long(), dim=1) - 1

        if hidden_states.dtype != model_dtype:
            hidden_states = hidden_states.to(model_dtype)

        outputs = self.gemma3.model(
            inputs_embeds=hidden_states,
            attention_mask=att_4d,
            position_ids=position_ids,
        )
        cls_hidden_state = outputs.last_hidden_state[:, -1, :]

        value_logits = self.value_head(cls_hidden_state)
        value_probs = F.softmax(value_logits, dim=-1)
        predicted_value = (value_probs * self.value_head.bin_centers.to(dtype=value_probs.dtype)).sum(
            dim=-1, keepdim=True
        )

        return predicted_value.squeeze(-1)
