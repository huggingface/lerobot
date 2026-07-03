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
Architecture: the paper uses a 670M-parameter Gemma 3 VLM (the actor is 4B Gemma 3).
We match that scale on PaliGemma (PI05's Gemma 2B backbone) by truncating to 6 Gemma
LM layers and 13 SigLIP vision layers (~670M params), with a [CLS] token and linear
head predicting a categorical distribution over B=201 discrete value bins in [-1, 0].

Inputs: single image observation + task text prompt ("Task: {task}.")
Outputs: softmax distribution over value bins; expected value E[V] for inference.
Training: cross-entropy on HL-Gauss soft targets (or Dirac delta projection),
with optional one-hot targets for terminal states; MC returns normalized per task.

Weight initialization: vision tower, multi-modal projector, token embeddings, and
the first N transformer layers are copied from a pre-trained PI05 actor checkpoint.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_distributional_value_function import DistributionalVFConfig

if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from lerobot.policies.pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaRMSNorm,
        _gated_residual,
        _get_pi_gemma_decoder_layer_base,
    )
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
    PiGemmaRMSNorm = None
    _gated_residual = None
    _get_pi_gemma_decoder_layer_base = None

PALIGEMMA_VOCAB_SIZE = 257152


class DistributionalVFRewardModel(PreTrainedRewardModel):
    """Distributional value function model for RECAP.

    Predicts V^{pi_ref}(o_t, l) as a categorical distribution over B bins (default 201).
    Trained with cross-entropy on HL-Gauss or Dirac delta targets centered on
    per-task normalized Monte Carlo returns.

    Architecture: truncated PaliGemma (``num_hidden_layers=6``, ``num_vision_layers=13``),
    causal attention, [CLS] token, and Linear(D, num_bins) value head.
    The expected value is E[V] = sum(softmax(logits) * bin_centers).
    """

    name = "distributional_value_function"
    config_class = DistributionalVFConfig

    def __init__(self, config: DistributionalVFConfig, **kwargs) -> None:
        require_package("transformers", extra="recap")
        super().__init__(config)
        self.config = config

        from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

        from lerobot.policies.pi05.modeling_pi05 import get_gemma_config

        # Get base dimensions from the paligemma variant (OpenPI config format)
        base_config = get_gemma_config(config.paligemma_variant)
        hidden_dim = base_config.width
        mlp_dim = base_config.mlp_dim
        num_layers = config.num_hidden_layers

        # HuggingFace GemmaConfig for transformer layers
        gemma_config = CONFIG_MAPPING["gemma"](
            head_dim=base_config.head_dim,
            hidden_size=hidden_dim,
            intermediate_size=mlp_dim,
            num_attention_heads=base_config.num_heads,
            num_hidden_layers=num_layers,
            num_key_value_heads=base_config.num_kv_heads,
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            hidden_activation="gelu_pytorch_tanh",
        )
        self.gemma_config = gemma_config
        self.hidden_dim = hidden_dim
        self.num_value_bins = config.num_value_bins

        # Single learned [CLS] token for value prediction
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Value projection head: Linear(hidden_dim, num_bins)
        self.value_head = nn.Linear(in_features=hidden_dim, out_features=config.num_value_bins)

        # Transformer layers (overwritten by _initialize_from_actor on first run)
        self.rotary_emb = GemmaRotaryEmbedding(gemma_config)
        pi_gemma_decoder_layer_base = _get_pi_gemma_decoder_layer_base()
        self.layers = nn.ModuleList(
            [pi_gemma_decoder_layer_base(gemma_config, layer_idx=i) for i in range(num_layers)]
        )
        self.norm = PiGemmaRMSNorm(hidden_dim, eps=gemma_config.rms_norm_eps)

        # Vision tower + projector + token embedding (overwritten by _initialize_from_actor on first run)
        # PaliGemmaConfig wraps both vision and text configs into a single model
        paligemma_config = CONFIG_MAPPING["paligemma"]()
        paligemma_config.text_config = gemma_config
        paligemma_config.vision_config.image_size = config.image_resolution[0]
        paligemma_config.vision_config.intermediate_size = 4304
        paligemma_config.vision_config.projection_dim = 2048
        paligemma_config.vision_config.projector_hidden_act = "gelu_fast"

        paligemma_full = PaliGemmaForConditionalGenerationWithPiGemma(config=paligemma_config)
        self.vision_tower = paligemma_full.model.vision_tower
        self.multi_modal_projector = paligemma_full.model.multi_modal_projector
        self.token_embedding = paligemma_full.model.language_model.embed_tokens
        del paligemma_full

        # Truncate vision tower to num_vision_layers
        if hasattr(self.vision_tower, "vision_model") and hasattr(self.vision_tower.vision_model, "encoder"):
            vision_encoder = self.vision_tower.vision_model.encoder
            vision_encoder.layers = vision_encoder.layers[: config.num_vision_layers]

        # Bin support: evenly spaced centers from value_support_min to value_support_max
        bin_centers = torch.linspace(config.value_support_min, config.value_support_max, self.num_value_bins)
        self.register_buffer("bin_centers", bin_centers, persistent=False)
        bin_width = (config.value_support_max - config.value_support_min) / (self.num_value_bins - 1)
        self.hl_gauss_sigma = float(config.hl_gauss_sigma_ratio * bin_width)

        # Overwrite with pre-trained PI05 actor weights (first training run only)
        if config.init_from_actor_path:
            self._initialize_from_actor()

    def _initialize_from_actor(self) -> None:
        """Overwrite weights from a pre-trained PI05 actor checkpoint.

        Called on first training run only (when init_from_actor_path is set).
        """
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        actor_policy = PI05Policy.from_pretrained(self.config.init_from_actor_path)
        actor_model = actor_policy.model

        paligemma_model = actor_model.paligemma_with_expert.paligemma
        source_language_model = paligemma_model.model.language_model

        # Transformer components
        self.rotary_emb.load_state_dict(source_language_model.rotary_emb.state_dict())
        num_layers = self.gemma_config.num_hidden_layers
        for i in range(num_layers):
            self.layers[i].load_state_dict(source_language_model.layers[i].state_dict())
        self.norm.load_state_dict(source_language_model.norm.state_dict())

        # Vision tower (truncate source first, then copy)
        source_vision_tower = paligemma_model.model.vision_tower
        if hasattr(source_vision_tower, "vision_model") and hasattr(
            source_vision_tower.vision_model, "encoder"
        ):
            source_encoder = source_vision_tower.vision_model.encoder
            source_encoder.layers = source_encoder.layers[: self.config.num_vision_layers]
        self.vision_tower.load_state_dict(source_vision_tower.state_dict())

        # Multi-modal projector
        self.multi_modal_projector.load_state_dict(paligemma_model.model.multi_modal_projector.state_dict())

        # Token embedding table
        self.token_embedding.load_state_dict(paligemma_model.model.language_model.embed_tokens.state_dict())

        del actor_policy

    def embed_image(self, image: Tensor) -> Tensor:
        """Embed images using the value function's SigLIP vision tower.

        Args:
            image: [batch_size, channels, height, width] preprocessed images in [-1, 1].

        Returns:
            [batch_size, num_patches, hidden_dim] projected image features.
        """
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)

        image_outputs = self.vision_tower(image, return_dict=True)
        image_features = self.multi_modal_projector(image_outputs.last_hidden_state)
        image_features = image_features / (self.hidden_dim**0.5)

        if image_features.dtype != out_dtype:
            image_features = image_features.to(out_dtype)
        return image_features

    def embed_text(self, token_ids: Tensor) -> Tensor:
        """Embed text token IDs using the value function's token embedding table.

        Args:
            token_ids: [batch_size, seq_len] integer token IDs

        Returns:
            [batch_size, seq_len, hidden_dim] text embeddings
        """
        return self.token_embedding(token_ids)

    def _get_cls_embedding(self, batch_size: int) -> Tensor:
        """Get [CLS] token embedding expanded to batch size.

        Args:
            batch_size: number of samples in the batch.

        Returns:
            [batch_size, 1, hidden_dim] learned [CLS] embedding.
        """
        return self.cls_embedding.expand(batch_size, -1, -1)

    def forward_value(
        self, vision_features: Tensor, text_embeddings: Tensor, text_padding_mask: Tensor
    ) -> dict[str, Tensor]:
        """Core forward pass through the distributional value function.

        Args:
            vision_features: [batch_size, num_patches, hidden_dim]
            text_embeddings: [batch_size, seq_len, hidden_dim]
            text_padding_mask: [batch_size, seq_len] boolean mask for text tokens

        Returns:
            logits: [batch_size, num_value_bins]
            probs: [batch_size, num_value_bins]
            value: [batch_size, 1]
        """
        from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE

        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device

        # Build sequence: [vision, text, CLS]
        cls_embedding = self._get_cls_embedding(batch_size)
        hidden_states = torch.cat([vision_features, text_embeddings, cls_embedding], dim=1)

        # Build causal attention mask
        vision_len = vision_features.shape[1]
        vision_padding_mask = torch.ones(batch_size, vision_len, dtype=torch.bool, device=device)
        cls_padding_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        full_padding_mask = torch.cat([vision_padding_mask, text_padding_mask, cls_padding_mask], dim=1)

        full_seq_len = full_padding_mask.shape[1]

        # Causal mask
        causal_mask = torch.tril(torch.ones(full_seq_len, full_seq_len, device=device, dtype=torch.bool))
        # Combine causal mask with padding mask
        padding_mask_4d = full_padding_mask[:, None, None, :].expand(
            batch_size, 1, full_seq_len, full_seq_len
        )
        attention_mask = causal_mask[None, None, :, :] & padding_mask_4d
        attention_mask = torch.where(attention_mask, 0.0, OPENPI_ATTENTION_MASK_VALUE)

        position_ids = torch.cumsum(full_padding_mask.long(), dim=1) - 1
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            norm_output = layer.input_layernorm(hidden_states, cond=None)
            if isinstance(norm_output, tuple):
                hidden_states_normed, gate = norm_output
            else:
                hidden_states_normed, gate = norm_output, None

            input_shape = hidden_states_normed.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            query_states = layer.self_attn.q_proj(hidden_states_normed).view(hidden_shape).transpose(1, 2)
            key_states = layer.self_attn.k_proj(hidden_states_normed).view(hidden_shape).transpose(1, 2)
            value_states = layer.self_attn.v_proj(hidden_states_normed).view(hidden_shape).transpose(1, 2)

            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            attention_output, _ = modeling_gemma.eager_attention_forward(
                layer.self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                layer.self_attn.scaling,
            )

            attention_output = attention_output.reshape(batch_size, -1, self.gemma_config.hidden_size)
            if attention_output.dtype != layer.self_attn.o_proj.weight.dtype:
                attention_output = attention_output.to(layer.self_attn.o_proj.weight.dtype)
            projected_attention = layer.self_attn.o_proj(attention_output)

            if gate is not None:
                projected_attention = _gated_residual(hidden_states, projected_attention, gate)
            else:
                projected_attention = hidden_states + projected_attention

            after_attention_residual = projected_attention.clone()

            norm_output = layer.post_attention_layernorm(projected_attention, cond=None)
            if isinstance(norm_output, tuple):
                mlp_input, gate = norm_output
            else:
                mlp_input, gate = norm_output, None

            mlp_output = layer.mlp(mlp_input)

            if gate is not None:
                hidden_states = _gated_residual(after_attention_residual, mlp_output, gate)
            else:
                hidden_states = after_attention_residual + mlp_output

        hidden_states = self.norm(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        # Extract [CLS] token (last position in the sequence)
        cls_hidden_state = hidden_states[:, -1, :]  # [batch_size, hidden_dim]

        # Value head: Linear(hidden_dim, num_bins) -> logits
        value_logits = self.value_head(cls_hidden_state)  # [batch_size, num_value_bins]
        value_probs = F.softmax(value_logits, dim=-1)
        predicted_value = (value_probs * self.bin_centers.to(dtype=value_probs.dtype)).sum(
            dim=-1, keepdim=True
        )

        return {"logits": value_logits, "probs": value_probs, "value": predicted_value}

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
        target_value = target_value.to(dtype=self.bin_centers.dtype)

        # Bin edges: half a bin-width outside the first/last center
        bin_width = (self.config.value_support_max - self.config.value_support_min) / (
            self.num_value_bins - 1
        )
        support_edges = torch.linspace(
            self.config.value_support_min - bin_width / 2,
            self.config.value_support_max + bin_width / 2,
            self.num_value_bins + 1,
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
        target_value = target_value.to(dtype=self.bin_centers.dtype)

        bin_width = self.bin_centers[1] - self.bin_centers[0]
        normalized_position = (target_value - self.config.value_support_min) / bin_width
        lower_bin_idx = normalized_position.floor().long().clamp(0, self.num_value_bins - 1)
        upper_bin_idx = normalized_position.ceil().long().clamp(0, self.num_value_bins - 1)

        weight_upper = normalized_position - lower_bin_idx.float()
        weight_lower = upper_bin_idx.float() - normalized_position

        same_bin = lower_bin_idx == upper_bin_idx
        weight_upper = torch.where(same_bin, torch.zeros_like(weight_upper), weight_upper)
        weight_lower = torch.where(same_bin, torch.ones_like(weight_lower), weight_lower)

        batch_size = target_value.shape[0]
        target_distribution = torch.zeros(batch_size, self.num_value_bins, device=target_value.device)
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
        target_value = target_value.to(dtype=self.bin_centers.dtype)
        nearest_bin_idx = torch.argmin(
            torch.abs(self.bin_centers.unsqueeze(0) - target_value.unsqueeze(-1)), dim=-1
        )
        return F.one_hot(nearest_bin_idx, num_classes=self.num_value_bins).to(dtype=self.bin_centers.dtype)

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
        from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        # Get first image key from batch
        image_keys = [k for k in batch if k.startswith(f"{OBS_IMAGES}.") or k == OBS_IMAGES]
        if not image_keys:
            raise KeyError(f"No image keys found in batch. Expected keys starting with '{OBS_IMAGES}.'")
        images = batch[image_keys[0]]

        token_ids = batch[OBS_LANGUAGE_TOKENS]
        text_padding_mask = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
        mc_return = batch["mc_return"]
        is_terminal = batch["is_terminal"]

        # Embed observations
        vision_features = self.embed_image(images)
        text_embeddings = self.embed_text(token_ids)

        # Forward through value function transformer
        vf_output = self.forward_value(vision_features, text_embeddings, text_padding_mask)
        value_logits = vf_output["logits"]
        predicted_value = vf_output["value"]

        # Compute target distribution
        target_distribution = self.compute_target_distribution(
            mc_return,
            is_terminal,
            method=self.config.target_method,
            use_one_hot_terminal=self.config.use_one_hot_terminal,
        )

        # Cross-entropy loss (Eq. 1 in pi*0.6 paper)
        log_probs = F.log_softmax(value_logits, dim=-1)
        loss = -(target_distribution * log_probs).sum(dim=-1).mean()

        output_dict = {
            "loss": loss.item(),
            "predicted_value_mean": predicted_value.mean().item(),
            "mc_return_mean": mc_return.mean().item(),
        }

        return loss, output_dict

    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute V(s) for a batch of observations. Used for advantage scoring.

        Args:
            batch: preprocessed batch with images and tokenized text

        Returns:
            [batch_size] tensor of predicted values V(s)
        """
        from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        image_keys = [k for k in batch if k.startswith(f"{OBS_IMAGES}.") or k == OBS_IMAGES]
        if not image_keys:
            raise KeyError(f"No image keys found in batch. Expected keys starting with '{OBS_IMAGES}.'")
        images = batch[image_keys[0]]

        token_ids = batch[OBS_LANGUAGE_TOKENS]
        text_padding_mask = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()

        vision_features = self.embed_image(images)
        text_embeddings = self.embed_text(token_ids)

        vf_output = self.forward_value(vision_features, text_embeddings, text_padding_mask)
        return vf_output["value"].squeeze(-1)  # [batch_size]
