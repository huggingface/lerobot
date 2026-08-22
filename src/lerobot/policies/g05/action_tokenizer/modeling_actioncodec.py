# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional

from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_actioncodec import G05ActionCodecConfig
from .modular_actioncodec import (
    ActionCodecDecoder,
    ActionCodecEncoder,
    ActionTimeContrastiveLoss,
    BlockDCT,
    ResidualVectorQuantizer,
    compute_consistency_loss,
    time_shift_positive,
)

if TYPE_CHECKING or _transformers_available:
    from transformers import PreTrainedModel
else:
    # PreTrainedModel is a base class, so it cannot be bound to None the way a
    # plain symbol can; fail here with the actionable install hint instead.
    require_package("transformers", extra="g05")


class G05ActionCodecModel(PreTrainedModel):
    """Neural action codec for G0.5 policies.

    Components share one codec because they are padded to the same width before
    encoding. The dictionary boundary preserves component semantics while the
    network processes every component together as one larger batch.
    """

    config_class = G05ActionCodecConfig
    base_model_prefix = ""

    def __init__(self, config: G05ActionCodecConfig) -> None:
        super().__init__(config)
        self.block_dct = BlockDCT(config.block_dct_block_size) if config.use_block_dct else None
        self.conv_in = nn.Conv2d(
            config.horizon_patch_size,
            config.encoder_channels,
            kernel_size=(1, config.conv_in_action_kernel),
        )
        self.encoder = ActionCodecEncoder(config)
        self.rvq = ResidualVectorQuantizer(config)
        self.action_time_contrastive_loss: ActionTimeContrastiveLoss | None = None
        if config.consistency_loss_weight > 0 and config.consistency_loss_type == "action_time_contrastive":
            self.action_time_contrastive_loss = ActionTimeContrastiveLoss(
                mode=config.action_time_contrastive_mode,
                temperature_init=config.action_time_contrastive_temperature_init,
                bias_init=config.action_time_contrastive_bias_init,
            )
        self.decoder = ActionCodecDecoder(config)
        self.conv_out = nn.ConvTranspose2d(
            config.encoder_channels,
            config.horizon_patch_size,
            kernel_size=(1, config.conv_in_action_kernel),
        )
        self.post_init()

    def _normalize_components(self, components: dict[str, Tensor]) -> tuple[list[str], Tensor, int]:
        if not components:
            raise ValueError("ActionCodec requires at least one action component")
        names = list(components)
        batch_size = components[names[0]].shape[0]
        normalized = []
        for name in names:
            values = components[name].float()
            if values.ndim != 3 or values.shape[0] != batch_size:
                raise ValueError(f"component {name!r} must have shape [B,T,D] with a shared batch size")
            values = values[:, : self.config.horizon, : self.config.max_component_dim]
            values = functional.pad(
                values,
                (
                    0,
                    self.config.max_component_dim - values.shape[-1],
                    0,
                    self.config.horizon - values.shape[-2],
                ),
            )
            normalized.append(values)
        return names, torch.cat(normalized), batch_size

    def encode(self, components: dict[str, Tensor]) -> dict[str, Tensor]:
        """Encode ``{part: [B,T,D]}`` to ``{part: [B,residual,code]}``."""
        names, values, batch_size = self._normalize_components(components)
        _, codes, _ = self._encode_tensor(values)
        return {
            name: codes[index * batch_size : (index + 1) * batch_size] for index, name in enumerate(names)
        }

    def _encode_tensor(
        self,
        values: Tensor,
        return_level_data: bool = False,
        return_encoder_hidden: bool = False,
    ) -> tuple:
        if self.block_dct is not None:
            values = self.block_dct.dct(values)
        # The temporal patch axis becomes Conv2d channels; the remaining two
        # axes are the coarse time grid and padded component width.
        hidden_states = rearrange(
            values,
            "batch (coarse patch) width -> batch patch coarse width",
            patch=self.config.horizon_patch_size,
        )
        hidden_states = self.encoder(self.conv_in(hidden_states))
        hidden_states = rearrange(
            hidden_states, "batch channels height width -> batch channels (height width)"
        )
        quantized = self.rvq(hidden_states, return_level_data=return_level_data)
        if return_encoder_hidden:
            return (*quantized, hidden_states)
        return quantized

    def _decode_tensor(self, hidden_states: Tensor) -> Tensor:
        hidden_states = rearrange(
            hidden_states,
            "batch channels (height width) -> batch channels height width",
            height=self.config.code_height,
            width=self.config.code_width,
        )
        values = self.conv_out(self.decoder(hidden_states))
        values = rearrange(values, "batch patch coarse width -> batch (coarse patch) width")
        if self.block_dct is not None:
            values = self.block_dct.idct(values, self.config.horizon)
        return values[:, : self.config.horizon]

    def decode(
        self,
        codes: dict[str, Tensor],
        component_dims: dict[str, int] | None = None,
    ) -> dict[str, Tensor]:
        """Decode RVQ codes into time-domain action components."""
        if not codes:
            raise ValueError("ActionCodec requires at least one code component")
        names = list(codes)
        batch_size = codes[names[0]].shape[0]
        packed_codes = torch.cat([codes[name] for name in names])
        values = self._decode_tensor(self.rvq.decode(packed_codes))
        dimensions = component_dims or self.config.parts_meta
        return {
            name: values[
                index * batch_size : (index + 1) * batch_size, :, : dimensions.get(name, values.shape[-1])
            ]
            for index, name in enumerate(names)
        }

    def forward(
        self,
        components: dict[str, Tensor],
        d_original: dict[str, int] | None = None,
        x_pos_dict: dict[str, Tensor] | None = None,
        layer_weights: list[float] | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Run the complete ActionCodec training objective.

        The aggregate loss combines padded time-domain reconstruction, RVQ
        commitment, and the configured consistency regularizer. Public
        ``encode``/``decode`` retain their inference contracts.
        """
        names, values, batch_size = self._normalize_components(components)
        target = values.clone()
        original_dims = d_original or {name: components[name].shape[-1] for name in names}
        packed_batch_size = batch_size * len(names)
        consistency_type = self.config.consistency_loss_type
        use_consistency = self.config.consistency_loss_weight > 0
        if x_pos_dict is None and use_consistency and consistency_type == "action_time_contrastive":
            x_pos_dict = {name: time_shift_positive(components[name]) for name in names}

        if x_pos_dict is None:
            quantized, packed_codes, commitment_loss = self._encode_tensor(values)
            consistency_residuals = None
            level_codes = None
            encoder_hidden = None
        else:
            if set(x_pos_dict) != set(names):
                raise ValueError("x_pos_dict must contain exactly the same keys as components")
            _, positive_values, positive_batch_size = self._normalize_components(
                {name: x_pos_dict[name] for name in names}
            )
            if positive_batch_size != batch_size:
                raise ValueError("x_pos_dict must use the same batch size as components")
            return_level_data = use_consistency and consistency_type == "token_residual"
            return_encoder_hidden = use_consistency and consistency_type == "action_time_contrastive"
            encoded = self._encode_tensor(
                torch.cat((values, positive_values)),
                return_level_data=return_level_data,
                return_encoder_hidden=return_encoder_hidden,
            )
            if return_level_data and return_encoder_hidden:
                (
                    all_quantized,
                    all_codes,
                    commitment_loss,
                    consistency_residuals,
                    level_codes,
                    encoder_hidden,
                ) = encoded
            elif return_level_data:
                (
                    all_quantized,
                    all_codes,
                    commitment_loss,
                    consistency_residuals,
                    level_codes,
                ) = encoded
                encoder_hidden = None
            elif return_encoder_hidden:
                all_quantized, all_codes, commitment_loss, encoder_hidden = encoded
                consistency_residuals = level_codes = None
            else:
                all_quantized, all_codes, commitment_loss = encoded
                consistency_residuals = level_codes = encoder_hidden = None
            quantized = all_quantized[:packed_batch_size]
            packed_codes = all_codes[:packed_batch_size]

        reconstructed = self._decode_tensor(quantized)
        reconstruction_loss = functional.mse_loss(reconstructed, target)
        loss = (
            self.config.reconstruction_loss_weight * reconstruction_loss
            + self.config.commitment_loss_weight * commitment_loss
        )
        loss_dict: dict[str, Tensor] = {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
        }

        for index, name in enumerate(names):
            component_reconstruction = reconstructed[index * batch_size : (index + 1) * batch_size]
            component_target = target[index * batch_size : (index + 1) * batch_size]
            dimension = original_dims.get(name, self.config.max_component_dim)
            loss_dict[f"recon/{name}"] = functional.mse_loss(
                component_reconstruction[..., :dimension], component_target[..., :dimension]
            ).detach()

        for level, quantizer in enumerate(self.rvq.quantizers):
            cluster_size = quantizer.cluster_size.float()
            total = cluster_size.sum()
            if total > 0:
                probabilities = cluster_size / total
                perplexity = torch.exp(-(probabilities * torch.log(probabilities + 1e-10)).sum())
                utilization = (cluster_size >= quantizer.threshold_ema_dead).float().mean()
            else:
                perplexity = cluster_size.new_tensor(1.0)
                utilization = cluster_size.new_tensor(0.0)
            loss_dict[f"codebook/perplexity_l{level}"] = perplexity.detach()
            loss_dict[f"codebook/utilization_l{level}"] = utilization.detach()

        if x_pos_dict is not None and use_consistency and consistency_type == "token_residual":
            effective_layer_weights = layer_weights or [1.0] * self.config.n_codebooks
            consistency_loss, consistency_metrics = compute_consistency_loss(
                consistency_residuals,
                level_codes,
                packed_batch_size,
                effective_layer_weights,
            )
            loss = loss + self.config.consistency_loss_weight * consistency_loss
            loss_dict["loss"] = loss
            loss_dict.update(consistency_metrics)
        elif x_pos_dict is not None and use_consistency and consistency_type == "action_time_contrastive":
            if self.action_time_contrastive_loss is None or encoder_hidden is None:
                raise RuntimeError("action-time contrastive loss was not initialized")
            consistency_loss, consistency_metrics = self.action_time_contrastive_loss(
                encoder_hidden[:packed_batch_size], encoder_hidden[packed_batch_size:]
            )
            loss = loss + self.config.consistency_loss_weight * consistency_loss
            loss_dict["loss"] = loss
            loss_dict.update(consistency_metrics)

        codes = {
            name: packed_codes[index * batch_size : (index + 1) * batch_size]
            for index, name in enumerate(names)
        }
        reconstructions = {
            name: reconstructed[
                index * batch_size : (index + 1) * batch_size,
                :,
                : original_dims.get(name, self.config.max_component_dim),
            ]
            for index, name in enumerate(names)
        }
        return {
            "loss": loss,
            "reconstructions": reconstructions,
            "codes": codes,
            "loss_dict": loss_dict,
        }
