# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional
from transformers import PreTrainedModel

from .configuration_actioncodec import G05ActionCodecConfig
from .modular_actioncodec import (
    ActionCodecDecoder,
    ActionCodecEncoder,
    BlockDCT,
    ResidualVectorQuantizer,
)


class G05ActionCodecModel(PreTrainedModel):
    """Neural action codec used by released G0.5 checkpoints.

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
        _, codes = self.rvq.encode(hidden_states)
        return {
            name: codes[index * batch_size : (index + 1) * batch_size] for index, name in enumerate(names)
        }

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
        hidden_states = self.rvq.decode(packed_codes)
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
        dimensions = component_dims or self.config.parts_meta
        return {
            name: values[
                index * batch_size : (index + 1) * batch_size, :, : dimensions.get(name, values.shape[-1])
            ]
            for index, name in enumerate(names)
        }

    def forward(self, components: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return reconstructions and codes; intended for codec diagnostics."""
        codes = self.encode(components)
        reconstructions = self.decode(codes, {name: values.shape[-1] for name, values in components.items()})
        loss = torch.stack(
            [
                functional.mse_loss(reconstructions[name], values[:, : self.config.horizon].float())
                for name, values in components.items()
            ]
        ).mean()
        return {"loss": loss, "reconstructions": reconstructions, "codes": codes}
