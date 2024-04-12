import logging
import math

import einops
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class _Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class _ConditionalResidualBlock1D(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.film_scale_modulation = film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = _Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = _Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Two types of conditioning can be applied:
    - Global: Conditioning information that is aggregated over the whole observation window. This is
        incorporated via the FiLM technique in the residual convolution blocks of the Unet's encoder/decoder.
    - Local: Conditioning information for each timestep in the observation window. This is incorporated
        by encoding the information via 1D convolutions and adding the resulting embeddings to the inputs and
        outputs of the Unet's encoder/decoder.
    """

    def __init__(
        self,
        input_dim: int,
        local_cond_dim: int | None = None,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: int | None = None,
        kernel_size: int = 3,
        n_groups: int = 8,
        film_scale_modulation: bool = False,
    ):
        super().__init__()

        if down_dims is None:
            down_dims = [256, 512, 1024]

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            _SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = diffusion_step_embed_dim
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        self.local_cond_down_encoder = None
        self.local_cond_up_encoder = None
        if local_cond_dim is not None:
            # Encoder for the local conditioning. The output gets added to the Unet encoder input.
            self.local_cond_down_encoder = _ConditionalResidualBlock1D(
                local_cond_dim,
                down_dims[0],
                cond_dim=cond_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                film_scale_modulation=film_scale_modulation,
            )
            # Encoder for the local conditioning. The output gets added to the Unet encoder output.
            self.local_cond_up_encoder = _ConditionalResidualBlock1D(
                local_cond_dim,
                down_dims[0],
                cond_dim=cond_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                film_scale_modulation=film_scale_modulation,
            )

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(input_dim, down_dims[0])] + list(zip(down_dims[:-1], down_dims[1:], strict=True))

        # Unet encoder.
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        _ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            film_scale_modulation=film_scale_modulation,
                        ),
                        _ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            film_scale_modulation=film_scale_modulation,
                        ),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                _ConditionalResidualBlock1D(
                    down_dims[-1],
                    down_dims[-1],
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    film_scale_modulation=film_scale_modulation,
                ),
                _ConditionalResidualBlock1D(
                    down_dims[-1],
                    down_dims[-1],
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    film_scale_modulation=film_scale_modulation,
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        _ConditionalResidualBlock1D(
                            dim_in * 2,  # x2 as it takes the encoder's skip connection as well
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            film_scale_modulation=film_scale_modulation,
                        ),
                        _ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            film_scale_modulation=film_scale_modulation,
                        ),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            _Conv1dBlock(down_dims[0], down_dims[0], kernel_size=kernel_size),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, local_cond=None, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            local_cond: (B, T, local_cond_dim)
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim)
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")
        if local_cond is not None:
            if self.local_cond_down_encoder is None or self.local_cond_up_encoder is None:
                raise ValueError(
                    "`local_cond` was provided but the relevant encoders weren't built at initialization."
                )
            local_cond = einops.rearrange(local_cond, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        encoder_skip_features: list[Tensor] = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and local_cond is not None:
                x = x + self.local_cond_down_encoder(local_cond, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            # Note: The condition in the original implementation is:
            # if idx == len(self.up_modules) and local_cond is not None:
            # But as they mention in their comments, this is incorrect. We use the correct condition here.
            if idx == (len(self.up_modules) - 1) and local_cond is not None:
                x = x + self.local_cond_up_encoder(local_cond, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x
