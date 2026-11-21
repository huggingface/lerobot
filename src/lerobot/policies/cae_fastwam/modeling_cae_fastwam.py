# Copyright 2026 Advanced Micro Devices, Inc. and The HuggingFace Inc. team. All rights reserved.
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

"""CAE-FastWAM: FastWAM with a Compact Action Expert and a suite-conditioned action VAE.

The action encoder and decoder, and the training objective (`CAEFastWAM.build_inputs`), are adapted
from `FastWAM/src/fastwam/models/action_ae.py` and `FastWAM/src/fastwam/models/wan22/fastwam.py` of
https://github.com/AMD-AGI/faster-action-models (Apache-2.0, Copyright (c) 2026 Advanced Micro
Devices, Inc.). Everything else is LeRobot's FastWAM.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
from lerobot.policies.fastwam.wan import FastWAM

from .configuration_cae_fastwam import CAEFastWAMConfig


def _upsampling_strides(ratio: int) -> list[int]:
    """Factorise the token-to-action resampling ratio into (transposed-)convolution strides."""
    if ratio == 1:
        return [1]
    strides = []
    for factor in (4, 2, 3):
        while ratio % factor == 0:
            strides.append(factor)
            ratio //= factor
    if ratio > 1:
        strides.append(ratio)
    return strides


class CAEResidualConv1dBlock(nn.Module):
    """Residual block of two 1D convolutions with GroupNorm, as in the released action VAE."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(min(8, channels), channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return `x` plus the block's residual."""
        return x + self.net(x)


class CAEActionEncoder(nn.Module):
    """Encoder of CAE-FastWAM's temporal action VAE (frozen; used for training targets only).

    Maps `[B, action_horizon, action_dim]` normalized actions to `num_tokens` latent tokens: a linear
    projection plus an optional per-suite embedding, strided 1D convolutions with residual blocks, and
    a linear bottleneck. A variational encoder predicts `(mu, log_var)` and returns the posterior sample
    `z = mu + exp(log_var / 2) * eps` (drawn from the global torch generator), as the original
    implementation does when it builds the action targets. It runs in float32.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        action_horizon: int,
        num_tokens: int,
        latent_dim: int,
        hidden_dim: int,
        num_suites: int,
        variational: bool,
    ):
        super().__init__()
        if action_horizon % num_tokens != 0:
            raise ValueError(
                f"action_horizon ({action_horizon}) must be divisible by num_tokens ({num_tokens})."
            )
        self.action_horizon = action_horizon
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        self.variational = variational
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        layers: list[nn.Module] = []
        for stride in _upsampling_strides(action_horizon // num_tokens):
            layers.extend(
                [
                    nn.Conv1d(
                        hidden_dim, hidden_dim, kernel_size=stride * 2 - 1, stride=stride, padding=stride - 1
                    ),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.SiLU(),
                    CAEResidualConv1dBlock(hidden_dim),
                ]
            )
        self.down = nn.Sequential(*layers)
        self.suite_embed = None
        if num_suites > 0:
            self.suite_embed = nn.Embedding(num_suites, hidden_dim)
            nn.init.zeros_(self.suite_embed.weight)
        self.bottleneck = nn.Linear(hidden_dim, latent_dim * (2 if variational else 1))

    def forward(self, actions: Tensor, suite_ids: Tensor | None = None) -> dict[str, Tensor]:
        """Encode `[B, action_horizon, action_dim]` actions; returns `z` (and `mu`, `log_var`)."""
        if actions.shape[1] != self.action_horizon:
            raise ValueError(
                f"The action encoder takes {self.action_horizon}-step chunks, got {actions.shape[1]} steps."
            )
        x = self.input_proj(actions)
        if self.suite_embed is not None:
            if suite_ids is None:
                raise ValueError("This action encoder is suite-conditioned; `suite_ids` is required.")
            x = x + self.suite_embed(suite_ids).unsqueeze(1)
        x = self.down(x.transpose(1, 2)).transpose(1, 2)
        output = self.bottleneck(x)
        if not self.variational:
            return {"z": output}
        mu, log_var = output.chunk(2, dim=-1)
        z = mu + (0.5 * log_var).exp() * torch.randn_like(mu)
        return {"z": z, "mu": mu, "log_var": log_var}


class CAEActionDecoder(nn.Module):
    """Decoder of CAE-FastWAM's temporal action VAE.

    Maps `[B, num_tokens, latent_dim]` latents to `[B, action_horizon, action_dim]` normalized
    actions: a linear projection plus an optional per-suite embedding, transposed-convolution
    upsampling with residual 1D convolutions, and a linear output projection. It runs in float32.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        action_horizon: int,
        num_tokens: int,
        latent_dim: int,
        hidden_dim: int,
        num_suites: int,
    ):
        super().__init__()
        if action_horizon % num_tokens != 0:
            raise ValueError(
                f"action_horizon ({action_horizon}) must be divisible by num_tokens ({num_tokens})."
            )
        self.action_horizon = action_horizon
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        layers: list[nn.Module] = []
        for stride in _upsampling_strides(action_horizon // num_tokens):
            layers.extend(
                [
                    nn.ConvTranspose1d(
                        hidden_dim, hidden_dim, kernel_size=stride * 2, stride=stride, padding=stride // 2
                    ),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.SiLU(),
                    CAEResidualConv1dBlock(hidden_dim),
                ]
            )
        self.up = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        self.suite_embed = None
        if num_suites > 0:
            self.suite_embed = nn.Embedding(num_suites, hidden_dim)
            nn.init.zeros_(self.suite_embed.weight)

    def forward(self, latents: Tensor, suite_ids: Tensor | None = None) -> Tensor:
        """Decode `[B, num_tokens, latent_dim]` latents into `[B, action_horizon, action_dim]` actions."""
        x = self.input_proj(latents)
        if self.suite_embed is not None:
            if suite_ids is None:
                raise ValueError("This action decoder is suite-conditioned; `suite_ids` is required.")
            x = x + self.suite_embed(suite_ids).unsqueeze(1)
        x = self.up(x.transpose(1, 2))
        if x.shape[2] > self.action_horizon:
            x = x[:, :, : self.action_horizon]
        elif x.shape[2] < self.action_horizon:
            x = functional.pad(x, (0, self.action_horizon - x.shape[2]))
        return self.output_proj(x.transpose(1, 2))


class CAEFastWAM(FastWAM):
    """FastWAM core whose action expert denoises latent tokens that `action_decoder` decodes.

    The policy attaches `action_encoder` and `action_decoder` after construction and sets `suite_id`
    before each call. Training (`training_loss`, inherited) runs FastWAM's joint flow-matching objective
    on the action latents that `build_inputs` encodes from the ground-truth chunk.
    """

    action_encoder: CAEActionEncoder
    action_decoder: CAEActionDecoder
    suite_id: int | None = None

    def train(self, mode: bool = True) -> CAEFastWAM:
        """Set the training mode; the frozen action VAE always stays in eval mode."""
        super().train(mode)
        # The action VAE is frozen and stays in eval mode, as in the original trainer (`model.eval()`,
        # then `dit.train()` and `proprio_encoder.train()` only).
        for name in ("action_encoder", "action_decoder"):
            module = getattr(self, name, None)
            if module is not None:
                module.eval()
        return self

    def _suite_ids(self, module: nn.Module, batch_size: int) -> Tensor | None:
        """Suite ids (`suite_id`, set from `task_suite`) for a suite-conditioned action encoder/decoder."""
        if module.suite_embed is None:
            return None
        if self.suite_id is None:
            raise ValueError("CAE-FastWAM is suite-conditioned: set `suite_id` (from `task_suite`) first.")
        device = module.input_proj.weight.device
        return torch.full((batch_size,), int(self.suite_id), dtype=torch.long, device=device)

    @torch.no_grad()
    def encode_actions(self, actions: Tensor) -> Tensor:
        """Encode `[B, action_horizon, action_dim]` normalized actions into `[B, num_tokens, latent_dim]`
        latents (a posterior sample for a variational encoder) in the model dtype.

        The encoder runs in float32 on the model-dtype chunk, under the caller's autocast context.
        """
        weight = self.action_encoder.input_proj.weight
        suite_ids = self._suite_ids(self.action_encoder, actions.shape[0])
        latents = self.action_encoder(actions.to(device=weight.device, dtype=weight.dtype), suite_ids)["z"]
        return latents.to(device=self.device, dtype=self.torch_dtype)

    def build_inputs(self, sample, tiled: bool = False):
        """FastWAM's training inputs with the action chunk replaced by its action-VAE latents.

        `action_is_pad` is pooled to the latent tokens (a token is padding if any of its steps is).
        """
        inputs = super().build_inputs(sample, tiled=tiled)
        latents = self.encode_actions(inputs["action"])
        action_is_pad = inputs["action_is_pad"]
        if action_is_pad is not None:
            batch_size, num_tokens = latents.shape[:2]
            if action_is_pad.shape[1] % num_tokens:
                raise ValueError("Action padding horizon is not divisible by the latent horizon.")
            ratio = action_is_pad.shape[1] // num_tokens
            inputs["action_is_pad"] = action_is_pad.view(batch_size, num_tokens, ratio).any(dim=2)
        inputs["action"] = latents
        return inputs

    @torch.no_grad()
    def decode_action_latents(self, latents: Tensor) -> Tensor:
        """Decode `[B, num_tokens, latent_dim]` latents into `[B, action_horizon, action_dim]` actions.

        The decoder runs in float32 on the model-dtype latents and its output is rounded to the model
        dtype, as in the original implementation.
        """
        weight = self.action_decoder.input_proj.weight
        suite_ids = self._suite_ids(self.action_decoder, latents.shape[0])
        actions = self.action_decoder(latents.to(device=weight.device, dtype=weight.dtype), suite_ids)
        return actions.to(dtype=self.torch_dtype)

    def infer(self, *args, **kwargs) -> dict[str, Any]:
        """Joint video and action inference is not supported: CAE-FastWAM's action expert denoises
        latent tokens, which only `infer_action` decodes into actions."""
        raise NotImplementedError(
            "CAE-FastWAM supports action inference only (`infer_action`); joint video and action "
            "inference (`infer`) would return undecoded action latents."
        )

    @torch.no_grad()
    def infer_action(self, prompt, input_image, action_horizon, *args, **kwargs) -> dict[str, Any]:
        """FastWAM action inference over `num_tokens` latent tokens, decoded to `action_horizon` actions.

        `noise`, when given, is the initial latent sample `[1, num_tokens, latent_dim]`. Returns the
        decoded chunk under `"action"` and the denoised latents under `"action_latents"`.
        """
        decoder = self.action_decoder
        if int(action_horizon) != decoder.action_horizon:
            raise ValueError(
                f"CAE-FastWAM decodes {decoder.action_horizon}-step chunks, got action_horizon={action_horizon}."
            )
        output = super().infer_action(prompt, input_image, decoder.num_tokens, *args, **kwargs)
        latents = output["action"]
        actions = self.decode_action_latents(latents.unsqueeze(0))[0]
        return {"action": actions.to(device="cpu", dtype=torch.float32), "action_latents": latents}


class CAEFastWAMPolicy(FastWAMPolicy):
    """LeRobot policy wrapper for CAE-FastWAM.

    It reuses FastWAM's video expert, mixture-of-transformers, Wan2.2 VAE and UMT5 text encoder, and
    adds the action VAE: its decoder turns the denoised latent tokens into an action chunk, its
    encoder (training only) turns ground-truth chunks into the latent targets. Both are frozen.

    Args:
        config (CAEFastWAMConfig): CAE-FastWAM policy configuration.
        dataset_stats (dict[str, dict[str, Tensor]] | None): Unused, as in FastWAM (statistics live
            in the processors).
    """

    config_class = CAEFastWAMConfig
    name = "cae_fastwam"
    _core_model_class = CAEFastWAM

    def __init__(
        self,
        config: CAEFastWAMConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, dataset_stats, **kwargs)
        self.config: CAEFastWAMConfig = config
        # Kept in float32 whatever `config.dtype` is: the decoder is tiny and was trained in float32.
        self.model.action_decoder = CAEActionDecoder(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            num_tokens=config.action_latent_tokens,
            latent_dim=config.action_latent_dim,
            hidden_dim=config.action_decoder_hidden_dim,
            num_suites=len(config.task_suites),
        ).to(device=config.device, dtype=torch.float32)
        self.model.action_encoder = CAEActionEncoder(
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            num_tokens=config.action_latent_tokens,
            latent_dim=config.action_latent_dim,
            hidden_dim=config.action_encoder_hidden_dim,
            num_suites=len(config.task_suites),
            variational=config.action_encoder_variational,
        ).to(device=config.device, dtype=torch.float32)
        # The action VAE is pretrained and frozen: only the mixture-of-transformers (`model.dit`) and the
        # proprio encoder are trained, as in the original recipe (see `get_optim_params`).
        for module in (self.model.action_encoder, self.model.action_decoder):
            module.requires_grad_(False)
            module.eval()

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict[str, Any]]:
        """Compute the CAE-FastWAM training loss for a preprocessed LeRobot batch.

        FastWAM's training loss, with the action targets replaced by the frozen action-VAE encoder's
        latents of the normalized ground-truth chunk (suite `config.task_suite`; a posterior sample).

        Args:
            batch (dict[str, Tensor]): Batch with the FastWAM training keys (see `FastWAMPolicy.forward`).
            reduction (str): "mean" (scalar loss) or "none" (per-sample losses `(B,)`).

        Returns:
            tuple[Tensor, dict[str, Any]]: The loss and `{"loss_video", "loss_action"}`.
        """
        self.model.suite_id = self.config.suite_id()
        return super().forward(batch, reduction=reduction)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        """Predict a `[B, action_horizon, action_dim]` normalized action chunk.

        Args:
            batch (dict[str, Tensor]): Preprocessed observation batch (see `FastWAMPolicy`).
            noise (Tensor | None): Optional initial latent sample `[B, action_latent_tokens,
                action_latent_dim]`; drawn from `config.inference_seed` when None.

        Returns:
            Tensor: Action chunk `[B, action_horizon, action_dim]`.
        """
        self.model.suite_id = self.config.suite_id()
        return super().predict_action_chunk(batch, noise=noise)
