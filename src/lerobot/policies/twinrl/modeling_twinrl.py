#!/usr/bin/env python

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

"""TwinRL-VLA policy.

Joint IL+RL training objective (paper Eq. 6):
    L_actor(ψ) = β * L_IL + η * L_Q

where:
    L_IL = recon_loss (ConRFT Karras-weighted) or MSE(π_mean, a_demo) for Gaussian
    L_Q  = -E[mean_i Q_i(s, π(s))]         -- RL Q-gradient

Critic uses Cal-QL (paper Appendix C, official code calql_critic_loss_fn):
    L_critic = L_TD + α * L_CQL

where L_CQL applies a logsumexp penalty over sampled OOD actions (3n), with
Monte Carlo return lower-bound clipping to prevent Q underestimation.

Paper: https://arxiv.org/abs/2602.09023
"""

import math
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
from typing import Literal

import einops
import torch
import torch.nn as nn
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import is_image_feature
from lerobot.policies.sac.modeling_sac import (
    Policy,
    SpatialLearnedEmbeddings,
)
from lerobot.policies.twinrl.configuration_twinrl import TwinRLConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


class TwinRLMLP(nn.Module):
    """Matches JAX MLP precisely: Linear -> (Dropout) -> (LayerNorm) -> Activation.

    Default behavior uses SiLU and NO LayerNorm, with xavier_uniform initialization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: nn.Module = nn.Tanh(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i + 1 < len(hidden_dims) or activate_final:
                if dropout_rate is not None and dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(activations)
            in_dim = out_dim
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TwinRLCriticEnsemble(nn.Module):
    """Matches JAX Critic precisely: shared final Dense(1) layer across ensemblized backbones."""

    def __init__(
        self,
        encoder: nn.Module,
        ensemble_size: int,
        input_dim: int,
        hidden_dims: list[int],
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        # Ensemble of MLPs (the backbone)
        self.mlps = nn.ModuleList(
            [
                TwinRLMLP(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    activations=nn.Tanh(),
                    activate_final=False,
                    use_layer_norm=True,
                )
                for _ in range(ensemble_size)
            ]
        )
        # Shared final layer
        self.shared_final = nn.Linear(hidden_dims[-1], 1)

        # Initialization
        if init_final is not None:
            nn.init.uniform_(self.shared_final.weight, -init_final, init_final)
            nn.init.uniform_(self.shared_final.bias, -init_final, init_final)
        else:
            nn.init.xavier_uniform_(self.shared_final.weight)
            if self.shared_final.bias is not None:
                nn.init.zeros_(self.shared_final.bias)

    def forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        obs_enc = self.encoder(observations, cache=observation_features)
        inputs = torch.cat([obs_enc, actions], dim=-1)

        # Apply each MLP to the inputs
        outputs = []
        for mlp in self.mlps:
            outputs.append(mlp(inputs))

        # outputs shape: [ensemble_size, batch, hidden_dim]
        outputs = torch.stack(outputs, dim=0)

        # Apply shared final layer
        q_values = self.shared_final(outputs).squeeze(-1)  # [ensemble_size, batch]
        return q_values


class OctoActorEncoder(nn.Module):
    """Frozen OctoTransformer used as actor feature extractor.

    Loads weights from HuggingFace (requires octo-pytorch package).
    Outputs the mean-pooled readout_action token: (b, token_embedding_size).
    The transformer weights are frozen; only downstream layers are trained.
    """

    def __init__(self, config):
        super().__init__()
        from octo_pytorch.model.modeling_octo import OctoModel

        octo = OctoModel.from_pretrained(config.octo_model_name)
        self.transformer = octo.octo_transformer
        self.text_processor = octo.text_processor
        self._output_dim = self.transformer.token_embedding_size
        for param in self.transformer.parameters():
            param.requires_grad = False

        from lerobot.utils.constants import OBS_STATE

        self.use_proprio = OBS_STATE in config.input_features
        if self.use_proprio:
            state_dim = config.input_features[OBS_STATE].shape[0]
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, config.state_encoder_hidden_dim),
                nn.LayerNorm(config.state_encoder_hidden_dim),
                nn.Tanh(),
            )
            self._output_dim += config.state_encoder_hidden_dim
        else:
            self.state_encoder = None

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        observations: dict[str, Tensor],
        obs_features: Tensor | None = None,
        cache: Tensor | None = None,
        detach: bool = False,
        repeat: int = -1,
    ) -> Tensor:
        """Return (b, token_embedding_size) readout feature vector."""
        features = obs_features if obs_features is not None else cache
        if features is not None:
            feats = features if not detach else features.detach()
            if repeat > 1:
                return feats.unsqueeze(1).expand(-1, repeat, -1)
            return feats

        device = next(self.transformer.parameters()).device
        img = observations.get(
            "image_primary",
            observations.get("observation.image_primary", observations.get("observation.image")),
        )
        if img is None:
            raise ValueError("OctoActorEncoder requires 'image_primary' in observations.")

        if img.ndim == 4:
            img = img.unsqueeze(1)

        b, horizon, c, h, w = img.shape
        # Octo expects (256, 256) for primary and (128, 128) for wrist
        if h != 256 or w != 256:
            img = torch.nn.functional.interpolate(
                img.view(b * horizon, c, h, w), size=(256, 256), mode="bilinear", align_corners=False
            ).view(b, horizon, c, 256, 256)

        # Scale to [0, 255] if input is [0, 1]
        if img.max() <= 1.1:
            img = img * 255.0

        obs_octo = {
            "image_primary": img.permute(0, 1, 3, 4, 2),
            "pad_mask_dict": {"image_primary": torch.ones(b, horizon, dtype=torch.bool, device=device)},
        }

        w_img = observations.get("image_wrist", observations.get("observation.image_wrist"))
        if w_img is not None:
            if w_img.ndim == 4:
                w_img = w_img.unsqueeze(1)

            bw, hw, cw, hh, ww = w_img.shape
            if hh != 128 or ww != 128:
                w_img = torch.nn.functional.interpolate(
                    w_img.view(bw * hw, cw, hh, ww), size=(128, 128), mode="bilinear", align_corners=False
                ).view(bw, hw, cw, 128, 128)

            if w_img.max() <= 1.1:
                w_img = w_img * 255.0

            obs_octo["image_wrist"] = w_img.permute(0, 1, 3, 4, 2)
            obs_octo["pad_mask_dict"]["image_wrist"] = torch.ones(bw, hw, dtype=torch.bool, device=device)

        # Skip language if empty (matching JAX Octo behavior)
        instruction = observations.get("language_instruction", [""] * b)
        if isinstance(instruction, list) and all(t == "" for t in instruction):
            tasks = {}
        else:
            tasks = {
                "language_instruction": self.text_processor.encode(instruction),
                "pad_mask_dict": {"language_instruction": torch.ones(b, dtype=torch.bool, device=device)},
            }
            for k, v in tasks["language_instruction"].items():
                if isinstance(v, torch.Tensor):
                    tasks["language_instruction"][k] = v.to(device)

        window = obs_octo["image_primary"].shape[1]
        pad_mask = torch.ones(b, window, dtype=torch.bool, device=device)

        with torch.no_grad():
            outputs = self.transformer(obs_octo, tasks, pad_mask)

        # tokens: (b, window, n_tokens, embed) → mean over tokens → last window step
        feats = outputs["readout_action"].tokens.mean(dim=-2)[:, -1, :]  # (b, embed)

        if self.use_proprio:
            from lerobot.utils.constants import OBS_STATE

            state_feat = self.state_encoder(observations[OBS_STATE])
            feats = torch.cat([feats, state_feat], dim=-1)

        if repeat > 1:
            feats = feats.unsqueeze(1).expand(-1, repeat, -1)

        return feats


class _SinusoidalPosEmb(nn.Module):
    """Matches JAX timeMLP's SinusoidalPosEmb: sin/cos of log-spaced frequencies."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:  # x: (b,)
        half = self.dim // 2
        freqs = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=x.device) * -freqs)  # (half,)
        emb = x.unsqueeze(-1) * freqs.unsqueeze(0)  # (b, half)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (b, dim)


class ConsistencyActorHead(nn.Module):
    """Karras consistency model actor (ConRFT-style, matches JAX ConsistencyPolicy_octo).

    base_network(x_t, sigma, obs_enc) predicts x_0 via boundary conditions:
        c_skip * x_t + c_out * network([c_in * x_t, t_embed, obs_enc])

    Backbone: flat TwinRLMLP matching JAX MLP (not ResNet).
    Time embed: sinusoidal (JAX timeMLP) -> two-layer MLP projection.
    """

    def __init__(self, encoder: nn.Module, obs_enc_dim: int, action_dim: int, config):
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.sigma_data = config.sigma_data

        t_dim = config.consistency_t_dim
        hidden_dims = list(config.actor_network_kwargs.hidden_dims)

        # JAX timeMLP: SinusoidalPosEmb(t_dim) -> Dense(2*t_dim) -> activations -> Dense(t_dim)
        self.sinusoidal = _SinusoidalPosEmb(t_dim)
        self.t_proj = nn.Sequential(nn.Linear(t_dim, 2 * t_dim), nn.Tanh(), nn.Linear(2 * t_dim, t_dim))
        # JAX MLP backbone: flat hidden layers with activate_final=True
        net_in_dim = action_dim + t_dim + obs_enc_dim
        self.network = TwinRLMLP(
            input_dim=net_in_dim,
            hidden_dims=hidden_dims,
            activations=nn.Tanh(),
            activate_final=False,
            use_layer_norm=True,
        )
        self.output_proj = nn.Linear(hidden_dims[-1], action_dim)

        sigmas = self._karras_sigmas(config.num_scales, config.sigma_min, config.sigma_max, config.rho)
        self.register_buffer("karras_sigmas", sigmas)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _karras_sigmas(n: int, sigma_min: float, sigma_max: float, rho: float) -> Tensor:
        ramp = torch.linspace(0, 1, n)
        min_inv = sigma_min ** (1 / rho)
        max_inv = sigma_max ** (1 / rho)
        sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
        return torch.cat([sigmas, sigmas.new_zeros(1)])  # append 0 sentinel

    def _scalings(self, sigma: Tensor):
        c_skip = self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)
        c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1.0 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def base_network(self, x_t: Tensor, sigma: Tensor, obs_enc: Tensor, repeat: int = -1) -> Tensor:
        """Predict x_0 from noisy x_t at noise level sigma.
        sigma: (b,) or (b, r). x_t: (b, d) or (b, r, d).
        """
        c_skip, c_out, c_in = self._scalings(sigma)
        c_skip = c_skip.unsqueeze(-1)
        c_out = c_out.unsqueeze(-1)
        c_in = c_in.unsqueeze(-1)
        rescaled_t = 1000.0 * 0.25 * torch.log(sigma + 1e-44)
        t_embed = self.t_proj(self.sinusoidal(rescaled_t))  # (b, t_dim) or (b, r, t_dim)

        cont_axis = -1
        if repeat > 1:
            # Match JAX extend_and_repeat logic for t_embed and obs_enc
            if t_embed.ndim == 2:
                t_embed = t_embed.unsqueeze(1).expand(-1, repeat, -1)
            if obs_enc.ndim == 2:
                obs_enc = obs_enc.unsqueeze(1).expand(-1, repeat, -1)
            cont_axis = 2

        net_in = torch.cat([c_in * x_t, t_embed, obs_enc], dim=cont_axis)
        features = self.network(net_in)
        denoised = self.output_proj(features)
        return c_out * denoised + c_skip * x_t

    def forward(self, observations: dict[str, Tensor], obs_features: Tensor | None = None, repeat: int = -1):
        """Single-step inference: denoise from x_t ~ N(0, sigma_max²)."""
        obs_enc = self.encoder(observations, obs_features)
        b, device = obs_enc.shape[0], obs_enc.device

        x_shape = (b, repeat, self.action_dim) if repeat > 1 else (b, self.action_dim)
        x_t = torch.randn(x_shape, device=device) * self.sigma_max
        sigma = self.karras_sigmas[0].expand(b)
        if repeat > 1:
            sigma = sigma.unsqueeze(1).expand(-1, repeat)

        x_0 = self.base_network(x_t, sigma, obs_enc, repeat=repeat).clamp(-1, 1)
        return x_0, None, x_0


class ResNetBlock(nn.Module):
    def __init__(self, in_filters, out_filters, strides=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_filters,
            out_filters,
            kernel_size=3,
            stride=strides,
            padding=1 if strides == (1, 1) else 0,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=out_filters, eps=1e-5)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=out_filters, eps=1e-5)
        self.act = nn.ReLU()

        if strides != (1, 1) or in_filters != out_filters:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=strides, padding=0, bias=False),
                nn.GroupNorm(num_groups=4, num_channels=out_filters, eps=1e-5),
            )
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        # Handle asymmetric padding for stride 2 to match JAX "SAME"
        # For kernel 3, stride 2, JAX "SAME" results in (0, 1, 0, 1) padding
        if self.conv1.stride == (2, 2):
            x_padded = torch.nn.functional.pad(x, (0, 1, 0, 1))  # Pad right and bottom
            y = self.act(self.norm1(self.conv1(x_padded)))
        else:
            y = self.act(self.norm1(self.conv1(x)))

        y = self.norm2(self.conv2(y))

        # Residual projection: For kernel 1, stride 2, JAX "SAME" is symmetric (no padding needed)
        residual = self.residual_proj(x)

        return self.act(residual + y)


class ResNet10Encoder(nn.Module):
    def __init__(self, num_filters=64, num_spatial_blocks=8, bottleneck_dim=256, pretrained=False):
        super().__init__()
        self.image_size = (128, 128)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.conv_init = nn.Conv2d(3, num_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm_init = nn.GroupNorm(num_groups=4, num_channels=num_filters, eps=1e-5)
        self.act = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.layer1 = ResNetBlock(num_filters, num_filters)
        self.layer2 = ResNetBlock(num_filters, num_filters * 2, strides=2)
        self.layer3 = ResNetBlock(num_filters * 2, num_filters * 4, strides=2)
        self.layer4 = ResNetBlock(num_filters * 4, num_filters * 8, strides=2)

        if pretrained:
            self.load_resnet10_weights()
            # In JAX resnet-pretrained, the backbone is frozen
            for param in self.parameters():
                param.requires_grad = False

        # Final spatial pooling (always trainable in SERL even if backbone is frozen)
        self.spatial_pool = SpatialLearnedEmbeddings(4, 4, num_filters * 8, num_features=num_spatial_blocks)
        # TODO: IS DETERMINISTIC NEEDED TO BE SET HERE
        self.dropout = nn.Dropout(0.1)
        self.bottleneck = nn.Sequential(
            nn.Linear(num_filters * 8 * num_spatial_blocks, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.Tanh(),
        )
        self.output_dim = bottleneck_dim

    def load_resnet10_weights(self):
        import os

        weight_path = os.path.expanduser("~/.serl/resnet10_params.pt")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Pretrained ResNet-10 weights not found at {weight_path}. Please run conversion script first."
            )

        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        # Only load keys that exist in our model (backbone only)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)} layers of pretrained ResNet-10 weights.")

    def forward(self, x, train=True):
        # Resize to 128x128
        if x.shape[-2:] != self.image_size:
            x = torch.nn.functional.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)

        # Normalize [0, 1] to ImageNet
        if x.max() > 1.1:
            x = x / 255.0
        x = (x - self.mean) / self.std

        # Initial layers
        x = self.act(self.norm_init(self.conv_init(x)))
        # Asymmetric padding for max_pool to match JAX "SAME"
        x = torch.nn.functional.pad(x, (0, 1, 0, 1))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.spatial_pool(x)
        if train:
            x = self.dropout(x)
        x = self.bottleneck(x)
        return x


class TwinRLObservationEncoder(nn.Module):
    def __init__(self, config: TwinRLConfig):
        super().__init__()
        self.config = config
        self.image_keys = [k for k in config.input_features if is_image_feature(k)]
        self.state_keys = [k for k in config.input_features if k in (OBS_STATE, OBS_ENV_STATE)]

        self.image_encoders = nn.ModuleDict()
        for key in self.image_keys:
            name = key.replace(".", "_")
            self.image_encoders[name] = ResNet10Encoder(
                bottleneck_dim=config.latent_dim,
                pretrained=(config.vision_encoder_name == "resnet-pretrained"),
            )

        self.use_proprio = OBS_STATE in config.input_features
        if self.use_proprio:
            state_dim = config.input_features[OBS_STATE].shape[0]
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, config.state_encoder_hidden_dim),
                nn.LayerNorm(config.state_encoder_hidden_dim),
                nn.Tanh(),
            )
        else:
            self.state_encoder = None

        self._out_dim = len(self.image_keys) * config.latent_dim
        if self.use_proprio:
            self._out_dim += config.state_encoder_hidden_dim

    @property
    def output_dim(self):
        return self._out_dim

    def forward(self, obs, cache=None, detach=False):
        parts = []
        for key in self.image_keys:
            name = key.replace(".", "_")
            x = self.image_encoders[name](obs[key], train=self.training)
            if detach:
                x = x.detach()
            parts.append(x)

        if self.use_proprio:
            x = self.state_encoder(obs[OBS_STATE])
            if detach:
                x = x.detach()
            parts.append(x)

        return torch.cat(parts, dim=-1)


class TwinRLPolicy(PreTrainedPolicy):
    """TwinRL policy with joint BC+RL actor loss and Cal-QL critic."""

    config_class = TwinRLConfig
    name = "twinrl"

    def __init__(self, config: TwinRLConfig | None = None, **kwargs):
        super().__init__(config)
        self.config = config
        config.validate_features()

        action_dim = config.output_features[ACTION].shape[0]
        self._init_encoders()
        self._init_critics(action_dim)
        self._init_actor(action_dim)
        self.register_buffer("_offline_update_step", torch.zeros((), dtype=torch.long))
        self.to(config.device)

    def get_optim_params(self) -> list[dict]:
        return [
            {
                "params": [
                    p
                    for n, p in self.actor.named_parameters()
                    if not n.startswith("encoder") or not self.shared_encoder
                ],
                "lr": self.config.actor_lr,
            },
            {
                "params": list(self.critic_ensemble.parameters()),
                "lr": self.config.critic_lr,
            },
        ]

    def reset(self):
        pass

    def update(self):
        self.update_target_networks()
        self._offline_update_step.add_(1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Deterministic action selection: return policy mean."""
        _, _, mean_actions = self.actor(batch, None)
        return mean_actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("TwinRLPolicy returns single actions, not chunks.")

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic"] | None = None,
        reduction: str = "mean",
    ) -> dict[str, Tensor] | tuple[Tensor, dict[str, Tensor]]:
        """Compute loss for either the actor or critic, or both.

        If model is None, it returns (loss, output_dict) for compatibility with lerobot_train.py.
        Otherwise, it returns a dict for compatibility with learner.py.

        For "actor":
            batch must contain observations, ACTION (demo actions for BC term),
            and optionally "observation_feature".
        For "critic":
            batch must also contain "reward", next observations, "done",
            and optionally "complementary_info" with "mc_returns".
        """
        # --- Batch mapping ---
        # Handle flat batches from lerobot_train.py or nested batches from learner.py
        if "state" in batch and isinstance(batch["state"], dict):
            observations = batch["state"]
        else:
            # Assume flat batch; filter keys starting with 'observation.'
            observations = {k: v for k, v in batch.items() if k.startswith("observation.")}

        obs_features: Tensor | None = batch.get("observation_feature")

        train_batch = self._prepare_training_batch(batch, observations, obs_features)

        # Handle specific model requests (learner.py style)
        if model == "critic":
            if not train_batch["has_critic_data"]:
                raise ValueError("TwinRL critic update requires reward and next observations.")

            _, critic_info = self.calql_critic_loss_fn(batch, train_batch)
            return critic_info

        if model == "actor":
            _, actor_info = self.policy_loss_fn(batch, train_batch)
            return actor_info

        # --- Offline training (lerobot_train.py style) ---
        # Mirrors examples/train_offline.py:
        #   for _ in range(config.cta_ratio - 1): update_calql(..., {"critic"})
        #   update_calql(..., {"critic", "actor"})
        networks_to_update = self._lerobot_train_networks_to_update(train_batch)
        return self.update_calql(batch, train_batch, networks_to_update=networks_to_update)

    def _should_update_actor(self) -> bool:
        cta_ratio = max(1, self.config.cta_ratio)
        return int(self._offline_update_step.item()) % cta_ratio == cta_ratio - 1

    def _lerobot_train_networks_to_update(self, train_batch: dict) -> frozenset[str]:
        train_critic_networks_to_update = frozenset({"critic"})
        train_actor_networks_to_update = frozenset({"actor"})
        train_networks_to_update = frozenset({"critic", "actor"})

        if not train_batch["has_critic_data"]:
            return train_actor_networks_to_update
        if self._should_update_actor():
            return train_networks_to_update
        return train_critic_networks_to_update

    def policy_loss_fn(self, batch: dict, train_batch: dict) -> tuple[Tensor, dict[str, Tensor]]:
        actor_loss = self.compute_loss_actor(
            observations=train_batch["actor_observations"],
            demo_actions=train_batch["actions"],
            obs_features=train_batch["obs_features"],
        )
        return actor_loss, {"loss_actor": actor_loss}

    def critic_loss_fn(self, batch: dict, train_batch: dict) -> tuple[Tensor, dict[str, Tensor]]:
        critic_loss, info = self.compute_loss_critic(
            observations=train_batch["critic_observations"],
            actions=train_batch["actions"],
            rewards=train_batch["rewards"],
            next_observations=train_batch["next_observations"],
            done=train_batch["done"],
            mc_returns=None,
            obs_features=train_batch["obs_features"],
            next_obs_features=batch.get("next_observation_feature"),
            use_calql=False,
        )
        info["loss_critic"] = critic_loss
        return critic_loss, info

    def calql_critic_loss_fn(self, batch: dict, train_batch: dict) -> tuple[Tensor, dict[str, Tensor]]:
        critic_loss, info = self.compute_loss_critic(
            observations=train_batch["critic_observations"],
            actions=train_batch["actions"],
            rewards=train_batch["rewards"],
            next_observations=train_batch["next_observations"],
            done=train_batch["done"],
            mc_returns=self._get_mc_returns(batch),
            obs_features=train_batch["obs_features"],
            next_obs_features=batch.get("next_observation_feature"),
        )
        info["loss_critic"] = critic_loss
        return critic_loss, info

    def calql_loss_fns(
        self, batch: dict, train_batch: dict
    ) -> dict[str, Callable[[], tuple[Tensor, dict[str, Tensor]]]]:
        return {
            "actor": partial(self.policy_loss_fn, batch, train_batch),
            "critic": partial(self.calql_critic_loss_fn, batch, train_batch),
        }

    def loss_fns(
        self, batch: dict, train_batch: dict
    ) -> dict[str, Callable[[], tuple[Tensor, dict[str, Tensor]]]]:
        return {
            "actor": partial(self.policy_loss_fn, batch, train_batch),
            "critic": partial(self.critic_loss_fn, batch, train_batch),
        }

    def update_calql(
        self,
        batch: dict,
        train_batch: dict,
        *,
        networks_to_update: frozenset[str] = frozenset({"actor", "critic"}),
    ) -> tuple[Tensor, dict[str, Tensor]]:
        calql_loss_fns = self.calql_loss_fns(batch, train_batch)
        if not networks_to_update.issubset(calql_loss_fns.keys()):
            raise ValueError(f"Invalid gradient steps: {networks_to_update}")

        # Only compute gradients for specified steps. The generic LeRobot
        # trainer performs the backward call after this function returns.
        print(f"  [DEBUG] update_calql: networks_to_update={list(networks_to_update)}")
        loss_dict = {}
        total_loss = None
        for name in ("critic", "actor"):
            if name not in networks_to_update:
                continue

            loss, info = calql_loss_fns[name]()
            loss_dict.update(info)
            total_loss = loss if total_loss is None else total_loss + loss

        if total_loss is None:
            total_loss = train_batch["actions"].sum() * 0.0

        loss_dict["loss"] = total_loss
        return total_loss, loss_dict

    def _prepare_training_batch(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        observations: dict[str, Tensor],
        obs_features: Tensor | None,
    ) -> dict:
        # LeRobotDataset with delta_timestamps returns (b, horizon, ...).
        has_sequence = False
        horizon = 1
        for key, value in observations.items():
            if key.startswith("observation.image") and value.ndim == 5:
                has_sequence = True
                horizon = value.shape[1]
                break
            if key.startswith("observation.state") and value.ndim == 3:
                has_sequence = True
                horizon = value.shape[1]
                break

        if has_sequence and horizon == 2:
            actor_observations = {key: value[:, 0] for key, value in observations.items()}
            next_obs_from_seq = {key: value[:, 1] for key, value in observations.items()}
        elif has_sequence and horizon == 1:
            actor_observations = {key: value.squeeze(1) for key, value in observations.items()}
            next_obs_from_seq = None
        elif has_sequence:
            actor_observations = observations
            next_obs_from_seq = None
        else:
            actor_observations = observations
            next_obs_from_seq = None

        actions = batch[ACTION]
        if actions.ndim == 3 and actions.shape[1] == 1:
            actions = actions.squeeze(1)

        next_observations = None
        if next_obs_from_seq is not None:
            next_observations = next_obs_from_seq
        elif "next_state" in batch and isinstance(batch["next_state"], dict):
            next_observations = batch["next_state"]
        elif any(key.startswith("next.observation.") for key in batch):
            next_observations = {
                key.replace("next.", ""): (
                    value.squeeze(1) if value.ndim == 5 and value.shape[1] == 1 else value
                )
                for key, value in batch.items()
                if key.startswith("next.observation.")
            }

        rewards = batch.get("reward", batch.get("next.reward"))
        if rewards is not None and rewards.ndim == 2 and rewards.shape[1] == 1:
            rewards = rewards.squeeze(1)

        done = None
        if rewards is not None:
            done = batch.get("done", batch.get("next.done", torch.zeros_like(rewards)))
            if done.ndim == 2 and done.shape[1] == 1:
                done = done.squeeze(1)

        critic_observations = actor_observations
        if has_sequence and horizon > 2:
            critic_observations = {key: value[:, -1] for key, value in actor_observations.items()}

        return {
            "actions": actions,
            "actor_observations": actor_observations,
            "critic_observations": critic_observations,
            "done": done,
            "has_critic_data": rewards is not None and next_observations is not None,
            "next_observations": next_observations,
            "obs_features": obs_features,
            "rewards": rewards,
        }

    # ------------------------------------------------------------------
    # Actor loss: β * L_IL + η * L_Q  (paper Eq. 6)
    # ------------------------------------------------------------------

    def compute_loss_actor(
        self,
        observations: dict[str, Tensor],
        demo_actions: Tensor,
        obs_features: Tensor | None = None,
    ) -> Tensor:
        """Joint BC + RL actor loss.

        ConRFT path (use_consistency_policy=True):
            IL  = Karras-weighted consistency reconstruction loss (matches JAX policy_loss_fn)
            Q   = -mean_Q on single-step denoised action

        Gaussian path (use_consistency_policy=False):
            IL  = MSE(policy_mean, demo_action)
            Q   = -mean_Q on reparameterized sample
        """
        if self.config.use_consistency_policy:
            b, device = demo_actions.shape[0], demo_actions.device

            # Octo encoder is frozen; detach matches JAX stop_gradient=True
            obs_enc = self.actor.encoder(observations, obs_features).detach()

            # --- ConRFT recon loss (JAX policy_loss_fn lines 327-345) ---
            indices = torch.randint(0, self.config.num_scales - 1, (b,), device=device)
            sigma = self.actor.karras_sigmas[indices]  # (b,)
            noise = torch.randn_like(demo_actions)
            x_t = demo_actions + noise * sigma.unsqueeze(-1)
            distiller = self.actor.base_network(x_t, sigma, obs_enc)
            snrs = sigma**-2
            weights = snrs + 1.0 / self.config.sigma_data**2  # Karras weighting
            il_loss = ((distiller - demo_actions) ** 2).mean(dim=-1).mul(weights).mean()

            # --- Q-gradient: single-step denoised action (JAX lines 347-353) ---
            x_t = torch.randn(b, demo_actions.shape[-1], device=device) * self.config.sigma_max
            sigma_top = self.actor.karras_sigmas[0].expand(b)
            new_actions = self.actor.base_network(x_t, sigma_top, obs_enc).clamp(-1, 1)
            with self._critic_grad_disabled():
                q_values = self._critic_forward(observations, new_actions, obs_features)
            q_loss = -q_values.mean(dim=0).mean()
        else:
            sampled_actions, _, mean_actions = self.actor(observations, obs_features)
            il_loss = torch.nn.functional.mse_loss(mean_actions, demo_actions)
            with self._critic_grad_disabled():
                q_values = self._critic_forward(observations, sampled_actions, obs_features)
            q_loss = -q_values.mean(dim=0).mean()

        return self.config.bc_weight * il_loss + self.config.q_weight * q_loss

    # ------------------------------------------------------------------
    # Critic loss: L_TD + α * L_CQL  (Cal-QL from official TwinRL code)
    # ------------------------------------------------------------------

    def compute_loss_critic(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        rewards: Tensor,
        next_observations: dict[str, Tensor],
        done: Tensor,
        mc_returns: Tensor | None = None,
        obs_features: Tensor | None = None,
        next_obs_features: Tensor | None = None,
        use_calql: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # --- TD target (standard Bellman) ---
        with torch.no_grad():
            next_actions, _, _ = self.actor(next_observations, next_obs_features)
            next_q = self._critic_forward(next_observations, next_actions, next_obs_features, target=True)
            if self.config.num_subsample_critics is not None:
                idx = torch.randperm(self.config.num_critics)[: self.config.num_subsample_critics]
                next_q = next_q[idx]
            min_next_q = next_q.min(dim=0)[0]
            td_target = rewards + (1 - done) * self.config.discount * min_next_q

        q_preds = self._critic_forward(observations, actions, obs_features)
        td_target_exp = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        td_loss = torch.nn.functional.mse_loss(q_preds, td_target_exp, reduction="mean")

        info = {
            "td_loss": td_loss,
            "rewards": rewards.mean(),
            "target_qs": td_target.mean(),
            "predicted_qs": q_preds.mean(),
        }

        if not use_calql or not self.config.use_calql or self.config.cql_alpha == 0.0:
            return td_loss, info

        # --- CQL penalty with optional Cal-QL MC lower bound ---
        cql_diff = self._compute_cql_loss(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            mc_returns=mc_returns,
            obs_features=obs_features,
            next_obs_features=next_obs_features,
        )
        cql_loss = cql_diff.clamp(self.config.cql_clip_diff_min, self.config.cql_clip_diff_max).mean()

        info["cql_loss"] = cql_loss
        info["cql_diff"] = cql_diff.mean()

        return td_loss + self.config.cql_alpha * cql_loss, info

    def _compute_cql_loss(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        next_observations: dict[str, Tensor],
        mc_returns: Tensor | None,
        obs_features: Tensor | None,
        next_obs_features: Tensor | None,
    ) -> Tensor:
        """Cal-QL: logsumexp over OOD actions - Q(s, a_demo), lower-bounded by MC returns."""
        batch_size, action_dim = actions.shape
        n = self.config.cql_n_actions
        device = actions.device

        # Sample OOD actions: uniform random + policy at s + policy at s'
        random_actions = torch.rand(batch_size, n, action_dim, device=device) * 2 - 1
        with torch.no_grad():
            policy_actions_s, _, _ = self.actor(observations, obs_features, repeat=n)
            policy_actions_s_next, _, _ = self.actor(next_observations, next_obs_features, repeat=n)

        # Concatenate: (b, 3*n, action_dim)  — order: [random, current, next]
        all_actions = torch.cat([random_actions, policy_actions_s, policy_actions_s_next], dim=1)

        # Q values for all sampled actions: (num_critics, b, 3*n)
        q_sampled = self._critic_forward_multiple(observations, all_actions, obs_features)

        # Q on demo actions: (num_critics, b)
        q_demo = self._critic_forward(observations, actions, obs_features)

        # Cal-QL: clip sampled Q values to be >= MC return lower bound
        if mc_returns is not None:
            mc_lb = mc_returns.view(1, batch_size, 1).expand_as(q_sampled)
            q_sampled = torch.maximum(q_sampled, mc_lb)

        # Append q_demo as an extra column, apply log(K) importance correction, then logsumexp
        # Matches official: cql_q_samples = concat([ood, q_pred], axis=-1); subtract log(K)*temp
        k = q_sampled.shape[-1] + 1  # 3n + 1
        q_extended = torch.cat([q_sampled, q_demo.unsqueeze(-1)], dim=-1)
        q_extended = q_extended - math.log(k) * self.config.cql_temp
        ood_values = (
            torch.logsumexp(q_extended / self.config.cql_temp, dim=-1) * self.config.cql_temp
        )  # (num_critics, b)

        cql_diff = ood_values - q_demo  # (num_critics, b)
        return cql_diff

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def update_target_networks(self):
        tau = self.config.critic_target_update_weight
        for target_p, p in zip(
            self.critic_target.parameters(), self.critic_ensemble.parameters(), strict=True
        ):
            target_p.data.copy_(p.data * tau + target_p.data * (1.0 - tau))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _critic_grad_disabled(self):
        params = list(self.critic_ensemble.parameters())
        previous = [param.requires_grad for param in params]
        for param in params:
            param.requires_grad_(False)
        try:
            yield
        finally:
            for param, requires_grad in zip(params, previous, strict=True):
                param.requires_grad_(requires_grad)

    def _critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        obs_features: Tensor | None = None,
        target: bool = False,
    ) -> Tensor:
        critics = self.critic_target if target else self.critic_ensemble
        return critics(observations, actions, obs_features)

    def _critic_forward_multiple(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        obs_features: Tensor | None = None,
    ) -> Tensor:
        """Evaluate critic for a batch of actions per state.

        actions: (b, n_actions, action_dim)
        Returns: (num_critics, b, n_actions)
        """
        b, n, _ = actions.shape
        # Flatten to (b*n, action_dim) and tile observations
        flat_actions = actions.reshape(b * n, -1)
        flat_obs = {
            k: v.unsqueeze(1).expand(-1, n, *v.shape[1:]).reshape(b * n, *v.shape[1:])
            for k, v in observations.items()
        }
        flat_features = None
        if obs_features is not None:
            flat_features = (
                obs_features.unsqueeze(1)
                .expand(-1, n, *obs_features.shape[1:])
                .reshape(b * n, *obs_features.shape[1:])
            )

        q_flat = self.critic_ensemble(flat_obs, flat_actions, flat_features)  # (num_critics, b*n)
        return q_flat.reshape(q_flat.shape[0], b, n)

    @staticmethod
    def _get_mc_returns(batch: dict) -> Tensor | None:
        comp = batch.get("complementary_info")
        if comp is not None:
            return comp.get("mc_returns")
        return None

    # ------------------------------------------------------------------
    # Initialisation helpers (reuse SAC modules)
    # ------------------------------------------------------------------

    def _init_encoders(self):
        self.encoder_critic = TwinRLObservationEncoder(self.config)
        if self.config.actor_encoder_type == "octo":
            # Octo actor encoder is always separate from the SAC critic encoder
            self.encoder_actor = OctoActorEncoder(self.config)
            self.shared_encoder = False
        else:
            self.shared_encoder = self.config.shared_encoder
            self.encoder_actor = (
                self.encoder_critic if self.shared_encoder else TwinRLObservationEncoder(self.config)
            )

    def _init_critics(self, action_dim: int):
        """Build shared-final critic ensemble matching JAX parity."""
        hidden_dims = self.config.critic_network_kwargs.hidden_dims
        init_final = getattr(self.config.critic_network_kwargs, "init_final", None)

        self.critic_ensemble = TwinRLCriticEnsemble(
            encoder=self.encoder_critic,
            ensemble_size=self.config.num_critics,
            input_dim=self.encoder_critic.output_dim + action_dim,
            hidden_dims=hidden_dims,
            init_final=init_final,
        )
        self.critic_target = TwinRLCriticEnsemble(
            encoder=self.encoder_critic,
            ensemble_size=self.config.num_critics,
            input_dim=self.encoder_critic.output_dim + action_dim,
            hidden_dims=hidden_dims,
            init_final=init_final,
        )
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

    def _init_actor(self, action_dim: int):
        if self.config.use_consistency_policy:
            self.actor = ConsistencyActorHead(
                encoder=self.encoder_actor,
                obs_enc_dim=self.encoder_actor.output_dim,
                action_dim=action_dim,
                config=self.config,
            )
        else:
            self.actor = Policy(
                encoder=self.encoder_actor,
                network=TwinRLMLP(
                    input_dim=self.encoder_actor.output_dim,
                    **asdict(self.config.actor_network_kwargs),
                ),
                action_dim=action_dim,
                encoder_is_shared=self.shared_encoder,
                **asdict(self.config.policy_kwargs),
            )
