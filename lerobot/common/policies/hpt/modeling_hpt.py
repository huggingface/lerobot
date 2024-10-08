#!/usr/bin/env python

# Copyright 2024 Lirui Wang and The HuggingFace Inc. team. All rights reserved.
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
"""Heterogeneous Pre-trained Transformer Policy

As per Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers (https://liruiw.github.io/hpt/).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

from collections import deque
from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from einops import rearrange, repeat
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, einsum, nn

from lerobot.common.policies.hpt.configuration_hpt import HPTConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import get_device_from_parameters, populate_queues

LOSS = partial(F.smooth_l1_loss, beta=0.05)
INIT_CONST = 0.02
_LAYER_NORM = partial(nn.LayerNorm, eps=1e-6)


class HPTPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "hpt"],
):
    """
    Heterogeneous Pre-trained Transformer Policy as per Scaling Proprioceptive-Visual Learning
    with Heterogeneous Pre-trained Transformers  (paper: https://liruiw.github.io/hpt/, code: https://github.com/liruiw/HPT-Transfer)
    """

    name = "hpt"

    def __init__(
        self,
        config: HPTConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = HPTConfig()

        self.config: HPTConfig = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self.model = HPT(config)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {}
        for key in self.config.output_shapes:
            self._queues[key] = deque(maxlen=self.config.n_action_steps)
        for key in self.config.input_shapes:
            self._queues[key] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        batch = self.normalize_inputs(batch)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.model.generate_actions(batch)[:, : self.config.n_action_steps]
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        loss = self.model.compute_loss(batch)
        loss_dict = {"loss": loss}
        return loss_dict


class HPT(nn.Module):
    """Heterogeneous Pre-trained Transformer: The underlying neural network for HPTPolicy.

    Note: In this code we use the terms `stem`, 'trunk', `head`. The meanings are as follows.
        -  The stem, consisting of a proprioception tokenizer and a vision tokenizer, maps the
          vision and proprioception observations of different embodiments to a fixed number (e.g. 16) of tokens.
        -  The shared trunk, which is a Transformer, maps the concatenated tokens into shared representations.
        -  The head then maps the processed tokens to actions in different downstream tasks.
        For a specific embodiment, one stem/head pair is activated.

    ┌─────────────────────────────────────────────────────┐
    |                         Outputs                     |
    |                            ▲                        |
    |     ┌───────────┐   ┌───────────┐   ┌───────────┐   |
    |     |   Head 1  |   |   Head 2  |   |   Head 3  |   |
    |     └───────▲───┘   └───────▲───┘   └─────▲─────┘   |
    |             │               │             │         |
    |            ┌─────────────────────────────────┐      |
    |            |          Trunk Transformer.     │      |
    |            └────────▲──────────▲─────────────┘      |
    |                     │          │                    |
    |           ┌─────────┴──────┬───┴────────────┐       |
    |           │                │                │       |
    |   ┌───────┴────┐   ┌─────┴─────┐   ┌────┴──────┐    |
    |   │  Stem.     │   │  Stem.    │   │  Stem.    │    |
    |   │  Encoder 1 │   │  Encoder 2│   │  Encoder 3│    |
    |   │  ┌─────────┐   │ ┌─────────┐   │ ┌─────────┐    |
    |   │  │image emb│   │ │image emb│   │ │image emb│    |
    |   │  │state emb│   │ │state emb│   │ │state emb│    |
    |   └──┴─────────┘   └─┴─────────┘   └─┴─────────┘    |
    |                                                     |
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, config: HPTConfig):
        super().__init__()
        self.config = config
        self.use_robot_state = "observation.state" in config.input_shapes
        self.use_images = any(k.startswith("observation.image") for k in config.input_shapes)
        self.embed_dim = config.embed_dim

        self.trunk = self._create_policy_trunk(config.embed_dim, config.num_blocks, config.num_heads)
        self.stems = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        self.encoders = nn.ModuleDict()
        self.domains = []
        self.n_obs_steps = config.n_obs_steps
        self.action_chunk_size = config.action_chunk_size
        self.token_postprocessing = config.token_postprocessing

        # initialize modules.
        if "image" in config.modalities and not self.config.freeze_encoders:
            self.init_encoders("image", ResNet(resnet_model=self.config.vision_backbone))
        self.init_domain_stem(self.config.domain_name)
        self.init_domain_head(self.config.domain_name)

    def _init_weights(self, m: nn.Module):
        """
        Weight initialization for transformer
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_encoders(self, modality: str, encoder: nn.Module):
        """
        Add image/language encoders into the policy parameters in the case of joint finetuning
        """
        self.encoders[modality] = encoder
        self.encoders = nn.ModuleDict(self.encoders)

    def init_domain_stem(self, domain_name: str):
        """
        Initialize an observation stem for each domain
        """
        self.stem_spec = self.config
        self.modalities = self.config.modalities
        for modality in self.modalities:
            self.stems[domain_name + "_" + modality] = MLPStem(
                input_dim=getattr(self.config, modality + "_input_dim"),
                output_dim=getattr(self.config, modality + "_output_dim"),
                widths=getattr(self.config, modality + "_widths"),
                num_of_copy=getattr(self.config, modality + "_num_of_copy"),
            )
            self.stems[domain_name + "_" + modality].init_cross_attn(self.config, modality)

    def init_domain_head(self, domain_name: str):
        """initialize an action head for each domain, along with normalizer"""
        self.head_spec = self.config
        self.action_chunk_size = self.config.action_chunk_size
        self.domains.append(domain_name)
        if self.config.head_architecture == "diffusion":
            self.heads[domain_name] = DiffusionHead(
                config=self.config,
                embed_dim=self.config.embed_dim,
                action_chunk_size=self.config.action_chunk_size,
                action_dim=self.config.head_action_dim,
            )

        elif self.config.head_architecture == "transformer":
            self.heads[domain_name] = TransformerHead(
                config=self.config,
                action_chunk_size=self.config.action_chunk_size,
            )


    def _create_policy_trunk(
        self,
        embed_dim: int = 1024,
        num_blocks: int = 24,
        num_heads: int = 16,
        drop_path: float = 0.0,
        weight_init_style: str = "pytorch",
    ):
        """create the shared representation for pretraining"""

        def instantiate_trunk(embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6) if pre_transformer_ln else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
                weight_init_style=weight_init_style,
            )

        trunk = {}
        trunk["trunk"] = instantiate_trunk(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=drop_path,
        )
        self.trunk = nn.ModuleDict(trunk)

        if len(self.config.load_pretrained) > 0:
            self.load_trunk(self.config.load_pretrained)

        return self.trunk

    def _reset_parameters(self):
        self.apply(self._init_weights)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the HPT to generate actions at test time. Assume inputs have been normalized.

        `batch` should have the following structure:
        {
            "observation.state" (optional): (B, state_dim) batch of robot states.

            "observation.images": (B, n_cameras, C, H, W) batch of images.
                AND/OR
            "observation.environment_state": (B, env_dim) batch of environment states.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
        """
        domain = self.config.domain_name
        features = self.forward_features(batch)
        action = self.heads[domain](features)
        return action

    def generate_actions(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the HPT to generate actions at test time. Assume inputs have been normalized."""
        domain = self.config.domain_name
        features = self.forward_features(batch)
        action = self.heads[domain](features)
        start = self.config.n_obs_steps - 1
        action = action[:, start:]

        return action

    def preprocess_tokens(self, domain: str, features: List[Tensor]) -> Tensor:
        """
        Shared modality layers and add modality tokens. Add positional and time embeddings.
        """
        tokens = torch.cat(features, dim=-2)
        position_tokens = self.get_position_embedding(tokens)
        tokens = tokens + position_tokens
        return tokens

    def get_position_embedding(self, feature: Tensor) -> Tensor:
        """
        Add positional embedding to the features
        """
        pos_embedding = get_sinusoid_encoding_table(0, feature.shape[1], self.embed_dim)
        pos_embedding = pos_embedding.repeat((1, 1, 1)).to(feature.device)
        return pos_embedding

    def postprocess_tokens(self, trunk_tokens: Tensor) -> Tensor:
        """
        Postprocesses the trunk tokens to obtain the final features.

        Args:
            trunk_tokens (Tensor): The trunk tokens of shape (N, L, D), where N is the batch size,
                                        L is the sequence length, and D is the token dimension.

        Returns:
            Tensor: The postprocessed tokens of shape (N, D), where N is the batch size and D is the
                          final feature dimension.
        """
        if self.token_postprocessing == "mean":
            return trunk_tokens.mean(dim=1)
        elif self.token_postprocessing == "no-op":
            return trunk_tokens

    def mapped_modality_keys(self, modality: str, data: dict[str, Tensor]) -> str | Tensor:
        """Select the data for the given modality"""
        selected_keys = []
        selected_data = []

        for k in sorted(data):  # maintain order
            if modality in k and k in self.config.input_shapes:
                selected_keys.append(k)
                selected_data.append(data[k])

        if len(selected_keys) == 0:
            raise ValueError(f"{modality=} not found in data keys")

        if modality == "image":
            data = torch.stack(selected_data, dim=-4)
            if data.ndim < 6:
                data = rearrange(
                    data, "b t c h w -> b t 1 c h w" if data.ndim == 5 else "b c h w -> b 1 1 c h w"
                )

        if modality == "state":
            data = torch.cat(selected_data, dim=-1)
            if data.ndim < 4:
                data = rearrange(data, "b t d -> b t 1 d" if data.ndim == 3 else "b d -> b 1 1 d")
        return modality, data

    def stem_process(self, domain: str, data: dict):
        """
        Pass through the stem to a fixed number of tokens.
        Args:
            data: dictionary of tensors of different modalities
        """
        feats = []

        for policy_modality in self.modalities:
            stem = self.stems[domain + "_" + policy_modality]
            modality, modality_data = self.mapped_modality_keys(policy_modality, data)

            # add time horizon and instance number
            if "image" in modality and "image" in self.encoders:
                modality_data = self.encoders["image"](modality_data)

            # positional embedding for observations
            data_shape = modality_data.shape
            data_horizon = data_shape[1]
            horizon = data_horizon

            if self.training and self.stem_spec.random_horizon_masking and data_horizon > 1:
                horizon = np.random.randint(1, data_horizon + 1)
                modality_data = modality_data[:, data_horizon - horizon : data_horizon]

            # data is N x T x M x ... x D where M is the # of instances for that sensor
            positional_embedding = get_sinusoid_encoding_table(
                0, horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]
            ).to(modality_data)
            positional_embedding = repeat(
                positional_embedding, "b h w -> (repeat b) h w", repeat=data_shape[0]
            )

            modality_data = modality_data + positional_embedding.view(modality_data.shape)
            stem_token = stem.compute_latent(modality_data)
            feats.append(stem_token)

        return feats

    def forward_features(self, data: dict) -> Tensor:
        """
        Compute the features for the given domain and data.
        Args:
            domain (str): The domain of the data.
            data (Tensor): The input data.
        """
        domain = self.config.domain_name

        # stem pass
        self.stem_tokens = self.stem_process(domain, data)

        # combine tokens
        self.trunk_tokens = self.preprocess_tokens(domain, self.stem_tokens)

        # trunk pass
        self.trunk_tokens = self.trunk["trunk"](self.trunk_tokens)

        # pooling the features
        return self.postprocess_tokens(self.trunk_tokens)

    def load_trunk(self, path: str):
        """load the trunk part of the model"""

        def module_mean_param(module):
            def maybe_mean(x):
                return float(torch.abs(x).mean()) if x is not None else 0

            max_data = np.mean([(maybe_mean(param.data)) for name, param in module.named_parameters()])
            return max_data

        path = "liruiw/hpt-" + path
        print(f"Loading trunk from {path}")
        import huggingface_hub

        download_path = huggingface_hub.snapshot_download(path)
        self.trunk.load_state_dict(torch.load(download_path + "/trunk.pth"))

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute the loss for the training loop forward pass."""
        domain = self.config.domain_name
        features = self.forward_features(batch)
        loss = self.heads[domain].compute_loss(features, batch)
        return loss

class DiffusionHead(nn.Module):
    """Diffusion based policy head based on the diffusion implementation"""

    def __init__(self, config, embed_dim: int, action_chunk_size: int, action_dim: int) -> None:
        super().__init__()
        from ..diffusion.modeling_diffusion import DiffusionConditionalUnet1d, _make_noise_scheduler

        self.model = DiffusionConditionalUnet1d(config=config, global_cond_dim=embed_dim)
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    def conditional_sample(
        self,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """
        Perform conditional sampling using the diffusion process.
        """
        model = self.model
        scheduler = self.noise_scheduler
        device = get_device_from_parameters(model)
        trajectory = torch.randn(
            size=(len(global_cond), self.action_chunk_size, self.action_dim),
            dtype=global_cond.dtype,
            device=device,
            generator=generator,
        )

        global_cond = global_cond.to(device)
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            model_output = model(
                trajectory,
                torch.full(trajectory.shape[:1], t, dtype=torch.long, device=trajectory.device),
                global_cond=global_cond,
            )
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator).prev_sample
        return trajectory.transpose(0,1)

    def forward(self, global_cond: Tensor):
        return self.conditional_sample(global_cond)

    def compute_loss(self, global_cond: Tensor, data: Tensor) -> Tensor:
        trajectory = data["action"].reshape(
            (len(global_cond), self.action_chunk_size, self.action_dim)
        )  # Reshape the action tensor
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (len(trajectory),), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)
        target = noise
        return F.mse_loss(pred, target)


class TransformerHead(nn.Module):
    def __init__(
        self,
        config,
        action_chunk_size: int = 4,
    ) -> None:
        """
        Transformer decoder similar to ACT head.
        """
        super().__init__()
        from ..act.modeling_act import ACTDecoder

        self.config = config
        self.action_chunk_size = action_chunk_size
        self.decoder = ACTDecoder(config)
        self.tokens = nn.Parameter(torch.randn(action_chunk_size, self.config.embed_dim) * INIT_CONST)
        self.head_mlp = nn.Linear(self.config.embed_dim, self.config.head_action_dim)

    def forward(self, context: Tensor) -> Tensor:
        """
        context: (B, input_dim)
        """
        decoder_in = torch.zeros(
            (self.tokens.shape[0], len(context), self.config.embed_dim),
            dtype=context.dtype,
            device=context.device,
        )
        if len(context.shape) == 2:
            context = context.unsqueeze(1)

        decoder_out = self.decoder(
            decoder_in,  # [100, 8, 512]
            context.transpose(0, 1),  #  [1, 8, 512]
            decoder_pos_embed=self.tokens.unsqueeze(1),  #  [100, 1, 512]
        )
        out = self.head_mlp(decoder_out).transpose(0, 1).contiguous()
        return out

    def compute_loss(self, x: Tensor, target: dict) -> Tensor:
        target_action = target["action"]
        pred_action = self(x).view(target_action.shape)
        return F.l1_loss(pred_action, target_action)


class PolicyStem(nn.Module):
    """policy stem"""

    def __init__(self, **kwargs):
        super().__init__()

    def init_cross_attn(self, stem_spec, modality: str):
        """initialize cross attention module and the learnable tokens"""
        token_num = getattr(stem_spec, modality + "_crossattn_latent")
        self.tokens = nn.Parameter(torch.randn(1, token_num, stem_spec.embed_dim) * INIT_CONST)

        self.cross_attention = CrossAttention(
            stem_spec.embed_dim,
            heads=stem_spec.crossattn_heads,
            dim_head=stem_spec.crossattn_dim_head,
            dropout=stem_spec.crossattn_modality_dropout,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_latent(self, x: Tensor) -> Tensor:
        """
        Computes the latent representations of input data by attention.

        Args:
            Input tensor with shape [32, 3, 1, 49, 512] representing the batch size,
            horizon, instance (e.g. num of views), number of features, and feature dimensions respectively.

        Returns:
            Output tensor with latent tokens, shape [32, 16, 128], where 16 is the number
            of tokens and 128 is the dimensionality of each token.

        Examples for vision features from ResNet:
        >>> x = np.random.randn(32, 3, 1, 49, 512)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)

        Examples for proprioceptive features:
        >>> x = np.random.randn(32, 3, 1, 7)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)
        """
        # Initial reshape to adapt to token dimensions
        stem_feat = self(x)  # (32, 3, 1, 49, 128)
        stem_feat = stem_feat.reshape(stem_feat.shape[0], -1, stem_feat.shape[-1])  # (32, 147, 128)

        # Replicating tokens for each item in the batch and computing cross-attention
        stem_tokens = self.tokens.repeat(len(stem_feat), 1, 1)  # (32, 16, 128)
        stem_tokens = self.cross_attention(stem_tokens, stem_feat)  # (32, 16, 128)
        return stem_tokens


class MLPStem(PolicyStem):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: Tuple[int] = (512, 512),
        tanh_end: bool = False,
        ln: bool = True,
        num_of_copy: int = 1,
    ) -> None:
        """MLP Stem class"""
        super().__init__()
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self.num_of_copy = num_of_copy
        if self.num_of_copy > 1:
            self.net = nn.ModuleList([nn.Sequential(*modules) for _ in range(num_of_copy)])

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W]
        Returns:
            Flatten tensor with shape [B, M, 512]
        """
        if self.num_of_copy > 1:
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            y = torch.stack(out, dim=1)
        else:
            y = self.net(x)
        return y


class CrossAttention(nn.Module):
    """
    CrossAttention module used in the Perceiver IO model.

    Args:
        query_dim (int): The dimension of the query input.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout probability. Defaults to 0.0.
    """

    def __init__(self, query_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, context: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the CrossAttention module.

        Args:
            x (Tensor): The query input tensor.
            context (Tensor): The context input tensor.
            mask (Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor.
        """
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> (b h) n d", h=h)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=h)
        v = rearrange(v, "b n (h d) -> (b h) n d", h=h)

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            # fill in the masks with negative values
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BlockWithMasking(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_target: Callable,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        ffn_dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.attn = attn_target()
        self.norm_1 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
        )
        self.norm_2 = norm_layer(dim)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = x + self.attn(self.norm_1(x), attn_mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class MultiheadAttention(nn.MultiheadAttention):
    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        return super().forward(x, x, x, need_weights=False, attn_mask=attn_mask)[0]


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        attn_target: Callable,
        embed_dim: int,
        num_blocks: int,
        block: Callable = BlockWithMasking,
        pre_transformer_layer: Optional[Callable] = None,
        post_transformer_layer: Optional[Callable] = None,
        norm_layer: Callable = _LAYER_NORM,
        mlp_ratio: int = 4,
        ffn_dropout_rate: float = 0.0,
        weight_init_style: str = "pytorch",
    ):
        """
        Simple Transformer with the following features
        1. Supports masked attention
        2. Supports DropPath
        3. Supports LayerScale
        4. Supports Dropout in Attention and FFN
        5. Makes few assumptions about the input except that it is a Tensor
        """
        super().__init__()
        self.pre_transformer_layer = pre_transformer_layer
        self.blocks = nn.Sequential(
            *[
                block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    ffn_dropout_rate=ffn_dropout_rate,
                    norm_layer=norm_layer,
                )
                for i in range(num_blocks)
            ]
        )
        self.post_transformer_layer = post_transformer_layer
        self.weight_init_style = weight_init_style
        self.apply(self._init_weights)

    def forward(self, tokens: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Inputs
        - tokens: data of shape N x L x D (or L x N x D depending on the attention implementation)
        - attn: mask of shape L x L

        Output
        - x: data of shape N x L x D (or L x N x D depending on the attention implementation)
        """
        if self.pre_transformer_layer:
            tokens = self.pre_transformer_layer(tokens)
        for _, blk in enumerate(self.blocks):
            tokens = blk(tokens, attn_mask=attn_mask)
        if self.post_transformer_layer:
            tokens = self.post_transformer_layer(tokens)
        return tokens

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class EinOpsRearrange(nn.Module):
    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def forward(self, x: Tensor) -> Tensor:
        assert isinstance(x, Tensor)
        return rearrange(x, self.rearrange_expr, **self.kwargs)


def get_sinusoid_encoding_table(position_start: int, position_end: int, d_hid: int) -> Tensor:
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position: int):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(position_start, position_end)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ResNet(PolicyStem):
    def __init__(
        self,
        output_dim: int = 10,
        weights: str = "DEFAULT",
        resnet_model: str = "resnet18",
        num_of_copy: int = 1,
    ) -> None:
        """ResNet Encoder for Images"""
        super().__init__()
        pretrained_model = getattr(torchvision.models, resnet_model)(weights=weights)
        self.num_of_copy = num_of_copy
        self.net = nn.Sequential(*list(pretrained_model.children())[:-2])
        self.input = input
        self.out_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size,
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]
        """
        b, *_, h, w = x.shape
        x = x.view(len(x), -1, 3, h, w)
        x = x.view(-1, 3, h, w)
        # fixed image size
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        feat = self.net(x).view(b, 512, -1).transpose(1, 2).contiguous()
        return feat
