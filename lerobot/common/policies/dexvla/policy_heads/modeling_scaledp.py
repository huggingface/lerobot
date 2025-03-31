#!/usr/bin/env python

# Copyright 2025 DexVLA Team and The HuggingFace Inc. team. All rights reserved.
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
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.checkpoint
from timm.models.vision_transformer import Mlp, use_fused_attn
from torch.jit import Final
from transformers.modeling_utils import PreTrainedModel

from .configuration_scaledp import ScaleDPPolicyConfig

_logger = logging.getLogger(__name__)


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = func.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            # attn = q @ k.transpose(-2, -1)
            # if attn_mask is not None:
            #     attn += attn_mask
            # attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)
            # x = attn @ v
            attn_scores = torch.matmul(q, k.transpose(-2, -1))

            # Add attention mask if provided
            if attn_mask is not None:
                attn_scores += attn_mask

            # Apply softmax to get attention weights (softmax is applied along the last dimension)
            attn_weights = func.softmax(attn_scores, dim=-1)

            # Dropout on attention weights (if dropout is used)
            attn_weights = self.attn_drop(attn_weights)

            # Apply attention weights to value tensor (V)
            x = torch.matmul(attn_weights, v)

        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


logger = logging.getLogger(__name__)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.bfloat16) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(dtype=torch.bfloat16)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core ScaleDP Model                                #
#################################################################################


class ScaleDPBlock(nn.Module):
    """
    A ScaleDP block with adaptive layer norm zero (adaLN-Zero) conScaleDPioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(
            6, dim=1
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask
        )  # norm, scale&shift, attn, scale,
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of ScaleDP.
    """

    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ScaleDP(PreTrainedModel):
    """
    Diffusion models with a Transformer backbone.
    """

    config_class = ScaleDPPolicyConfig

    def __init__(
        self,
        config: ScaleDPPolicyConfig,
    ):
        super().__init__(config)
        # compute number of tokens for main trunk and conScaleDPion encoder
        if config.n_obs_steps is None:
            config.n_obs_steps = config.prediction_horizon
        t = config.prediction_horizon
        t_cond = 1
        if not config.time_as_cond:
            t += 1
            t_cond -= 1
        obs_as_cond = config.cond_dim > 0
        if obs_as_cond:
            assert config.time_as_cond
            t_cond += config.n_obs_steps

        # self.combine = nn.Linear(cond_dim+state_dim, cond_dim)
        self.combine = nn.Sequential(
            nn.Linear(config.cond_dim + config.state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, config.cond_dim),
        )
        self.learn_sigma = config.learn_sigma
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim * 2 if config.learn_sigma else config.output_dim
        self.num_heads = config.num_heads

        self.x_embedder = nn.Linear(config.input_dim, config.n_emb)
        self.t_embedder = TimestepEmbedder(config.n_emb)
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(config.cond_dim, config.n_emb)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, config.prediction_horizon, config.n_emb))

        self.blocks = nn.ModuleList(
            [
                ScaleDPBlock(config.n_emb, config.num_heads, mlp_ratio=config.mlp_ratio)
                for _ in range(config.depth)
            ]
        )
        self.final_layer = FinalLayer(config.n_emb, output_dim=config.output_dim)
        # self.initialize_weights()
        # constants
        self.t = t
        self.t_cond = t_cond
        self.prediction_horizon = config.prediction_horizon
        self.time_as_cond = config.time_as_cond
        self.action_dim = config.output_dim
        self.obs_as_cond = obs_as_cond
        logger.info("number of parameters in ScaleDP: %e", sum(p.numel() for p in self.parameters()))

        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        self.num_inference_timesteps = config.num_inference_timesteps
        # self.proj_to_action = nn.Identity()
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,  # 100
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )
        self.num_queries = config.num_queries  # 16
        self.noise_samples = config.noise_samples  # 1
        # self.num_inference_timesteps = config.num_inference_timesteps # 100

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.cond_obs_emb.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.cond_obs_emb.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in ScaleDP blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the models into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, Attention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _p in m.named_parameters():
                fpn = "{}.{}".format(mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters {} made it into both decay/no_decay sets!".format(
            str(inter_params)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters {} were not separated into either decay/no_decay set!".format(
                str(param_dict.keys() - union_params),
            )
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, actions, hidden_states, states, is_pad):
        """
        Forward pass for the diffusion head.
        :param actions: target actions, shape [b, Ta, D] D:10 = 3+6+1
        :param hidden_states: hidden states from the llava_pythia, as the conScaleDPion for the diffusion, shape [b,Tokens, D] 8 1200 1024
        :param states: robot states, shape [b, D]
        :return: loss
        """
        if actions is not None:  # training time
            b = actions.size(0)
            actions = actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]
            num_noise_samples = self.noise_samples
            # sample noise to add to actions
            noise = torch.randn(
                [num_noise_samples] + list(actions.shape), device=actions.device, dtype=actions.dtype
            )  # num_noise, b, Ta, D(1, 2, 16, 14)
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (b,), device=actions.device
            ).long()

            timesteps, noise = timesteps.to(actions.device), noise.to(actions.device)

            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = torch.cat(
                [self.noise_scheduler.add_noise(actions, noise[i], timesteps) for i in range(len(noise))],
                dim=0,
            )  # [num_noise_samples * b, Ta, action_dim]

            noisy_actions = noisy_actions.to(dtype=actions.dtype)
            assert hidden_states.ndim == 3

            hidden_states = hidden_states.repeat(num_noise_samples, 1, 1)
            timesteps = timesteps.repeat(num_noise_samples)
            is_pad = is_pad.repeat(num_noise_samples, 1)
            states = states.repeat(num_noise_samples, 1)

            noise_pred = self.model_forward(
                noisy_actions, timesteps, global_cond=hidden_states, states=states
            )
            noise = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])
            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none")
            loss = (loss * ~is_pad.unsqueeze(-1)).mean()
            # loss_dict['loss'] = loss
            return {"loss": loss}
            # return loss
        else:  # inference time
            b = 1
            tp = self.num_queries
            action_dim = self.action_dim

            # initialize action from Gaussian noise
            noisy_action = torch.randn((b, tp, action_dim)).cuda()

            naction = noisy_action.to(dtype=hidden_states.dtype)
            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.model_forward(naction, k, global_cond=hidden_states, states=states)

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample

            return naction

    def model_forward(self, x, t, global_cond, states):
        """
        Forward pass of ScaleDP.
        x: (N, T, input_dim) noisy actions
        t: (N,) tensor of diffusion timesteps
        global_cond: (N, n_obs_steps, D) tensor of conScaleDPions: image embeddings
        """
        global_cond = global_cond.squeeze(1)
        global_cond = torch.cat([global_cond, states], dim=-1) if states is not None else global_cond
        global_cond = self.combine(global_cond)

        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        elif torch.is_tensor(t) and len(t.shape) == 0:
            t = t[None].to(x.device)
        t = t.expand(t.shape[0])

        x = self.x_embedder(x) + self.pos_embed.to(
            device=x.device, dtype=x.dtype
        )  # (N, T, D), where T = prediction_horizon
        t = self.t_embedder(t)  # (N, D)
        if self.obs_as_cond:
            global_cond = self.cond_obs_emb(global_cond)  # (N, D)
        # c = t + global_cond.sum(dim=1)  # (N, D)
        c = t + global_cond  # (N, D)
        for block in self.blocks:
            # x = block(x, c, attn_mask=self.mask)  # (N, T, D)
            x = block(x, c, attn_mask=None)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, output_dim)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   ScaleDP Configs                                  #
#################################################################################


def scaledp_h(**kwargs):
    return ScaleDP(depth=32, n_emb=1280, num_heads=16, **kwargs)


def scaledp_l(**kwargs):
    return ScaleDP(depth=24, n_emb=1024, num_heads=16, **kwargs)
