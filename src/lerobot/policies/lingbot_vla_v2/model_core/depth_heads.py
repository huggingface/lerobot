# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Alignment heads for the native-depth / DINO-video distillation branch.

Verbatim port of the upstream LingBot-VLA 2.0 resampler heads
(``lingbotvla/models/vla/vision_models/align_heads/{resampler,depth_head}.py``,
originally modified from open_flamingo's perceiver resampler). Only the classes
the released 6B "Native Depth" checkpoint actually instantiates are ported:

- ``FeedForward`` / ``PerceiverAttention`` / ``TaskTokenResampler``
- ``TaskTokenDepthHead`` (wraps a ``TaskTokenResampler`` as ``self.projector``)

Module/attribute names are load-bearing: the released checkpoint stores e.g.
``model.depth_align_head.projector.proj_in1.weight`` and
``model.future_video_align_head.projector.layers.0.0.to_q.weight`` (F32), and
the converter relies on exact name identity — do not rename anything here.
``Resampler``/``DepthHead`` (learned-query variants) and the ``ResamplerXL*``
family are unused by this branch and intentionally not ported.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, latent_length, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, latent_length, -1)

        return self.to_out(out)


class TaskTokenResampler(nn.Module):
    """Perceiver-style resampler whose queries come from the caller.

    Unlike ``Resampler`` there is no learned ``self.queries`` parameter — the
    256 alignment queries live on the model (``depth_align_embs`` etc.) so they
    can double as the task tokens pooled into the VLM prefix.
    """

    def __init__(
        self,
        dim_in=768,
        dim_mid=1024,
        dim_head=64,
        dim_out=1024,
        num_layers=8,
        num_queries=8,
        num_heads=16,
        ff_mult=4,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.proj_in1 = nn.Linear(dim_in, dim_mid)
        self.proj_in2 = nn.Linear(dim_in, dim_mid)
        self.proj_out = nn.Linear(dim_mid, dim_out)
        self.norm_out = nn.LayerNorm(dim_out)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim_mid, dim_head=dim_head, heads=num_heads),
                        FeedForward(dim=dim_mid, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, queries):
        queries = self.proj_in1(queries)
        x = self.proj_in2(x)

        for attn, ff in self.layers:
            queries = attn(x, queries) + queries
            queries = ff(queries) + queries

        queries = self.proj_out(queries)
        queries = self.norm_out(queries)
        return queries


class TaskTokenDepthHead(nn.Module):
    """Distillation head projecting (prefix tokens, alignment queries) -> teacher embedding space."""

    def __init__(
        self,
        proj_config=None,
        llm_hidden_size=4096,
        use_intermediate_depth=False,
    ):
        super().__init__()

        self.projector = TaskTokenResampler(
            dim_in=llm_hidden_size,
            dim_mid=llm_hidden_size,
            dim_head=proj_config["dim_head"],
            dim_out=proj_config["dim_out"],
            num_layers=proj_config["num_layers"],
            num_heads=proj_config["num_heads"],
            num_queries=proj_config["num_backbone_tokens"],
            ff_mult=proj_config["ff_mult"],
        )

    def forward(self, llm_feats, queries):
        queries = self.projector(llm_feats, queries)
        return queries
