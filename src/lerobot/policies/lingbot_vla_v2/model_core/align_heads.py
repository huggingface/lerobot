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

"""Dual-query distillation heads, ported from upstream
``lingbotvla/models/vla/vision_models/align_heads/{resampler,depth_head}.py``. Module names match the released
checkpoints.

The perceiver resampler is modified from open_flamingo
(https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py), licensed under the
MIT License:

    MIT License

    Copyright (c) 2023 Anas Awadalla, Irena Gao, Joshua Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu,
    Kalyani Marathe, Yonatan Bitton, Samir Gadre, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco,
    Mitchell Wortsman, Ludwig Schmidt.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import math

import torch
from torch import nn


def FeedForward(dim, mult=4):  # noqa: N802
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, length, _ = latents.shape
        q = self.to_q(latents)
        k, v = self.to_kv(torch.cat((x, latents), dim=-2)).chunk(2, dim=-1)
        q, k, v = (t.view(b, t.shape[1], self.heads, -1).transpose(1, 2) for t in (q, k, v))
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = (weight @ v).permute(0, 2, 1, 3).reshape(b, length, -1)
        return self.to_out(out)


class TaskTokenResampler(nn.Module):
    """Perceiver resampler whose queries are passed in (the model's ``*_align_embs``)."""

    def __init__(self, dim_in, dim_mid, dim_head, dim_out, num_layers, num_heads, ff_mult):
        super().__init__()
        self.proj_in1 = nn.Linear(dim_in, dim_mid)
        self.proj_in2 = nn.Linear(dim_in, dim_mid)
        self.proj_out = nn.Linear(dim_mid, dim_out)
        self.norm_out = nn.LayerNorm(dim_out)
        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    PerceiverAttention(dim=dim_mid, dim_head=dim_head, heads=num_heads),
                    FeedForward(dim=dim_mid, mult=ff_mult),
                ]
            )
            for _ in range(num_layers)
        )

    def forward(self, x, queries):
        queries = self.proj_in1(queries)
        x = self.proj_in2(x)
        for attn, ff in self.layers:
            queries = attn(x, queries) + queries
            queries = ff(queries) + queries
        return self.norm_out(self.proj_out(queries))


class TaskTokenDepthHead(nn.Module):
    """Projects (prefix hidden states, align queries) into a teacher's feature space."""

    def __init__(self, proj_config, llm_hidden_size):
        super().__init__()
        self.projector = TaskTokenResampler(
            dim_in=llm_hidden_size,
            dim_mid=llm_hidden_size,
            dim_head=proj_config["dim_head"],
            dim_out=proj_config["dim_out"],
            num_layers=proj_config["num_layers"],
            num_heads=proj_config["num_heads"],
            ff_mult=proj_config["ff_mult"],
        )

    def forward(self, llm_feats, queries):
        return self.projector(llm_feats, queries)
