#!/usr/bin/env python

# Copyright 2024 Seungjae Lee and Yibin Wang and Haritheja Etukuru
# and H. Jin Kim and Nur Muhammad Mahi Shafiullah and Lerrel Pinto
# and The HuggingFace Inc. team. All rights reserved.
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

import math
from functools import partial
from math import ceil
from random import randrange
from typing import Callable

import torch
import torch.distributed as distributed
import torch.nn.functional as F  # noqa: N812
from einops import pack, rearrange, reduce, repeat, unpack
from torch import einsum, nn
from torch.cuda.amp import autocast
from torch.optim import Optimizer

from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig

# ruff: noqa: N806

"""
This file is part of a VQ-BeT that utilizes code from the following repositories:

    - Vector Quantize PyTorch code is licensed under the MIT License:
        Origianl source: https://github.com/lucidrains/vector-quantize-pytorch

    - nanoGPT part is an adaptation of Andrej Karpathy's nanoGPT implementation in PyTorch.
        Original source: https://github.com/karpathy/nanoGPT

We also made some changes to the original code to adapt it to our needs. The changes are described in the code below.
"""

"""
This is a part for nanoGPT that utilizes code from the following repository:

    - Andrej Karpathy's nanoGPT implementation in PyTorch.
        Original source: https://github.com/karpathy/nanoGPT

    - The nanoGPT code is licensed under the MIT License:

    MIT License

    Copyright (c) 2022 Andrej Karpathy

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

    - We've made some changes to the original code to adapt it to our needs.

        Changed variable names:
            - n_head -> gpt_n_head
            - n_embd -> gpt_hidden_dim
            - block_size -> gpt_block_size
            - n_layer -> gpt_n_layer


        class GPT(nn.Module):
            - removed unused functions `def generate`, `def estimate_mfu`, and `def from_pretrained`
            - changed the `configure_optimizers` to `def configure_parameters` and made it to return only the parameters of the model: we use an external optimizer in our training loop.
            - in the function `forward`, we removed target loss calculation parts, since it will be calculated in the training loop (after passing through bin prediction and offset prediction heads).

"""


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.gpt_hidden_dim % config.gpt_n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.gpt_hidden_dim, 3 * config.gpt_hidden_dim)
        # output projection
        self.c_proj = nn.Linear(config.gpt_hidden_dim, config.gpt_hidden_dim)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.gpt_block_size, config.gpt_block_size)).view(
                1, 1, config.gpt_block_size, config.gpt_block_size
            ),
        )
        self.gpt_n_head = config.gpt_n_head
        self.gpt_hidden_dim = config.gpt_hidden_dim

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (gpt_hidden_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.gpt_hidden_dim, dim=2)
        k = k.view(B, T, self.gpt_n_head, C // self.gpt_n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.gpt_n_head, C // self.gpt_n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.gpt_n_head, C // self.gpt_n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    # causual self-attention block for GPT
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.gpt_hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.gpt_hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.gpt_hidden_dim, 4 * config.gpt_hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.gpt_hidden_dim, config.gpt_hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    Original comments:
    Full definition of a GPT Language Model, all of it in this single file.
    References:
    1) the official GPT-2 TensorFlow implementation released by OpenAI:
    https://github.com/openai/gpt-2/blob/master/src/model.py
    2) huggingface/transformers PyTorch implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    """

    def __init__(self, config: VQBeTConfig):
        """
        GPT model gets hyperparameters from a config object. Please refer configuration_vqbet.py for more details.
        """
        super().__init__()
        assert config.gpt_output_dim is not None
        assert config.gpt_block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Linear(config.gpt_input_dim, config.gpt_hidden_dim),
                "wpe": nn.Embedding(config.gpt_block_size, config.gpt_hidden_dim),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.gpt_n_layer)]),
                "ln_f": nn.LayerNorm(config.gpt_hidden_dim),
            }
        )
        self.lm_head = nn.Linear(config.gpt_hidden_dim, config.gpt_output_dim, bias=False)
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.gpt_n_layer))

        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: {:.2f}M".format(n_params / 1e6))

    def forward(self, input, targets=None):
        device = input.device
        b, t, d = input.size()
        assert t <= self.config.gpt_block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.gpt_block_size}"
        )

        # positional encodings that are added to the input embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(input)  # token embeddings of shape (b, t, gpt_hidden_dim)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, gpt_hidden_dim)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def crop_block_size(self, gpt_block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert gpt_block_size <= self.config.gpt_block_size
        self.config.gpt_block_size = gpt_block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:gpt_block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :gpt_block_size, :gpt_block_size]

    def configure_parameters(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _p in m.named_parameters():
                fpn = "{}.{}".format(mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    # all biases will not be decayed
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

        decay = [param_dict[pn] for pn in sorted(decay)]
        no_decay = [param_dict[pn] for pn in sorted(no_decay)]
        # return the parameters that require weight decay, and the parameters that don't separately.
        return decay, no_decay


"""
This file is a part for Residual Vector Quantization that utilizes code from the following repository:

    - Phil Wang's vector-quantize-pytorch implementation in PyTorch.
        Origianl source: https://github.com/lucidrains/vector-quantize-pytorch

    - The vector-quantize-pytorch code is licensed under the MIT License:

        MIT License

        Copyright (c) 2020 Phil Wang

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

    - We've made some changes to the original code to adapt it to our needs.

        class ResidualVQ(nn.Module):
            - added `self.register_buffer('freeze_codebook', torch.tensor(False))` to the __init__ method:
                This enables the user to save an indicator whether the codebook is frozen or not.
            - changed the name of function `get_codes_from_indices` → `get_codebook_vector_from_indices`:
                This is to make the function name more descriptive.

        class VectorQuantize(nn.Module):
            - removed the `use_cosine_sim` and `layernorm_after_project_in` parameters from the __init__ method:
                These parameters are not used in the code.
            - changed the name of function `get_codes_from_indices` → `get_codebook_vector_from_indices`:
                This is to make the function name more descriptive.

"""


class ResidualVQ(nn.Module):
    """
    Residual VQ is composed of multiple VectorQuantize layers.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
        "Residual Vector Quantizer (a.k.a. multi-stage vector quantizer [36]) cascades Nq layers of VQ as follows. The unquantized input vector is
        passed through a first VQ and quantization residuals are computed. The residuals are then iteratively quantized by a sequence of additional
        Nq -1 vector quantizers, as described in Algorithm 1."


    self.project_in: function for projecting input to codebook dimension
    self.project_out: function for projecting codebook dimension to output dimension
    self.layers: nn.ModuleList of VectorQuantize layers that contains Nq layers of VQ as described in the paper.
    self.freeze_codebook: buffer to save an indicator whether the codebook is frozen or not. VQ-BeT will check this to determine whether to update the codebook or not.
    """

    def __init__(
        self,
        *,
        dim,
        num_quantizers,
        codebook_dim=None,
        shared_codebook=False,
        heads=1,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        accept_image_fmap=False,
        **kwargs,
    ):
        super().__init__()
        assert heads == 1, "residual vq is not compatible with multi-headed codes"
        codebook_dim = codebook_dim if (codebook_dim is not None) else dim
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.num_quantizers = num_quantizers

        self.accept_image_fmap = accept_image_fmap
        self.layers = nn.ModuleList(
            [
                VectorQuantize(
                    dim=codebook_dim, codebook_dim=codebook_dim, accept_image_fmap=accept_image_fmap, **kwargs
                )
                for _ in range(num_quantizers)
            ]
        )

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.register_buffer("freeze_codebook", torch.tensor(False))
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    @property
    def codebooks(self):
        codebooks = [layer._codebook.embed for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        codebooks = rearrange(codebooks, "q 1 c d -> q c d")
        return codebooks

    def get_codebook_vector_from_indices(self, indices):
        # this function will return the codes from all codebooks across layers corresponding to the indices
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], "b * q")

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0.0, (
                "quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations"
            )
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        # get ready for gathering

        codebooks = repeat(self.codebooks, "q c d -> q b c d", b=batch)
        gather_indices = repeat(indices, "b n q -> q b n d", d=codebooks.shape[-1])

        # take care of quantizer dropout

        mask = gather_indices == -1.0
        gather_indices = gather_indices.masked_fill(
            mask, 0
        )  # have it fetch a dummy code to be masked out later

        all_codes = codebooks.gather(2, gather_indices)  # gather all codes

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(mask, 0.0)

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        (all_codes,) = unpack(all_codes, ps, "q b * d")

        return all_codes

    def forward(self, x, indices=None, return_all_codes=False, sample_codebook_temp=None):
        """
        For given input tensor x, this function will return the quantized output, the indices of the quantized output, and the loss.
        First, the input tensor x is projected to the codebook dimension. Then, the input tensor x is passed through Nq layers of VectorQuantize.
        The residual value of each layer is fed to the next layer.
        """
        num_quant, quant_dropout_multiple_of, return_loss, device = (
            self.num_quantizers,
            self.quantize_dropout_multiple_of,
            (indices is not None),
            x.device,
        )

        x = self.project_in(x)

        assert not (self.accept_image_fmap and (indices is not None))

        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        if return_loss:
            assert not torch.any(indices == -1), (
                "some of the residual vq indices were dropped out. please use indices derived when the module is in eval mode to derive cross entropy loss"
            )
            ce_losses = []

        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices and loss

        if should_quantize_dropout:
            rand_quantize_dropout_index = randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    ceil((rand_quantize_dropout_index + 1) / quant_dropout_multiple_of)
                    * quant_dropout_multiple_of
                    - 1
                )

            null_indices_shape = (x.shape[0], *x.shape[-2:]) if self.accept_image_fmap else tuple(x.shape[:2])
            null_indices = torch.full(null_indices_shape, -1.0, device=device, dtype=torch.long)
            null_loss = torch.full((1,), 0.0, device=device, dtype=x.dtype)

        # go through the layers

        for quantizer_index, layer in enumerate(self.layers):
            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue

            layer_indices = None
            if return_loss:
                layer_indices = indices[..., quantizer_index]

            quantized, *rest = layer(
                residual,
                indices=layer_indices,
                sample_codebook_temp=sample_codebook_temp,
                freeze_codebook=self.freeze_codebook,
            )

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            if return_loss:
                ce_loss = rest[0]
                ce_losses.append(ce_loss)
                continue

            embed_indices, loss = rest

            all_indices.append(embed_indices)
            all_losses.append(loss)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # whether to early return the cross entropy loss

        if return_loss:
            return quantized_out, sum(ce_losses)

        # stack all losses and indices

        all_losses, all_indices = map(partial(torch.stack, dim=-1), (all_losses, all_indices))

        ret = (quantized_out, all_indices, all_losses)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers
            all_codes = self.get_codebook_vector_from_indices(all_indices)

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        return ret


class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim=None,
        heads=1,
        separate_codebook_per_head=False,
        decay=0.8,
        eps=1e-5,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        threshold_ema_dead_code=0,
        channel_last=True,
        accept_image_fmap=False,
        commitment_weight=1.0,
        commitment_use_cross_entropy_loss=False,
        orthogonal_reg_weight=0.0,
        orthogonal_reg_active_codes_only=False,
        orthogonal_reg_max_codes=None,
        stochastic_sample_codes=False,
        sample_codebook_temp=1.0,
        straight_through=False,
        reinmax=False,  # using reinmax for improved straight-through, assuming straight through helps at all
        sync_codebook=None,
        sync_affine_param=False,
        ema_update=True,
        learnable_codebook=False,
        in_place_codebook_optimizer: Callable[
            ..., Optimizer
        ] = None,  # Optimizer used to update the codebook embedding if using learnable_codebook
        affine_param=False,
        affine_param_batch_decay=0.99,
        affine_param_codebook_decay=0.9,
        sync_update_v=0.0,  # the v that controls optimistic vs pessimistic update for synchronous update rule (21) https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = codebook_dim if (codebook_dim is not None) else dim
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss  # whether to use cross entropy loss to codebook as commitment loss

        self.learnable_codebook = learnable_codebook

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        assert not (ema_update and learnable_codebook), "learnable codebook not compatible with EMA update"

        assert 0 <= sync_update_v <= 1.0
        assert not (sync_update_v > 0.0 and not learnable_codebook), "learnable codebook must be turned on"

        self.sync_update_v = sync_update_v

        gumbel_sample_fn = partial(
            gumbel_sample,
            stochastic=stochastic_sample_codes,
            reinmax=reinmax,
            straight_through=straight_through,
        )

        if sync_codebook is None:
            sync_codebook = distributed.is_initialized() and distributed.get_world_size() > 1

        codebook_kwargs = {
            "dim": codebook_dim,
            "num_codebooks": heads if separate_codebook_per_head else 1,
            "codebook_size": codebook_size,
            "kmeans_init": kmeans_init,
            "kmeans_iters": kmeans_iters,
            "sync_kmeans": sync_kmeans,
            "decay": decay,
            "eps": eps,
            "threshold_ema_dead_code": threshold_ema_dead_code,
            "use_ddp": sync_codebook,
            "learnable_codebook": has_codebook_orthogonal_loss or learnable_codebook,
            "sample_codebook_temp": sample_codebook_temp,
            "gumbel_sample": gumbel_sample_fn,
            "ema_update": ema_update,
        }

        if affine_param:
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param=True,
                sync_affine_param=sync_affine_param,
                affine_param_batch_decay=affine_param_batch_decay,
                affine_param_codebook_decay=affine_param_codebook_decay,
            )

        self._codebook = EuclideanCodebook(**codebook_kwargs)

        self.in_place_codebook_optimizer = (
            in_place_codebook_optimizer(self._codebook.parameters())
            if (in_place_codebook_optimizer is not None)
            else None
        )

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        codebook = self._codebook.embed

        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, "1 ... -> ...")

    @codebook.setter
    def codebook(self, codes):
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, "... -> 1 ...")

        self._codebook.embed.copy_(codes)

    def get_codebook_vector_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, "... h d -> ... (h d)")

        indices, ps = pack_one(indices, "b * h")
        indices = rearrange(indices, "b n h -> b h n")

        indices = repeat(indices, "b h n -> b h n d", d=codebook.shape[-1])
        codebook = repeat(codebook, "h n d -> b h n d", b=indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, "b h n d -> b n (h d)")
        codes = unpack_one(codes, ps, "b * d")
        return codes

    def forward(
        self,
        x,
        indices=None,
        mask=None,
        sample_codebook_temp=None,
        freeze_codebook=False,
    ):
        orig_input = x

        only_one = x.ndim == 2

        if only_one:
            assert mask is None
            x = rearrange(x, "b d -> b 1 d")

        shape, device, heads, is_multiheaded, _codebook_size, return_loss = (
            x.shape,
            x.device,
            self.heads,
            self.heads > 1,
            self.codebook_size,
            (indices is not None),
        )

        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = self.in_place_codebook_optimizer is not None

        # rearrange inputs

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, "b c h w -> b (h w) c")

        if need_transpose:
            x = rearrange(x, "b d n -> b n d")

        # project input

        x = self.project_in(x)

        # handle multi-headed separate codebooks

        if is_multiheaded:
            ein_rhs_eq = "h b n d" if self.separate_codebook_per_head else "1 (b h) n d"
            x = rearrange(x, f"b n (h d) -> {ein_rhs_eq}", h=heads)

        # l2norm for cosine sim, otherwise identity

        x = self._codebook.transform_input(x)

        # codebook forward kwargs

        codebook_forward_kwargs = {
            "sample_codebook_temp": sample_codebook_temp,
            "mask": mask,
            "freeze_codebook": freeze_codebook,
        }

        # quantize

        quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        # one step in-place update

        if should_inplace_optimize and self.training and not freeze_codebook:
            if mask is not None:
                loss = F.mse_loss(quantize, x.detach(), reduction="none")

                loss_mask = mask
                if is_multiheaded:
                    loss_mask = repeat(
                        mask,
                        "b n -> c (b h) n",
                        c=loss.shape[0],
                        h=loss.shape[1] // mask.shape[0],
                    )

                loss = loss[loss_mask].mean()

            else:
                loss = F.mse_loss(quantize, x.detach())

            loss.backward()
            self.in_place_codebook_optimizer.step()
            self.in_place_codebook_optimizer.zero_grad()

            # quantize again

            quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        if self.training:
            # determine code to use for commitment loss
            maybe_detach = torch.detach if not self.learnable_codebook or freeze_codebook else identity

            commit_quantize = maybe_detach(quantize)

            # straight through

            quantize = x + (quantize - x).detach()

            if self.sync_update_v > 0.0:
                # (21) in https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
                quantize = quantize + self.sync_update_v * (quantize - quantize.detach())

        # function for calculating cross entropy loss to distance matrix
        # used for (1) naturalspeech2 training residual vq latents to be close to the correct codes and (2) cross-entropy based commitment loss

        def calculate_ce_loss(codes):
            if not is_multiheaded:
                dist_einops_eq = "1 b n l -> b l n"
            elif self.separate_codebook_per_head:
                dist_einops_eq = "c b n l -> b l n c"
            else:
                dist_einops_eq = "1 (b h) n l -> b l n h"

            ce_loss = F.cross_entropy(
                rearrange(distances, dist_einops_eq, b=shape[0]), codes, ignore_index=-1
            )

            return ce_loss

        # if returning cross entropy loss on codes that were passed in

        if return_loss:
            return quantize, calculate_ce_loss(indices)

        # transform embedding indices

        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, "h b n -> b n h", h=heads)
            else:
                embed_ind = rearrange(embed_ind, "1 (b h) n -> b n h", h=heads)

        if self.accept_image_fmap:
            embed_ind = rearrange(embed_ind, "b (h w) ... -> b h w ...", h=height, w=width)

        if only_one:
            embed_ind = rearrange(embed_ind, "b 1 -> b")

        # aggregate loss

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                if self.commitment_use_cross_entropy_loss:
                    if mask is not None:
                        ce_loss_mask = mask
                        if is_multiheaded:
                            ce_loss_mask = repeat(ce_loss_mask, "b n -> b n h", h=heads)

                        embed_ind.masked_fill_(~ce_loss_mask, -1)

                    commit_loss = calculate_ce_loss(embed_ind)
                else:
                    if mask is not None:
                        # with variable lengthed sequences
                        commit_loss = F.mse_loss(commit_quantize, x, reduction="none")

                        loss_mask = mask
                        if is_multiheaded:
                            loss_mask = repeat(
                                loss_mask,
                                "b n -> c (b h) n",
                                c=commit_loss.shape[0],
                                h=commit_loss.shape[1] // mask.shape[0],
                            )

                        commit_loss = commit_loss[loss_mask].mean()
                    else:
                        commit_loss = F.mse_loss(commit_quantize, x)

                loss = loss + commit_loss * self.commitment_weight

            if self.has_codebook_orthogonal_loss:
                codebook = self._codebook.embed

                # only calculate orthogonal loss for the activated codes for this batch

                if self.orthogonal_reg_active_codes_only:
                    assert not (is_multiheaded and self.separate_codebook_per_head), (
                        "orthogonal regularization for only active codes not compatible with multi-headed with separate codebooks yet"
                    )
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[:, unique_code_ids]

                num_codes = codebook.shape[-2]

                if (self.orthogonal_reg_max_codes is not None) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=device)[: self.orthogonal_reg_max_codes]
                    codebook = codebook[:, rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        # handle multi-headed quantized embeddings

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, "h b n d -> b n (h d)", h=heads)
            else:
                quantize = rearrange(quantize, "1 (b h) n d -> b n (h d)", h=heads)

        # project out

        quantize = self.project_out(quantize)

        # rearrange quantized embeddings

        if need_transpose:
            quantize = rearrange(quantize, "b n d -> b d n")

        if self.accept_image_fmap:
            quantize = rearrange(quantize, "b (h w) c -> b c h w", h=height, w=width)

        if only_one:
            quantize = rearrange(quantize, "b 1 d -> b d")

        # if masking, only return quantized for where mask has True

        if mask is not None:
            quantize = torch.where(rearrange(mask, "... -> ... 1"), quantize, orig_input)

        return quantize, embed_ind, loss


def noop(*args, **kwargs):
    pass


def identity(t):
    return t


def cdist(x, y):
    x2 = reduce(x**2, "b n d -> b n", "sum")
    y2 = reduce(y**2, "b n d -> b n", "sum")
    xy = einsum("b i d, b j d -> b i j", x, y) * -2
    return (rearrange(x2, "b i -> b i 1") + rearrange(y2, "b j -> b 1 j") + xy).sqrt()


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def ema_inplace(old, new, decay):
    is_mps = str(old.device).startswith("mps:")

    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(
    logits,
    temperature=1.0,
    stochastic=False,
    straight_through=False,
    reinmax=False,
    dim=-1,
    training=True,
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    assert not (reinmax and not straight_through), (
        "reinmax can only be turned on if using straight through gumbel softmax"
    )

    if not straight_through or temperature <= 0.0 or not training:
        return ind, one_hot

    # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
    # algorithm 2

    if reinmax:
        π0 = logits.softmax(dim=dim)
        π1 = (one_hot + (logits / temperature).softmax(dim=dim)) / 2
        π1 = ((log(π1) - logits).detach() + logits).softmax(dim=1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - π2.detach() + one_hot
    else:
        π1 = (logits / temperature).softmax(dim=dim)
        one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)


def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)

    distributed.barrier()
    return all_x


def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, "1 ... -> ...")

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)

    return rearrange(out, "... -> 1 ...")


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def kmeans(
    samples,
    num_clusters,
    num_iters=10,
    sample_fn=batched_sample_vectors,
    all_reduce_fn=noop,
):
    num_codebooks, dim, dtype, _device = (
        samples.shape[0],
        samples.shape[-1],
        samples.dtype,
        samples.device,
    )

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        dists = -torch.cdist(samples, means, p=2)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

        new_means.scatter_add_(1, repeat(buckets, "h n -> h n d", d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, "... -> ... 1")
        all_reduce_fn(new_means)

        means = torch.where(rearrange(zero_mask, "... -> ... 1"), means, new_means)

    return means, bins


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, "h b n -> h b n d", d=dim)
    embeds = repeat(embeds, "h c d -> h b c d", b=batch)
    return embeds.gather(2, indices)


def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = F.normalize(t, p=2, dim=-1)
    cosine_sim = einsum("h i d, h j d -> h i j", normed_codes, normed_codes)
    return (cosine_sim**2).sum() / (h * n**2) - (1 / n)


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks=1,
        kmeans_init=False,
        kmeans_iters=10,
        sync_kmeans=True,
        decay=0.8,
        eps=1e-5,
        threshold_ema_dead_code=2,
        reset_cluster_size=None,
        use_ddp=False,
        learnable_codebook=False,
        gumbel_sample=gumbel_sample,
        sample_codebook_temp=1.0,
        ema_update=True,
        affine_param=False,
        sync_affine_param=False,
        affine_param_batch_decay=0.99,
        affine_param_codebook_decay=0.9,
    ):
        super().__init__()
        self.transform_input = identity

        self.decay = decay
        self.ema_update = ema_update

        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = (
            reset_cluster_size if (reset_cluster_size is not None) else threshold_ema_dead_code
        )

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        assert not (use_ddp and num_codebooks > 1 and kmeans_init), (
            "kmeans init is not compatible with multiple codebooks in distributed environment for now"
        )

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer("initted", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(num_codebooks, codebook_size))
        self.register_buffer("embed_avg", embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer("embed", embed)

        # affine related params

        self.affine_param = affine_param
        self.sync_affine_param = sync_affine_param

        if not affine_param:
            return

        self.affine_param_batch_decay = affine_param_batch_decay
        self.affine_param_codebook_decay = affine_param_codebook_decay

        self.register_buffer("batch_mean", None)
        self.register_buffer("batch_variance", None)

        self.register_buffer("codebook_mean_needs_init", torch.Tensor([True]))
        self.register_buffer("codebook_mean", torch.empty(num_codebooks, 1, dim))
        self.register_buffer("codebook_variance_needs_init", torch.Tensor([True]))
        self.register_buffer("codebook_variance", torch.empty(num_codebooks, 1, dim))

    @torch.jit.ignore
    def init_embed_(self, data, mask=None):
        if self.initted:
            return

        if mask is not None:
            c = data.shape[0]
            data = rearrange(data[mask], "(c n) d -> c n d", c=c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn=self.sample_fn,
            all_reduce_fn=self.kmeans_all_reduce_fn,
        )

        embed_sum = embed * rearrange(cluster_size, "... -> ... 1")

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    @torch.jit.ignore
    def update_with_decay(self, buffer_name, new_value, decay):
        old_value = getattr(self, buffer_name)

        needs_init = getattr(self, buffer_name + "_needs_init", False)

        if needs_init:
            self.register_buffer(buffer_name + "_needs_init", torch.Tensor([False]))

        if not (old_value is not None) or needs_init:
            self.register_buffer(buffer_name, new_value.detach())

            return

        value = old_value * decay + new_value.detach() * (1 - decay)
        self.register_buffer(buffer_name, value)

    @torch.jit.ignore
    def update_affine(self, data, embed, mask=None):
        assert self.affine_param

        var_fn = partial(torch.var, unbiased=False)

        # calculate codebook mean and variance

        embed = rearrange(embed, "h ... d -> h (...) d")

        if self.training:
            self.update_with_decay(
                "codebook_mean",
                reduce(embed, "h n d -> h 1 d", "mean"),
                self.affine_param_codebook_decay,
            )
            self.update_with_decay(
                "codebook_variance",
                reduce(embed, "h n d -> h 1 d", var_fn),
                self.affine_param_codebook_decay,
            )

        # prepare batch data, which depends on whether it has masking

        data = rearrange(data, "h ... d -> h (...) d")

        if mask is not None:
            c = data.shape[0]
            data = rearrange(data[mask], "(c n) d -> c n d", c=c)

        # calculate batch mean and variance

        if not self.sync_affine_param:
            self.update_with_decay(
                "batch_mean",
                reduce(data, "h n d -> h 1 d", "mean"),
                self.affine_param_batch_decay,
            )
            self.update_with_decay(
                "batch_variance",
                reduce(data, "h n d -> h 1 d", var_fn),
                self.affine_param_batch_decay,
            )
            return

        num_vectors, device, dtype = data.shape[-2], data.device, data.dtype

        # number of vectors, for denominator

        num_vectors = torch.tensor([num_vectors], device=device, dtype=dtype)
        distributed.all_reduce(num_vectors)

        # calculate distributed mean

        batch_sum = reduce(data, "h n d -> h 1 d", "sum")
        distributed.all_reduce(batch_sum)
        batch_mean = batch_sum / num_vectors

        self.update_with_decay("batch_mean", batch_mean, self.affine_param_batch_decay)

        # calculate distributed variance

        variance_numer = reduce((data - batch_mean) ** 2, "h n d -> h 1 d", "sum")
        distributed.all_reduce(variance_numer)
        batch_variance = variance_numer / num_vectors

        self.update_with_decay("batch_variance", batch_variance, self.affine_param_batch_decay)

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(
            zip(batch_samples.unbind(dim=0), batch_mask.unbind(dim=0), strict=False)
        ):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, "... -> 1 ..."), mask.sum().item())
            sampled = rearrange(sampled, "1 ... -> ...")

            self.embed.data[ind][mask] = sampled

            self.cluster_size.data[ind][mask] = self.reset_cluster_size
            self.embed_avg.data[ind][mask] = sampled * self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "h ... d -> h (...) d")
        self.replace(batch_samples, batch_mask=expired_codes)

    @autocast(enabled=False)
    def forward(self, x, sample_codebook_temp=None, mask=None, freeze_codebook=False):
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = (
            sample_codebook_temp if (sample_codebook_temp is not None) else self.sample_codebook_temp
        )

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, "... -> 1 ...")

        flatten, ps = pack_one(x, "h * d")

        if mask is not None:
            mask = repeat(
                mask,
                "b n -> c (b h n)",
                c=flatten.shape[0],
                h=flatten.shape[-2] // (mask.shape[0] * mask.shape[1]),
            )

        self.init_embed_(flatten, mask=mask)

        if self.affine_param:
            self.update_affine(flatten, self.embed, mask=mask)

        embed = self.embed if self.learnable_codebook else self.embed.detach()

        if self.affine_param:
            codebook_std = self.codebook_variance.clamp(min=1e-5).sqrt()
            batch_std = self.batch_variance.clamp(min=1e-5).sqrt()
            embed = (embed - self.codebook_mean) * (batch_std / codebook_std) + self.batch_mean

        dist = -cdist(flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(
            dist, dim=-1, temperature=sample_codebook_temp, training=self.training
        )

        embed_ind = unpack_one(embed_ind, ps, "h *")

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, "h * c")
            quantize = einsum("h b n c, h c d -> h b n d", unpacked_onehot, embed)
        else:
            quantize = batched_embedding(embed_ind, embed)

        if self.training and self.ema_update and not freeze_codebook:
            if self.affine_param:
                flatten = (flatten - self.batch_mean) * (codebook_std / batch_std) + self.codebook_mean

            if mask is not None:
                embed_onehot[~mask] = 0.0

            cluster_size = embed_onehot.sum(dim=1)

            self.all_reduce_fn(cluster_size)
            ema_inplace(self.cluster_size.data, cluster_size, self.decay)

            embed_sum = einsum("h n d, h n c -> h c d", flatten, embed_onehot)
            self.all_reduce_fn(embed_sum.contiguous())
            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            cluster_size = laplace_smoothing(
                self.cluster_size, self.codebook_size, self.eps
            ) * self.cluster_size.sum(dim=-1, keepdim=True)

            embed_normalized = self.embed_avg / rearrange(cluster_size, "... -> ... 1")
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = tuple(rearrange(t, "1 ... -> ...") for t in (quantize, embed_ind))

        dist = unpack_one(dist, ps, "h * d")

        return quantize, embed_ind, dist
