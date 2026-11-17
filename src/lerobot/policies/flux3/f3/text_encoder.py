# Copyright 2026 Black Forest Labs. All rights reserved.
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
# Vendored from black-forest-labs/flux-action (src/flux_action/models/text_encoder.py).
"""Qwen3-VL text encoder: stacks hidden states from selected layers along the
channel dim to form the DiT text context. Uses Qwen3-VL-4B: `ctx` = selected hidden layers stacked along the channel dim
(width = len(output_layer) * qwen_hidden = 8 * 2560 = 20480). Needs `transformers`.

Prompts are encoded one at a time, right-padded to the next multiple of
``TEXT_PAD_MULTIPLE`` (capped at ``TEXT_PAD_MAX_LENGTH``). Prompt and negative
bucket to independent lengths, so CFG runs as two bs=1 forward passes (see
``lerobot.policies.flux3.f3.sampling``).
"""

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available, require_package

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

TEXT_PAD_MULTIPLE = 80
TEXT_PAD_MAX_LENGTH = 8192


def padded_text_token_count(
    real_len: int,
    pad_multiple: int,
    pad_max_length: int,
) -> int:
    return min(math.ceil(real_len / pad_multiple) * pad_multiple, pad_max_length)


@dataclass
class Qwen3VLEmbedderParams:
    output_layer: list[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24, 28, 32])
    torch_dtype: str = "bfloat16"
    stack_channels: bool = True
    use_compile: bool = False
    compile_config: dict | None = field(default_factory=lambda: {"dynamic_compile": True})


class Qwen3VLEmbedder(nn.Module):
    def __init__(self, model_spec: str, params: Qwen3VLEmbedderParams = Qwen3VLEmbedderParams()):
        super().__init__()
        require_package("transformers", "flux3")

        self.params = params
        self.dtype = getattr(torch, params.torch_dtype)
        self.output_layer = list(params.output_layer)
        self.stack_channels = params.stack_channels

        # Shared encoders may live in a pinned subfolder of the base model repo.
        hub_kwargs = {}
        if not os.path.exists(model_spec):
            body, _, revision = model_spec.partition("@")
            model_spec, _, subfolder = body.partition(":")
            hub_kwargs = {"revision": revision or None, "subfolder": subfolder}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_spec, torch_dtype=self.dtype, **hub_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_spec, **hub_kwargs)
        if self.processor.tokenizer.padding_side != "right":
            raise ValueError("the text encoder requires a right-padding tokenizer")

        if params.use_compile:
            cc = params.compile_config or {}
            for layer in self.model.model.language_model.layers:
                layer.forward = torch.compile(
                    layer.forward,
                    mode=cc.get("compile_mode", "default"),
                    fullgraph=cc.get("fullgraph", False),
                    dynamic=cc.get("dynamic_compile", False),
                    backend=cc.get("backend", "inductor"),
                )

    @torch.no_grad()
    def forward_bucketed(self, text: str, *, fixed_length: int | None = None) -> Tensor:
        """Single-string encode -> ``(1, L, len(output_layer) * qwen_hidden)``,
        right-padded to the bucketed length. The DiT's ``vector`` and
        ``timesteps_ctx`` are zeros, F3 does not condition on a pooled text
        vector."""
        formatted = self.processor.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        real_len = self.processor.tokenizer(
            formatted,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=fixed_length or TEXT_PAD_MAX_LENGTH,
        )["input_ids"].shape[1]
        target_length = fixed_length or padded_text_token_count(
            real_len,
            TEXT_PAD_MULTIPLE,
            TEXT_PAD_MAX_LENGTH,
        )
        toks = self.processor.tokenizer(
            formatted,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=target_length,
            padding_side="right",
        )
        input_ids = toks["input_ids"].to(self.model.device, non_blocking=True)
        attention_mask = toks["attention_mask"].to(self.model.device, non_blocking=True)

        out = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        if len(self.output_layer) == 1:
            result = out.hidden_states[self.output_layer[0]]
        else:
            stacked = torch.stack([out.hidden_states[k] for k in self.output_layer], dim=1)
            result = rearrange(stacked, "b c l d -> b l (c d)") if self.stack_channels else stacked
        return result.to(self.dtype)


VEC_DIM = 768  # pooled vector width; input is always zero


def load_text_encoder(
    spec: str,
    device: str | torch.device = "cpu",
    *,
    compile_model: bool = False,
) -> nn.Module:
    """Load and freeze Qwen3-VL-4B from a Hub id or local path."""
    if not spec or not spec.strip():
        raise ValueError("text_encoder_id must be a non-empty Hub ID or local path")
    enc = Qwen3VLEmbedder(spec, Qwen3VLEmbedderParams(use_compile=compile_model))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc.to(device)


@torch.no_grad()
def text_context(
    text_encoder: nn.Module, caption: str, device: str | torch.device, *, fixed_length: int | None = None
) -> Tensor:
    """Caption -> ctx ``(1, L, 20480)`` bf16, ``L`` a multiple of 80. The empty caption is the CFG null."""
    ctx = (
        text_encoder.forward_bucketed(caption, fixed_length=fixed_length)
        if fixed_length is not None
        else text_encoder.forward_bucketed(caption)
    )
    # produced under inference_mode: clone so autograd may consume it downstream
    return ctx.to(device=device, dtype=torch.bfloat16).clone()
