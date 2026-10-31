#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")

from lerobot.policies.pi0_fast import modeling_pi0_fast  # noqa: E402
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig  # noqa: E402
from lerobot.policies.pi_gemma import PaliGemmaForConditionalGenerationWithPiGemma  # noqa: E402

END_OF_ACTION = 3


class TinyPaliGemma(modeling_pi0_fast.PI0FastPaliGemma):
    """The PI0Fast wrapper around a tiny PaliGemma, so decoding runs without the full-size model."""

    def __init__(self, vlm_config, use_adarms=None, precision=torch.float32):
        nn.Module.__init__(self)
        config = transformers.PaliGemmaConfig(
            text_config=transformers.GemmaConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=64,
                hidden_activation="gelu_pytorch_tanh",
            ).to_dict(),
            vision_config=transformers.SiglipVisionConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=8,
                patch_size=4,
            ).to_dict(),
            image_token_index=63,
            projection_dim=16,
            vocab_size=64,
        )
        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config)
        # Every step predicts the end-of-action marker, so greedy decoding stops after one token.
        self.paligemma.lm_head = nn.Linear(16, 64)
        with torch.no_grad():
            self.paligemma.lm_head.weight.zero_()
            self.paligemma.lm_head.bias.zero_()
            self.paligemma.lm_head.bias[END_OF_ACTION] = 100


class Tokenizer:
    bos_token_id = 1

    def convert_tokens_to_ids(self, token):
        return END_OF_ACTION


def test_greedy_decoding_exports_and_keeps_its_early_stop_under_torch_compile(monkeypatch):
    monkeypatch.setattr(modeling_pi0_fast, "PI0FastPaliGemma", TinyPaliGemma)
    core = modeling_pi0_fast.PI0FastPytorch(
        PI0FastConfig(dtype=torch.float32), paligemma_tokenizer=Tokenizer()
    )

    class Decode(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core

        def forward(self, image, image_mask, tokens, masks):
            return self.core.sample_actions_fast_kv_cache(
                [image], [image_mask], tokens, masks, max_decoding_steps=4, temperature=0
            )

    model = Decode().eval()
    tokens = torch.tensor([[4, 5], [6, 7]])
    image, image_mask = torch.randn(2, 3, 8, 8), torch.ones(2, dtype=torch.bool)
    inputs = (image, image_mask, tokens, torch.ones_like(tokens, dtype=torch.bool))
    with torch.no_grad():
        eager = model(*inputs)
        assert eager.tolist() == [[END_OF_ACTION, 0, 0, 0]] * 2
        # torch.compile on PyTorch 2.11 also reports is_exporting(); the early stop must survive it.
        assert torch.equal(torch.compile(model, backend="eager")(*inputs), eager)
        # The exported loop runs every step, so it keeps emitting after the marker; decoding drops those.
        exported = torch.export.export(model, inputs, strict=False).module()(*inputs)
    assert exported.tolist() == [[END_OF_ACTION] * 4] * 2
