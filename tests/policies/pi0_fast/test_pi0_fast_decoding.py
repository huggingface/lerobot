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

"""Early stop of PI0-FAST's autoregressive decoding, run through both real loops on a tiny backbone."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")

from lerobot.policies.pi0_fast import modeling_pi0_fast  # noqa: E402
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig  # noqa: E402
from lerobot.policies.pi0_fast.modeling_pi0_fast import GemmaConfig, PI0FastPytorch  # noqa: E402

WIDTH, IMAGE_TOKENS, PROMPT, STEPS = 64, 4, 5, 12
BOS, END, OTHER, VOCAB = 2, 7, 11, 16
DECODERS = ("sample_actions_fast", "sample_actions_fast_kv_cache")


class _Tokenizer:
    bos_token_id = BOS

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "|"
        return END


class _ScriptedHead(nn.Module):
    """LM head whose row ``b`` emits ``END`` at decode step ``ends[b]`` and ``OTHER`` at every other step."""

    def __init__(self) -> None:
        super().__init__()
        self.ends: tuple[int | None, ...] = ()
        self.steps = 0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # A single finite logit also forces the scripted token when using multinomial.
        logits = torch.full((hidden.shape[0], 1, VOCAB), -torch.inf)
        logits[..., OTHER] = 0.0
        for row, end in enumerate(self.ends):
            if end == self.steps:
                logits[row, :, OTHER] = -torch.inf
                logits[row, :, END] = 0.0
        self.steps += 1
        return logits


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> PI0FastPytorch:
    tiny = GemmaConfig(width=WIDTH, depth=2, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
    paligemma_class = modeling_pi0_fast.PaliGemmaForConditionalGenerationWithPiGemma

    def make_tiny_paligemma(config: transformers.PaliGemmaConfig) -> nn.Module:
        # Shrink these before construction; replacing the vision tower afterwards still allocates it.
        config.text_config.vocab_size = VOCAB
        config.vision_config.hidden_size = 16
        config.vision_config.intermediate_size = 32
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 2
        return paligemma_class(config=config)

    monkeypatch.setattr(modeling_pi0_fast, "get_gemma_config", lambda variant: tiny)
    monkeypatch.setattr(
        modeling_pi0_fast, "PaliGemmaForConditionalGenerationWithPiGemma", make_tiny_paligemma
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = PI0FastPytorch(PI0FastConfig(device="cpu"), paligemma_tokenizer=_Tokenizer())
    backbone = model.paligemma_with_expert
    backbone.embed_image = lambda image: torch.zeros(image.shape[0], IMAGE_TOKENS, WIDTH)
    backbone.paligemma.lm_head = _ScriptedHead()
    return model.eval()


def _inputs(batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(batch, 3, 8, 8),
        torch.ones(batch, dtype=torch.bool),
        torch.full((batch, PROMPT), OTHER),
        torch.ones(batch, PROMPT, dtype=torch.bool),
    )


def _decode(
    model: PI0FastPytorch,
    decode: Callable[..., torch.Tensor],
    ends: tuple[int | None, ...],
    temperature: float = 0.0,
) -> tuple[torch.Tensor, int]:
    head = model.paligemma_with_expert.paligemma.lm_head
    head.ends, head.steps = ends, 0
    image, image_mask, prompt, mask = _inputs(len(ends))
    tokens = decode(
        images=[image],
        img_masks=[image_mask],
        tokens=prompt,
        masks=mask,
        max_decoding_steps=STEPS,
        temperature=temperature,
    )
    return tokens, head.steps


@pytest.mark.parametrize("decoder", DECODERS)
def test_greedy_decoding_stops_once_every_row_ended(model: PI0FastPytorch, decoder: str) -> None:
    tokens, steps = _decode(model, getattr(model, decoder), ends=(3, 5))

    assert steps == 6
    assert tokens.tolist() == [
        [OTHER] * 3 + [END] + [OTHER] * 2 + [0] * 6,
        [OTHER] * 5 + [END] + [0] * 6,
    ]


@pytest.mark.parametrize("decoder", DECODERS)
def test_greedy_decoding_runs_to_the_limit_while_a_row_is_unfinished(
    model: PI0FastPytorch, decoder: str
) -> None:
    tokens, steps = _decode(model, getattr(model, decoder), ends=(3, None))

    assert steps == STEPS
    assert tokens.tolist() == [[OTHER] * 3 + [END] + [OTHER] * 8, [OTHER] * STEPS]


@pytest.mark.parametrize("decoder", DECODERS)
def test_sampling_never_stops_early(model: PI0FastPytorch, decoder: str) -> None:
    # Skipping multinomial calls would shift the RNG state for later calls.
    with torch.random.fork_rng(devices=[]):
        tokens, steps = _decode(model, getattr(model, decoder), ends=(0, 0), temperature=1.0)

    assert steps == STEPS
    assert tokens.tolist() == [[END] + [OTHER] * (STEPS - 1)] * 2


def test_compiled_decoding_never_stops_early(model: PI0FastPytorch) -> None:
    # compile_model=True wraps this method. fullgraph=True also protects user-applied compilation
    # from a new data-dependent break; the eager backend needs no accelerator or code generation.
    torch.compiler.reset()
    try:
        compiled = torch.compile(model.sample_actions_fast, backend="eager", fullgraph=True)
        tokens, steps = _decode(model, compiled, ends=(0, 0))
    finally:
        torch.compiler.reset()

    assert steps == STEPS
    assert tokens.tolist() == [[END] + [OTHER] * (STEPS - 1)] * 2


class _Decode(nn.Module):
    def __init__(self, core: PI0FastPytorch) -> None:
        super().__init__()
        self.core = core

    def forward(
        self, image: torch.Tensor, image_mask: torch.Tensor, tokens: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        return self.core.sample_actions_fast([image], [image_mask], tokens, masks, max_decoding_steps=4)


@pytest.mark.parametrize("strict", [False, True])
def test_exported_decoding_never_stops_early(model: PI0FastPytorch, strict: bool) -> None:
    head = model.paligemma_with_expert.paligemma.lm_head
    head.ends, head.steps = (0, 0), 0
    inputs = _inputs(2)
    exported = torch.export.export(_Decode(model), inputs, strict=strict)

    # Export does not preserve the head's Python counter; check the actual graph output.
    assert exported.module()(*inputs).tolist() == [[END, OTHER, OTHER, OTHER]] * 2


def test_jit_traced_decoding_never_stops_early(model: PI0FastPytorch) -> None:
    head = model.paligemma_with_expert.paligemma.lm_head
    head.ends, head.steps = (0, 0), 0
    inputs = _inputs(2)
    # The scripted head is stateful, and eager deliberately has a zero-filled suffix. Compare
    # the traced output to the fixed-loop expectation instead of JIT's eager-output checker.
    traced = torch.jit.trace(_Decode(model), inputs, check_trace=False)

    assert traced(*inputs).tolist() == [[END, OTHER, OTHER, OTHER]] * 2
