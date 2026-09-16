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

"""AdaRMS conditioning tests for the shared PI Gemma blocks.

pi05 feeds the flow-matching timestep to the action expert *only* through AdaRMS, so
training-time RTC (arXiv 2512.05964) needs scale/shift/gate to vary per token. These
tests pin both conditioning shapes: the scalar-per-sample form used by ordinary
training, and the per-token form that marks the clean action prefix.
"""

import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.gemma.configuration_gemma import GemmaConfig  # noqa: E402

from lerobot.policies.pi_gemma import (  # noqa: E402
    PiGemmaRMSNorm,
    _get_pi_gemma_decoder_layer_base,
)

BATCH, TOKENS, HIDDEN = 2, 4, 16


def _decoder_layer() -> torch.nn.Module:
    config = GemmaConfig(
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
    )
    config.use_adarms = True
    config.adarms_cond_dim = HIDDEN
    config._attn_implementation = "eager"  # noqa: SLF001
    return _get_pi_gemma_decoder_layer_base()(config, 0)


def test_adarms_per_token_cond_keeps_sequence_shape():
    norm = PiGemmaRMSNorm(HIDDEN, cond_dim=HIDDEN)
    x = torch.randn(BATCH, TOKENS, HIDDEN)

    out, gate = norm(x, cond=torch.randn(BATCH, TOKENS, HIDDEN))

    assert out.shape == (BATCH, TOKENS, HIDDEN)
    assert gate.shape == (BATCH, TOKENS, HIDDEN)


def test_adarms_constant_per_token_cond_matches_scalar_cond():
    """A per-token cond repeated across tokens must reproduce the scalar-cond result."""
    torch.manual_seed(0)
    norm = PiGemmaRMSNorm(HIDDEN, cond_dim=HIDDEN)
    torch.nn.init.normal_(norm.dense.weight)  # zero-init would hide any modulation bug
    x = torch.randn(BATCH, TOKENS, HIDDEN)
    cond = torch.randn(BATCH, HIDDEN)

    scalar_out, scalar_gate = norm(x, cond=cond)
    token_out, token_gate = norm(x, cond=cond[:, None, :].expand(BATCH, TOKENS, HIDDEN))

    torch.testing.assert_close(token_out, scalar_out)
    torch.testing.assert_close(token_gate, scalar_gate.expand_as(token_gate))


def test_adarms_per_token_cond_actually_varies_per_token():
    torch.manual_seed(0)
    norm = PiGemmaRMSNorm(HIDDEN, cond_dim=HIDDEN)
    torch.nn.init.normal_(norm.dense.weight)
    x = torch.zeros(BATCH, TOKENS, HIDDEN)  # isolate the shift term

    cond = torch.zeros(BATCH, TOKENS, HIDDEN)
    cond[:, 0] = 1.0  # only the first token (the RTC prefix marker) is conditioned
    out, _ = norm(x, cond=cond)

    assert not torch.allclose(out[:, 0], out[:, 1])
    torch.testing.assert_close(out[:, 1], out[:, 2])


def test_adarms_rejects_token_count_mismatch():
    norm = PiGemmaRMSNorm(HIDDEN, cond_dim=HIDDEN)

    with pytest.raises(ValueError, match="must match"):
        norm(torch.randn(BATCH, TOKENS, HIDDEN), cond=torch.randn(BATCH, TOKENS + 1, HIDDEN))


@pytest.mark.parametrize("cond_shape", [(BATCH, HIDDEN), (BATCH, TOKENS, HIDDEN)])
def test_pi_gemma_decoder_layer_accepts_both_cond_shapes(cond_shape):
    """Regression: a per-token cond used to broadcast into (B, B, T, D) and blow up."""
    layer = _decoder_layer()
    hidden = torch.randn(BATCH, TOKENS, HIDDEN)

    out = layer(
        hidden,
        attention_mask=torch.zeros(BATCH, 1, TOKENS, TOKENS),
        position_ids=torch.arange(TOKENS)[None].expand(BATCH, TOKENS),
        position_embeddings=(torch.ones(BATCH, TOKENS, 8), torch.zeros(BATCH, TOKENS, 8)),
        adarms_cond=torch.randn(*cond_shape),
    )
    out = out[0] if isinstance(out, tuple) else out

    assert out.shape == (BATCH, TOKENS, HIDDEN)
