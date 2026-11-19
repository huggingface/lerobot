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

"""Characterize how PI0-FAST turns generated PaliGemma tokens back into FAST actions."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("scipy")

from lerobot.policies.pi0_fast import modeling_pi0_fast  # noqa: E402
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy  # noqa: E402
from lerobot.processor import ActionTokenizerProcessorStep  # noqa: E402

# PaliGemma's vocabulary, including location and segmentation tokens.
PALIGEMMA_VOCAB_SIZE = 257152


@pytest.mark.parametrize("fast_skip_tokens", [128, 1152], ids=["pi0_fast", "fineart_vla"])
def test_paligemma_mapping_round_trips_every_fast_id(fast_skip_tokens: int) -> None:
    paligemma = SimpleNamespace(vocab_size=PALIGEMMA_VOCAB_SIZE)
    processor = SimpleNamespace(fast_skip_tokens=fast_skip_tokens, _paligemma_tokenizer=paligemma)
    policy = SimpleNamespace(
        config=SimpleNamespace(fast_skip_tokens=fast_skip_tokens), _paligemma_tokenizer=paligemma
    )
    fast_ids = torch.arange(2048)

    model_ids = ActionTokenizerProcessorStep._act_tokens_to_paligemma_tokens(processor, fast_ids)

    # FAST IDs fill the vocabulary downward from just below the skipped tail tokens.
    assert model_ids[0] == PALIGEMMA_VOCAB_SIZE - 1 - fast_skip_tokens
    assert model_ids[-1] == PALIGEMMA_VOCAB_SIZE - fast_skip_tokens - 2048
    torch.testing.assert_close(PI0FastPolicy._paligemma_tokens_to_act_tokens(policy, model_ids), fast_ids)


def test_detokenize_actions_decodes_full_chunks_in_relaxed_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    horizon, action_dim, fast_id = 10, 3, 7
    decode = Mock(return_value=np.ones((1, horizon, action_dim)))
    monkeypatch.setattr(modeling_pi0_fast, "decode_fast_actions", decode)
    code_id = PALIGEMMA_VOCAB_SIZE - 1 - 128 - fast_id
    vocabulary = {1: "Action", 2: ":", 3: "|", code_id: "code"}
    policy = SimpleNamespace(
        config=SimpleNamespace(validate_action_token_prefix=True, fast_skip_tokens=128),
        action_tokenizer=object(),
        _paligemma_tokenizer=SimpleNamespace(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            encode=lambda text, **kwargs: [1, 2],
            convert_ids_to_tokens=lambda ids: [vocabulary[token] for token in ids],
            convert_tokens_to_ids=lambda tokens: [code_id for _ in tokens],
        ),
    )
    policy._paligemma_tokens_to_act_tokens = partial(PI0FastPolicy._paligemma_tokens_to_act_tokens, policy)

    # Wrapper tokens and everything after the "|" end marker are stripped before decoding.
    generated = torch.tensor([[1, 2, code_id, 3, code_id]])
    actions = PI0FastPolicy.detokenize_actions(policy, generated, horizon, action_dim)

    decode.assert_called_once_with(
        policy.action_tokenizer, [[fast_id]], action_dim, time_horizon=horizon, relaxed=True
    )
    torch.testing.assert_close(actions, torch.ones(1, horizon, action_dim))
