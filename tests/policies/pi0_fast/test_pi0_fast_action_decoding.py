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

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("scipy")

from lerobot.policies.pi0_fast import modeling_pi0_fast  # noqa: E402
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy  # noqa: E402
from lerobot.processor import fast_ids_to_paligemma_ids, paligemma_ids_to_fast_ids  # noqa: E402
from lerobot.processor.tokenizer_processor import ACTION_TOKENIZER_DIR  # noqa: E402

# PaliGemma's vocabulary, including location and segmentation tokens.
PALIGEMMA_VOCAB_SIZE = 257152


@pytest.mark.parametrize("fast_skip_tokens", [128, 1152], ids=["pi0_fast", "fineart_vla"])
def test_paligemma_mapping_round_trips_every_fast_id(fast_skip_tokens: int) -> None:
    fast_ids = torch.arange(2048)

    model_ids = fast_ids_to_paligemma_ids(fast_ids, PALIGEMMA_VOCAB_SIZE, fast_skip_tokens)

    # FAST IDs fill the vocabulary downward from just below the skipped tail tokens.
    assert model_ids[0] == PALIGEMMA_VOCAB_SIZE - 1 - fast_skip_tokens
    assert model_ids[-1] == PALIGEMMA_VOCAB_SIZE - fast_skip_tokens - 2048
    torch.testing.assert_close(
        paligemma_ids_to_fast_ids(model_ids, PALIGEMMA_VOCAB_SIZE, fast_skip_tokens), fast_ids
    )
    # Plain ints map the same way, as FineART uses them for its text-generation mask bounds.
    assert fast_ids_to_paligemma_ids(2047, PALIGEMMA_VOCAB_SIZE, fast_skip_tokens) == int(model_ids[-1])


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

    # Wrapper tokens and everything after the "|" end marker are stripped before decoding.
    generated = torch.tensor([[1, 2, code_id, 3, code_id]])
    actions = PI0FastPolicy.detokenize_actions(policy, generated, horizon, action_dim)

    decode.assert_called_once_with(
        policy.action_tokenizer, [[fast_id]], action_dim, time_horizon=horizon, relaxed=True
    )
    torch.testing.assert_close(actions, torch.ones(1, horizon, action_dim))


def _write_checkpoint(root: Path, action_tokenizer_name: str, *, with_tokenizer: bool) -> Path:
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"action_tokenizer_name": action_tokenizer_name}))
    if with_tokenizer:
        (root / ACTION_TOKENIZER_DIR).mkdir()
        (root / ACTION_TOKENIZER_DIR / "processor_config.json").write_text("{}")
    return root


@pytest.mark.parametrize(
    ("with_tokenizer", "configured_name", "uses_saved_copy"),
    [
        (True, "/fitted/on/another/machine", True),
        # Checkpoints saved before the processor stored its tokenizer.
        (False, "/fitted/on/another/machine", False),
        # Fine-tuning with another tokenizer keeps the configured one.
        (True, "lerobot/fast-action-tokenizer", False),
    ],
)
def test_decoding_tokenizer_comes_from_the_checkpoint_when_saved_for_it(
    tmp_path: Path, with_tokenizer: bool, configured_name: str, uses_saved_copy: bool
) -> None:
    checkpoint = _write_checkpoint(
        tmp_path / "checkpoint", "/fitted/on/another/machine", with_tokenizer=with_tokenizer
    )

    folder = modeling_pi0_fast._saved_action_tokenizer(checkpoint, configured_name)

    assert folder == (str(checkpoint / ACTION_TOKENIZER_DIR) if uses_saved_copy else None)


def test_decoding_tokenizer_falls_back_when_the_hub_checkpoint_is_unavailable() -> None:
    folder = modeling_pi0_fast._saved_action_tokenizer(
        "lerobot-tests/not-a-checkpoint", "lerobot/fast-action-tokenizer", local_files_only=True
    )

    assert folder is None
