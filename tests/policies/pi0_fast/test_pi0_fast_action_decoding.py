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

"""Characterize PI0-FAST's FAST decoding against the pinned default backend."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("scipy")

from scipy.fft import dct  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy  # noqa: E402
from lerobot.processor import ActionTokenizerProcessorStep  # noqa: E402

# PI0-FAST's default FAST tokenizer, pinned to the revision these tests characterize.
FAST_REPO_ID = "lerobot/fast-action-tokenizer"
FAST_REVISION = "3d4d8dd847c5333f1cb0f54d00074ba22b14a5b7"
HORIZON, ACTION_DIM = 10, 3
# PaliGemma's vocabulary, including location and segmentation tokens.
PALIGEMMA_VOCAB_SIZE = 257152


@pytest.fixture(scope="module")
def fast_backend() -> Any:
    try:
        return AutoProcessor.from_pretrained(FAST_REPO_ID, revision=FAST_REVISION, trust_remote_code=True)
    except OSError as error:
        pytest.skip(f"Cannot load {FAST_REPO_ID} from the Hub: {error}")


@pytest.fixture(scope="module")
def tokens(fast_backend: Any) -> list[list[int]]:
    time = np.linspace(0, 1, HORIZON)[:, None]
    actions = np.stack(
        [
            np.concatenate([np.sin(2 * np.pi * time), np.cos(2 * np.pi * time), 2 * time - 1], axis=1),
            np.concatenate([np.full_like(time, 0.5), -0.25 * time, time**2], axis=1),
        ]
    )
    return fast_backend(actions)


def _policy_decode(fast_backend: Any, tokens: list[list[int]], relaxed_decoding: bool = True) -> np.ndarray:
    """Call PI0-FAST's decoder without loading policy weights; it only reads the FAST backend."""
    policy = SimpleNamespace(action_tokenizer=fast_backend)
    return PI0FastPolicy.decode_actions_with_fast(
        policy, tokens, HORIZON, ACTION_DIM, relaxed_decoding=relaxed_decoding
    )


def _dct_coefficients(fast_backend: Any, actions: np.ndarray) -> np.ndarray:
    """Recover the flattened integer FAST coefficients of decoded chunks."""
    coefficients = dct(actions, axis=-2, norm="ortho") * fast_backend.scale
    return np.rint(coefficients).reshape(len(actions), -1)


def test_policy_decoder_matches_backend_for_complete_sequences(
    fast_backend: Any, tokens: list[list[int]]
) -> None:
    expected = fast_backend.decode(tokens, time_horizon=HORIZON, action_dim=ACTION_DIM)

    for relaxed_decoding in (True, False):
        decoded = _policy_decode(fast_backend, tokens, relaxed_decoding)
        assert decoded.shape == (len(tokens), HORIZON, ACTION_DIM)
        np.testing.assert_allclose(decoded, expected, rtol=0, atol=1e-12)


def test_relaxed_decoding_zero_fills_missing_trailing_coefficients(
    fast_backend: Any, tokens: list[list[int]]
) -> None:
    incomplete = [sequence[:-1] for sequence in tokens]

    full = _dct_coefficients(fast_backend, _policy_decode(fast_backend, tokens))
    partial = _dct_coefficients(fast_backend, _policy_decode(fast_backend, incomplete))

    for index, sequence in enumerate(incomplete):
        kept = len(fast_backend.bpe_tokenizer.decode(sequence))
        assert 0 < kept < HORIZON * ACTION_DIM
        np.testing.assert_array_equal(partial[index, :kept], full[index, :kept])
        np.testing.assert_array_equal(partial[index, kept:], 0)


def test_relaxed_decoding_drops_extra_trailing_coefficients(
    fast_backend: Any, tokens: list[list[int]]
) -> None:
    oversized = [sequence + sequence for sequence in tokens]

    np.testing.assert_array_equal(
        _policy_decode(fast_backend, oversized), _policy_decode(fast_backend, tokens)
    )


def test_strict_and_backend_decoding_return_zeros_for_incomplete_sequences(
    fast_backend: Any, tokens: list[list[int]]
) -> None:
    incomplete = [sequence[:-1] for sequence in tokens]
    zeros = np.zeros((len(tokens), HORIZON, ACTION_DIM))

    np.testing.assert_array_equal(_policy_decode(fast_backend, incomplete, relaxed_decoding=False), zeros)
    np.testing.assert_array_equal(
        fast_backend.decode(incomplete, time_horizon=HORIZON, action_dim=ACTION_DIM), zeros
    )


def test_out_of_vocabulary_ids_decode_to_zero_actions(fast_backend: Any) -> None:
    unknown = [[len(fast_backend.bpe_tokenizer) + 10]]

    np.testing.assert_array_equal(_policy_decode(fast_backend, unknown), np.zeros((1, HORIZON, ACTION_DIM)))


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
