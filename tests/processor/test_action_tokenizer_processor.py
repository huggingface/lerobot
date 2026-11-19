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

"""Shape, formatting and backend contracts for FAST action tokenization."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import ActionTokenizerProcessorStep, decode_fast_actions, encode_fast_actions
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import ACTION_CODE_TOKEN_MASK, ACTION_TOKEN_MASK, ACTION_TOKENS

# Default FAST tokenizers of PI0-FAST and FineART-VLA, pinned to the revisions these tests characterize.
LEROBOT_FAST_REVISION = "3d4d8dd847c5333f1cb0f54d00074ba22b14a5b7"
PI_FAST_REVISION = "ec4d7aa71691cac0b8bed6942be45684db2110f4"
PINNED_FAST_TOKENIZERS = {
    "lerobot/fast-action-tokenizer": LEROBOT_FAST_REVISION,
    "physical-intelligence/fast": PI_FAST_REVISION,
}
_TIME = np.linspace(0, 1, 10)[:, None]
# Two (H=10, D=3) chunks: a periodic trajectory and a constant/linear/quadratic one.
ACTION_CHUNKS = np.stack(
    [
        np.concatenate([np.sin(2 * np.pi * _TIME), np.cos(2 * np.pi * _TIME), 2 * _TIME - 1], axis=1),
        np.concatenate([np.full_like(_TIME, 0.5), -0.25 * _TIME, _TIME**2], axis=1),
    ]
).astype(np.float32)
# Raw FAST token IDs both pinned tokenizers produce for ACTION_CHUNKS.
EXPECTED_TOKENS = [
    [310, 976, 1293, 1522, 913, 1775, 311, 483, 274, 259],
    [344, 283, 314, 310, 309, 500, 1130, 1447],
]


@pytest.fixture(scope="module", params=list(PINNED_FAST_TOKENIZERS))
def fast_backend(request: pytest.FixtureRequest) -> Any:
    pytest.importorskip("transformers")
    pytest.importorskip("scipy")
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(
            request.param, revision=PINNED_FAST_TOKENIZERS[request.param], trust_remote_code=True
        )
    except OSError as error:
        pytest.skip(f"Cannot load {request.param} from the Hub: {error}")


@pytest.fixture
def processor() -> ActionTokenizerProcessorStep:
    processor = object.__new__(ActionTokenizerProcessorStep)
    processor.max_action_tokens = 12
    processor.fast_skip_tokens = 128
    processor.allow_truncation = False
    processor.action_tokenizer = Mock(
        side_effect=lambda chunks: [[int(chunk.sum()) % 100, 2, 3] for chunk in chunks]
    )
    processor._paligemma_tokenizer = SimpleNamespace(
        vocab_size=1000,
        bos_token_id=2,
        encode=lambda text, **kwargs: [10, 11] if text == "Action: " else [12, 1],
    )
    return processor


@pytest.mark.parametrize("shape", [(1, 2), (4, 2), (1, 4, 2), (2, 4, 2)])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
            ],
        ),
    ],
)
def test_action_chunks_preserve_horizon_and_formatting(
    processor: ActionTokenizerProcessorStep, shape: tuple[int, ...], device: str
) -> None:
    actions = torch.arange(torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(shape)
    original = actions.clone()

    result = processor(create_transition(action=actions))
    complementary = result[TransitionKey.COMPLEMENTARY_DATA]
    single_sample = len(shape) == 2
    chunks = actions.unsqueeze(0) if single_sample else actions
    expected_shape = (12,) if single_sample else (shape[0], 12)

    assert result[TransitionKey.ACTION] is actions
    torch.testing.assert_close(actions, original)
    # One backend call per batch, on CPU NumPy data.
    processor.action_tokenizer.assert_called_once()
    backend_input = processor.action_tokenizer.call_args.args[0]
    assert isinstance(backend_input, np.ndarray)
    np.testing.assert_array_equal(backend_input, chunks.cpu().numpy())

    raw_codes = [int(chunk.sum().item()) % 100 for chunk in chunks]
    expected_tokens = torch.tensor(
        [[2, 10, 11, 871 - code, 869, 868, 12, 1, 0, 0, 0, 0] for code in raw_codes], device=device
    )
    expected_mask = torch.tensor([True] * 8 + [False] * 4, device=device).expand(len(chunks), -1)
    expected_code_mask = torch.tensor([False] * 3 + [True] * 3 + [False] * 6, device=device).expand(
        len(chunks), -1
    )
    for key, expected in (
        (ACTION_TOKENS, expected_tokens),
        (ACTION_TOKEN_MASK, expected_mask),
        (ACTION_CODE_TOKEN_MASK, expected_code_mask),
    ):
        actual = complementary[key]
        assert actual.shape == expected_shape
        assert actual.device == actions.device
        torch.testing.assert_close(actual, expected.squeeze(0) if single_sample else expected)


@pytest.mark.parametrize("shape", [(), (6,), (1, 2, 3, 4)])
def test_action_tokenizer_rejects_unsupported_ranks(
    processor: ActionTokenizerProcessorStep, shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match=r"\(H, D\).*\(B, H, D\)"):
        processor._tokenize_action(torch.zeros(shape))
    processor.action_tokenizer.assert_not_called()


def test_pinned_fast_backends_encode_known_tokens(fast_backend: Any) -> None:
    assert fast_backend.scale == 10
    assert fast_backend.min_token == -354
    assert len(fast_backend.bpe_tokenizer) == 2048

    assert fast_backend(ACTION_CHUNKS) == EXPECTED_TOKENS
    assert fast_backend(ACTION_CHUNKS[:1]) == EXPECTED_TOKENS[:1]
    assert fast_backend(ACTION_CHUNKS[0]) == EXPECTED_TOKENS[:1]


@pytest.mark.parametrize("horizon, action_dim", [(10, 3), (50, 7)])
def test_pinned_fast_backends_reconstruct_within_quantization_step(
    fast_backend: Any, horizon: int, action_dim: int
) -> None:
    rng = np.random.default_rng(0)
    actions = (rng.uniform(-1, 1, (4, horizon, action_dim)).cumsum(axis=1) / 10).astype(np.float32)

    decoded = fast_backend.decode(fast_backend(actions), time_horizon=horizon, action_dim=action_dim)

    assert decoded.shape == actions.shape
    # Rounding moves each orthonormal DCT coefficient by at most 0.5 / scale.
    rmse = float(np.sqrt(np.mean((decoded - actions) ** 2)))
    assert rmse <= 0.5 / fast_backend.scale


def test_saved_fast_backend_reloads_without_hub_access(fast_backend: Any, tmp_path: Path) -> None:
    from transformers import AutoProcessor

    processor = object.__new__(ActionTokenizerProcessorStep)
    processor.action_tokenizer = fast_backend

    assert processor.save_artifacts(tmp_path) == {"action_tokenizer_name": "action_tokenizer"}
    saved = tmp_path / "action_tokenizer"
    # The backend class is remote code, so a portable artifact must carry its source file.
    assert (saved / "processing_action_tokenizer.py").is_file()
    reloaded = AutoProcessor.from_pretrained(saved, trust_remote_code=True, local_files_only=True)

    assert (reloaded.scale, reloaded.min_token) == (fast_backend.scale, fast_backend.min_token)
    assert reloaded(ACTION_CHUNKS) == EXPECTED_TOKENS


def test_encode_fast_actions_matches_backend(fast_backend: Any) -> None:
    assert encode_fast_actions(fast_backend, ACTION_CHUNKS) == EXPECTED_TOKENS
    assert encode_fast_actions(fast_backend, torch.from_numpy(ACTION_CHUNKS)) == EXPECTED_TOKENS


@pytest.mark.parametrize("shape", [(10, 3), (1, 1, 10, 3)])
def test_encode_fast_actions_rejects_non_batched_chunks(shape: tuple[int, ...]) -> None:
    tokenizer = Mock()

    with pytest.raises(ValueError, match=r"\(B, H, D\)"):
        encode_fast_actions(tokenizer, np.zeros(shape))
    tokenizer.assert_not_called()


def _dct_coefficients(fast_backend: Any, actions: np.ndarray) -> np.ndarray:
    """Recover the flattened integer FAST coefficients of decoded chunks."""
    from scipy.fft import dct

    coefficients = dct(actions, axis=-2, norm="ortho") * fast_backend.scale
    return np.rint(coefficients).reshape(len(actions), -1)


def test_decode_fast_actions_matches_backend_for_complete_sequences(fast_backend: Any) -> None:
    horizon, action_dim = ACTION_CHUNKS.shape[1:]
    expected = fast_backend.decode(EXPECTED_TOKENS, time_horizon=horizon, action_dim=action_dim)

    for relaxed in (True, False):
        decoded = decode_fast_actions(
            fast_backend, EXPECTED_TOKENS, action_dim, time_horizon=horizon, relaxed=relaxed
        )
        assert decoded.shape == ACTION_CHUNKS.shape
        np.testing.assert_allclose(decoded, expected, rtol=0, atol=1e-12)


def test_relaxed_decoding_zero_fills_missing_trailing_coefficients(fast_backend: Any) -> None:
    horizon, action_dim = ACTION_CHUNKS.shape[1:]
    incomplete = [sequence[:-1] for sequence in EXPECTED_TOKENS]

    full = decode_fast_actions(fast_backend, EXPECTED_TOKENS, action_dim, time_horizon=horizon, relaxed=True)
    partial = decode_fast_actions(fast_backend, incomplete, action_dim, time_horizon=horizon, relaxed=True)
    full_coefficients = _dct_coefficients(fast_backend, full)
    partial_coefficients = _dct_coefficients(fast_backend, partial)

    for index, sequence in enumerate(incomplete):
        kept = len(fast_backend.bpe_tokenizer.decode(sequence))
        assert 0 < kept < horizon * action_dim
        np.testing.assert_array_equal(partial_coefficients[index, :kept], full_coefficients[index, :kept])
        np.testing.assert_array_equal(partial_coefficients[index, kept:], 0)


def test_relaxed_decoding_drops_extra_trailing_coefficients(fast_backend: Any) -> None:
    horizon, action_dim = ACTION_CHUNKS.shape[1:]
    oversized = [sequence + sequence for sequence in EXPECTED_TOKENS]

    np.testing.assert_array_equal(
        decode_fast_actions(fast_backend, oversized, action_dim, time_horizon=horizon, relaxed=True),
        decode_fast_actions(fast_backend, EXPECTED_TOKENS, action_dim, time_horizon=horizon, relaxed=True),
    )


def test_strict_and_backend_decoding_return_zeros_for_incomplete_sequences(fast_backend: Any) -> None:
    horizon, action_dim = ACTION_CHUNKS.shape[1:]
    incomplete = [sequence[:-1] for sequence in EXPECTED_TOKENS]
    zeros = np.zeros(ACTION_CHUNKS.shape)

    np.testing.assert_array_equal(
        decode_fast_actions(fast_backend, incomplete, action_dim, time_horizon=horizon), zeros
    )
    np.testing.assert_array_equal(
        fast_backend.decode(incomplete, time_horizon=horizon, action_dim=action_dim), zeros
    )


@pytest.mark.parametrize("relaxed", [True, False])
def test_out_of_vocabulary_ids_decode_to_zero_actions(fast_backend: Any, relaxed: bool) -> None:
    horizon, action_dim = ACTION_CHUNKS.shape[1:]
    unknown = [[len(fast_backend.bpe_tokenizer) + 10], [-1]]

    decoded = decode_fast_actions(fast_backend, unknown, action_dim, time_horizon=horizon, relaxed=relaxed)

    np.testing.assert_array_equal(decoded, np.zeros((2, horizon, action_dim)))
