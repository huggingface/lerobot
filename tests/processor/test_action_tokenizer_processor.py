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
from lerobot.processor import ActionTokenizerProcessorStep
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
    processor.action_tokenizer = Mock(side_effect=lambda chunk: [[int(chunk.sum().item()) % 100, 2, 3]])
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
    assert processor.action_tokenizer.call_count == len(chunks)
    for call, chunk in zip(processor.action_tokenizer.call_args_list, chunks, strict=True):
        backend_input = call.args[0]
        assert backend_input.shape == (1, *chunk.shape)
        assert backend_input.device.type == "cpu"
        torch.testing.assert_close(backend_input, chunk.unsqueeze(0).cpu())

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
